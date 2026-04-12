"""
Point d'entrée Chainlit — Interface conversationnelle.
Branche l'agent agentic RAG multi-compétences avec la mémoire conversationnelle.
Fonctionnalités : speech-to-text, upload PDF dynamique, monitoring LLMOps.
"""

import io
import logging
import tempfile

import chainlit as cl
from dotenv import load_dotenv

from src.agents.agent import get_prompt_version, get_token_summary, run_agent
from src.memory.memory import add_exchange, get_session_history

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Initialisation de la session ──────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start():
    """Initialise une nouvelle session de chat."""
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    logger.info("Nouvelle session Chainlit : %s", session_id)

    await cl.Message(
        content=(
            "**Assistant Catastrophes Climatiques**\n\n"
            "Je peux :\n"
            "- Répondre à vos questions sur le corpus scientifique "
            "(GIEC, Copernicus, EM-DAT...)\n"
            "- Consulter la météo actuelle, historique ou les prévisions\n"
            "- Effectuer des calculs\n"
            "- Rechercher des actualités sur le web\n"
            "- Croiser toutes ces sources pour une analyse de risque\n"
            "- Accepter des documents PDF pour enrichir le corpus\n"
            "- Comprendre les messages vocaux\n\n"
            "Comment puis-je vous aider ?"
        )
    ).send()


# ── Traitement des messages texte + fichiers ──────────────────────────────


@cl.on_message
async def on_message(message: cl.Message):
    """Traite un message utilisateur (texte, fichiers PDF, ou les deux)."""
    session_id = cl.user_session.get("session_id")
    question = message.content

    # Vérifier si l'utilisateur a uploadé des fichiers PDF
    if message.elements:
        for element in message.elements:
            if element.name.endswith(".pdf"):
                await _integrer_pdf(element)

        # Si le message ne contient que des fichiers, pas de question
        if not question or not question.strip():
            return

    # Récupérer l'historique de la session
    history = get_session_history(session_id)
    chat_history = history.messages

    # Indicateur de chargement
    msg = cl.Message(content="")
    await msg.send()

    # Exécuter l'agent avec l'historique
    answer = run_agent(question, chat_history=chat_history)

    # Mettre à jour la mémoire (avec fenêtre glissante)
    add_exchange(session_id, question, answer)

    # Afficher la réponse
    msg.content = answer
    await msg.update()

    # Afficher le monitoring LLMOps
    tokens = get_token_summary()
    if tokens["total_tokens"] > 0:
        cost = tokens.get("estimated_cost_usd", 0)
        await cl.Message(
            content=(
                f"*Tokens : {tokens['total_tokens']} "
                f"(in: {tokens['total_input_tokens']}, "
                f"out: {tokens['total_output_tokens']}) — "
                f"Coût estimé : ${cost:.4f} — "
                f"Prompt {get_prompt_version()}*"
            ),
            author="system",
        ).send()


# ── Upload PDF dynamique ──────────────────────────────────────────────────


async def _integrer_pdf(element) -> None:
    """
    Intègre un PDF uploadé par l'utilisateur dans le corpus RAG en temps réel.
    Le document est découpé, vectorisé et ajouté au vector store FAISS.
    """
    logger.info("Upload PDF reçu : %s", element.name)

    try:
        from src.config import CHUNK_OVERLAP, CHUNK_SIZE, FAISS_STORE_PATH
        from src.rag.embeddings import charger_vector_store
        from src.rag.loader import decouper_documents

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_huggingface import HuggingFaceEmbeddings

        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(element.content if isinstance(element.content, bytes) else open(element.path, "rb").read())
            tmp_path = tmp.name

        # Charger et découper le PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = decouper_documents(pages)

        # Charger le vector store existant et ajouter les nouveaux chunks
        vector_store = charger_vector_store()
        if vector_store is not None:
            vector_store.add_documents(chunks)
            vector_store.save_local(FAISS_STORE_PATH)
            logger.info(
                "PDF %s intégré : %d pages, %d chunks ajoutés",
                element.name,
                len(pages),
                len(chunks),
            )
            await cl.Message(
                content=(
                    f"**Document intégré** : {element.name}\n"
                    f"- {len(pages)} pages lues\n"
                    f"- {len(chunks)} passages indexés\n"
                    f"- Disponible immédiatement pour les recherches"
                )
            ).send()
        else:
            await cl.Message(
                content="Le vector store n'est pas initialisé. Lancez d'abord embeddings.py."
            ).send()

    except Exception as exc:
        logger.error("Erreur lors de l'intégration du PDF : %s", exc)
        await cl.Message(
            content=f"Erreur lors de l'intégration de {element.name} : {exc}"
        ).send()


# ── Speech-to-text ────────────────────────────────────────────────────────


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Accumule les chunks audio envoyés par l'utilisateur."""
    if chunk.isStart:
        cl.user_session.set("audio_buffer", b"")
        cl.user_session.set("audio_mime", chunk.mimeType)

    buffer = cl.user_session.get("audio_buffer")
    cl.user_session.set("audio_buffer", buffer + chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list):
    """Transcrit l'audio en texte puis traite comme un message normal."""
    audio_buffer = cl.user_session.get("audio_buffer")
    if not audio_buffer:
        return

    logger.info("Audio reçu : %d octets", len(audio_buffer))

    try:
        from faster_whisper import WhisperModel

        # Sauvegarder l'audio temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_buffer)
            tmp_path = tmp.name

        # Transcrire avec faster-whisper (modèle small, CPU)
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path, language="fr")
        text = " ".join(segment.text for segment in segments)

        logger.info("Transcription : %s", text[:100])

        if text.strip():
            # Afficher la transcription
            await cl.Message(
                content=f"*Transcription : {text}*", author="user"
            ).send()

            # Traiter comme un message texte normal
            fake_message = cl.Message(content=text)
            await on_message(fake_message)
        else:
            await cl.Message(content="Aucun texte détecté dans l'audio.").send()

    except ImportError:
        await cl.Message(
            content="Speech-to-text non disponible. Installez : pip install faster-whisper"
        ).send()
    except Exception as exc:
        logger.error("Erreur STT : %s", exc)
        await cl.Message(content=f"Erreur de transcription : {exc}").send()
