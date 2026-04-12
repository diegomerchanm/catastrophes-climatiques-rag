"""
Point d'entrée Chainlit — Interface conversationnelle.
Branche l'agent agentic RAG multi-compétences avec la mémoire conversationnelle.
Fonctionnalités : streaming, badges de route, sources, STT, upload PDF, monitoring LLMOps.
"""

import logging
import tempfile

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from src.agents.agent import get_agent, get_prompt_version, get_token_summary
from src.memory.memory import add_exchange, get_session_history

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Détection de route ────────────────────────────────────────────────────

ROUTE_BADGES = {
    "search_corpus": "RAG",
    "get_weather": "Agent",
    "get_historical_weather": "Agent",
    "get_forecast": "Agent",
    "web_search": "Agent",
    "calculator": "Agent",
    "send_email": "Agent",
}


def _detecter_routes(messages: list) -> tuple[set, list]:
    """Détecte les outils appelés et les sources RAG dans les messages."""
    routes = set()
    sources = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                nom = tc["name"]
                if nom in ROUTE_BADGES:
                    routes.add(ROUTE_BADGES[nom])

        # Extraire les sources RAG
        if hasattr(msg, "content") and "[Source:" in str(msg.content):
            for ligne in str(msg.content).split("\n"):
                if "[Source:" in ligne and ligne.strip() not in sources:
                    sources.append(ligne.strip())

    if not routes:
        routes.add("Chat")

    return routes, sources


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
            "- Envoyer des alertes par email\n"
            "- Accepter des documents PDF pour enrichir le corpus\n"
            "- Comprendre les messages vocaux\n\n"
            "Comment puis-je vous aider ?"
        )
    ).send()


# ── Traitement des messages texte + fichiers ──────────────────────────────


@cl.on_message
async def on_message(message: cl.Message):
    """Traite un message utilisateur avec streaming et badges de route."""
    session_id = cl.user_session.get("session_id")
    question = message.content

    # Vérifier si l'utilisateur a uploadé des fichiers PDF
    if message.elements:
        for element in message.elements:
            if element.name.endswith(".pdf"):
                await _integrer_pdf(element)

        if not question or not question.strip():
            return

    # Récupérer l'historique de la session
    history = get_session_history(session_id)
    chat_history = history.messages

    # Message de streaming
    msg = cl.Message(content="")
    await msg.send()

    # Exécuter l'agent avec streaming
    from langchain_core.messages import HumanMessage

    agent = get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    answer = ""
    all_messages = []

    async for event in agent.astream_events(
        {"messages": messages}, version="v2"
    ):
        kind = event["event"]

        # Streaming des tokens de la réponse finale
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                msg.content = answer
                await msg.update()

        # Capturer tous les messages pour détecter les routes
        if kind == "on_chain_end" and "messages" in event.get("data", {}).get("output", {}):
            all_messages = event["data"]["output"]["messages"]

    # Si pas de streaming, fallback sur le résultat complet
    if not answer:
        from src.agents.agent import run_agent
        answer = run_agent(question, chat_history=chat_history)
        msg.content = answer
        await msg.update()

    # Mettre à jour la mémoire
    add_exchange(session_id, question, answer)

    # Détecter les routes et sources
    routes, sources = _detecter_routes(all_messages)

    # Badge de route
    badge_text = " + ".join(sorted(routes))
    badge_map = {
        "RAG": "RAG — Réponse basée sur le corpus scientifique",
        "Agent": "Agent — Outil externe utilisé",
        "Chat": "Chat — Conversation directe",
    }
    badges = [badge_map.get(r, r) for r in sorted(routes)]

    # Sources RAG en fin de message
    footer = f"\n\n---\n**Route :** {badge_text}"
    if sources:
        footer += "\n\n**Sources :**\n"
        for src in sources[:5]:
            footer += f"- {src}\n"

    # Monitoring LLMOps
    tokens = get_token_summary()
    if tokens["total_tokens"] > 0:
        cost = tokens.get("estimated_cost_usd", 0)
        footer += (
            f"\n*Tokens : {tokens['total_tokens']} "
            f"(in: {tokens['total_input_tokens']}, "
            f"out: {tokens['total_output_tokens']}) — "
            f"${cost:.4f} — {get_prompt_version()}*"
        )

    msg.content = answer + footer
    await msg.update()


# ── Upload PDF dynamique ──────────────────────────────────────────────────


async def _integrer_pdf(element) -> None:
    """Intègre un PDF uploadé dans le corpus RAG en temps réel."""
    logger.info("Upload PDF reçu : %s", element.name)

    try:
        from src.config import FAISS_STORE_PATH
        from src.rag.embeddings import charger_vector_store
        from src.rag.loader import decouper_documents

        from langchain_community.document_loaders import PyPDFLoader

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(
                element.content
                if isinstance(element.content, bytes)
                else open(element.path, "rb").read()
            )
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = decouper_documents(pages)

        vector_store = charger_vector_store()
        if vector_store is not None:
            vector_store.add_documents(chunks)
            vector_store.save_local(FAISS_STORE_PATH)
            logger.info(
                "PDF %s intégré : %d pages, %d chunks",
                element.name, len(pages), len(chunks),
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
                content="Vector store non initialisé. Lancez d'abord embeddings.py."
            ).send()

    except Exception as exc:
        logger.error("Erreur intégration PDF : %s", exc)
        await cl.Message(
            content=f"Erreur lors de l'intégration de {element.name} : {exc}"
        ).send()


# ── Speech-to-text ────────────────────────────────────────────────────────


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Accumule les chunks audio."""
    if chunk.isStart:
        cl.user_session.set("audio_buffer", b"")
        cl.user_session.set("audio_mime", chunk.mimeType)

    buffer = cl.user_session.get("audio_buffer")
    cl.user_session.set("audio_buffer", buffer + chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list):
    """Transcrit l'audio puis traite comme un message texte."""
    audio_buffer = cl.user_session.get("audio_buffer")
    if not audio_buffer:
        return

    logger.info("Audio reçu : %d octets", len(audio_buffer))

    try:
        from faster_whisper import WhisperModel

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_buffer)
            tmp_path = tmp.name

        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path, language="fr")
        text = " ".join(segment.text for segment in segments)

        logger.info("Transcription : %s", text[:100])

        if text.strip():
            await cl.Message(
                content=f"*Transcription : {text}*", author="user"
            ).send()
            fake_message = cl.Message(content=text)
            await on_message(fake_message)
        else:
            await cl.Message(content="Aucun texte détecté dans l'audio.").send()

    except ImportError:
        await cl.Message(
            content="STT non disponible. Installez : pip install faster-whisper"
        ).send()
    except Exception as exc:
        logger.error("Erreur STT : %s", exc)
        await cl.Message(content=f"Erreur de transcription : {exc}").send()
