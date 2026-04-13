"""
Point d'entrée Chainlit — Interface conversationnelle.
Combine le router conditionnel (P3 — Jayson) avec l'agent agentic RAG (P4 — Xia).
- Route "rag" et "chat" : streaming token par token via chain.astream() (pattern P3)
- Route "agent" : agent ReAct 9 outils avec astream_events (pattern P4)
- Fallback : si l'agent ou le streaming échoue, le router compilé P3 prend le relais
Fonctionnalités : streaming, badges de route, sources, STT, upload PDF/DOCX, monitoring LLMOps.
"""

import logging
import os
import tempfile

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.agent import get_agent, get_prompt_version, get_token_summary
from src.memory.memory import add_exchange, get_session_history
from src.router.router import (
    RAG_PROMPT,
    CHAT_PROMPT,
    classify_question,
    router as jayson_router,
    RouterState,
    rag_node,
    chat_node,
)
from src.config import get_llm

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Badges de route ──────────────────────────────────────────────────────

ROUTE_LABELS = {
    "rag": "RAG -- Reponse basee sur le corpus scientifique",
    "agent": "Agent -- Outils externes utilises",
    "chat": "Chat -- Conversation directe",
}

TOOL_BADGES = {
    "search_corpus": "RAG",
    "get_weather": "Agent",
    "get_historical_weather": "Agent",
    "get_forecast": "Agent",
    "web_search": "Agent",
    "calculator": "Agent",
    "send_email": "Agent",
    "predict_risk": "ML",
    "calculer_score_risque": "Scoring",
}


def _detecter_outils_appeles(messages: list) -> tuple[set, list]:
    """Detecte les outils appeles et les sources RAG dans les messages de l'agent."""
    outils = set()
    sources = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                nom = tc["name"]
                if nom in TOOL_BADGES:
                    outils.add(TOOL_BADGES[nom])

        if hasattr(msg, "content") and "[Source:" in str(msg.content):
            for ligne in str(msg.content).split("\n"):
                if "[Source:" in ligne and ligne.strip() not in sources:
                    sources.append(ligne.strip())

    return outils, sources


# ── Initialisation de la session ──────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start():
    """Initialise une nouvelle session de chat."""
    session_id = cl.user_session.get("id", "default")
    cl.user_session.set("session_id", session_id)
    logger.info("Nouvelle session Chainlit : %s", session_id)

    await cl.Message(
        content=(
            "**Assistant Catastrophes Climatiques -- SAEARCH**\n\n"
            "Je peux :\n"
            "- Repondre a vos questions sur le corpus scientifique "
            "(GIEC, Copernicus, EM-DAT...)\n"
            "- Consulter la meteo actuelle, historique ou les previsions\n"
            "- Effectuer des calculs\n"
            "- Rechercher des actualites sur le web\n"
            "- Predire le risque climatique par pays (modele ML)\n"
            "- Calculer un score de risque agrege multi-sources\n"
            "- Croiser toutes ces sources pour une analyse de risque\n"
            "- Envoyer des alertes par email\n"
            "- Accepter des documents PDF/DOCX pour enrichir le corpus\n"
            "- Comprendre les messages vocaux\n\n"
            "Comment puis-je vous aider ?"
        )
    ).send()


# ── Traitement des messages texte + fichiers ──────────────────────────────


@cl.on_message
async def on_message(message: cl.Message):
    """Traite un message utilisateur : routing, streaming, badges, monitoring."""
    session_id = cl.user_session.get("session_id")
    question = message.content

    # Upload PDF ou DOCX
    if message.elements:
        for element in message.elements:
            if element.name.endswith(".pdf"):
                await _integrer_document(element, "pdf")
            elif element.name.endswith(".docx"):
                await _integrer_document(element, "docx")

        if not question or not question.strip():
            return

    # Recuperer l'historique de la session
    history = get_session_history(session_id)
    chat_history = history.messages

    # Etape 1 : classifier la question via le router de Jayson (P3)
    state = RouterState(
        question=question,
        route="",
        answer="",
        sources=[],
        history=chat_history,
    )
    state = classify_question(state)
    route = state["route"]
    logger.info("Route detectee : %s", route)

    # Afficher le badge de route en debut de message (pattern P3)
    badge = ROUTE_LABELS.get(route, "")
    msg = cl.Message(content=f"**{badge}**\n\n")
    await msg.send()

    answer = ""
    sources = []
    agent_outils = set()

    # Etape 2 : traitement selon la route
    if route == "rag":
        # Streaming RAG token par token (pattern P3 — chain.astream)
        answer, sources = await _handle_rag(msg, question)

    elif route == "agent":
        # Agent ReAct 9 outils avec streaming (pattern P4 — astream_events)
        answer, agent_outils, sources = await _handle_agent(msg, question, chat_history)

    else:
        # Streaming Chat token par token (pattern P3 — chain.astream)
        answer = await _handle_chat(msg, question, chat_history)

    # Mettre a jour la memoire
    add_exchange(session_id, question, answer)

    # Footer : sources + monitoring LLMOps
    footer = ""
    if route == "agent" and agent_outils:
        badge_text = " + ".join(sorted(agent_outils))
        footer += f"\n\n---\n**Outils utilises :** {badge_text}"

    if sources:
        footer += "\n\n**Sources :**\n"
        for src_item in sources[:5]:
            footer += f"- {src_item}\n"

    # Monitoring LLMOps
    tokens = get_token_summary()
    if tokens["total_tokens"] > 0:
        cost = tokens.get("estimated_cost_usd", 0)
        footer += (
            f"\n*Tokens : {tokens['total_tokens']} "
            f"(in: {tokens['total_input_tokens']}, "
            f"out: {tokens['total_output_tokens']}) -- "
            f"${cost:.4f} -- {get_prompt_version()}*"
        )

    if footer:
        msg.content = answer + footer
        await msg.update()


# ── Route RAG : streaming via chain.astream (pattern P3) ─────────────────


async def _handle_rag(msg, question: str) -> tuple[str, list]:
    """Route RAG : recherche dans le corpus puis streaming de la reponse."""
    try:
        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import creer_retriever, interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            error_msg = "Vector store non disponible. Lancez d'abord embeddings.py."
            msg.content += error_msg
            await msg.update()
            return error_msg, []

        retriever = creer_retriever(vector_store)
        resultat = interroger_rag(retriever, question)
        contexte = resultat["contexte"]
        documents = resultat["documents"]

        # Sources nettoyees (pattern P3 — os.path.basename)
        sources = []
        for doc in documents:
            source = os.path.basename(doc.metadata.get("source", "inconnu"))
            page = doc.metadata.get("page", "?")
            sources.append(f"[Source: {source}, Page: {page}]")

        # Streaming token par token (pattern P3 — chain.astream)
        llm = get_llm("rag")
        chain = RAG_PROMPT | llm
        answer = ""

        async for chunk in chain.astream({"context": contexte, "question": question}):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{answer}"
                await msg.update()

        logger.info("RAG : %d sources, %d car", len(sources), len(answer))
        return answer, sources

    except Exception as exc:
        logger.error("Erreur RAG streaming : %s — fallback router P3", exc)
        # Fallback : utiliser rag_node de Jayson (P3) sans streaming
        try:
            state = RouterState(
                question=question,
                route="rag",
                answer="",
                sources=[],
                history=[],
            )
            result = rag_node(state)
            answer = result["answer"]
            sources = [f"[Source: {s}]" for s in result.get("sources", [])]
            msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{answer}"
            await msg.update()
            return answer, sources
        except Exception as exc2:
            logger.error("Erreur RAG fallback P3 : %s", exc2)
            error_msg = f"Erreur RAG : {exc2}"
            msg.content += error_msg
            await msg.update()
            return error_msg, []


# ── Route Agent : agent ReAct 9 outils (pattern P4) ──────────────────────


async def _handle_agent(
    msg, question: str, chat_history: list
) -> tuple[str, set, list]:
    """Route Agent : agent ReAct avec streaming astream_events."""
    agent = get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    answer = ""
    all_messages = []

    try:
        async for event in agent.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        answer += "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    else:
                        answer += content
                    msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
                    await msg.update()

            if kind == "on_chain_end" and "messages" in event.get("data", {}).get(
                "output", {}
            ):
                all_messages = event["data"]["output"]["messages"]

    except Exception as exc:
        logger.error("Erreur Agent streaming : %s", exc)

    # Fallback 1 : appel synchrone agent P4
    if not answer:
        try:
            from src.agents.agent import run_agent

            answer = run_agent(question, chat_history=chat_history)
            msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
            await msg.update()
        except Exception as exc:
            logger.warning("Agent P4 echoue : %s — fallback router P3", exc)
            # Fallback 2 : router complet de Jayson (P3)
            try:
                state = RouterState(
                    question=question,
                    route="agent",
                    answer="",
                    sources=[],
                    history=chat_history or [],
                )
                result = jayson_router.invoke(state)
                answer = result["answer"]
                msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
                await msg.update()
            except Exception as exc2:
                logger.error("Erreur Router P3 fallback : %s", exc2)
                answer = f"Erreur Agent : {exc2}"
                msg.content += answer
                await msg.update()

    # Detecter les outils appeles et les sources
    agent_outils, sources = _detecter_outils_appeles(all_messages)
    logger.info("Agent : %d outils, %d car", len(agent_outils), len(answer))
    return answer, agent_outils, sources


# ── Route Chat : streaming via chain.astream (pattern P3) ────────────────


async def _handle_chat(msg, question: str, chat_history: list) -> str:
    """Route Chat : conversation directe avec streaming."""
    llm = get_llm("chat")
    chain = CHAT_PROMPT | llm
    answer = ""

    try:
        async for chunk in chain.astream(
            {"question": question, "history": chat_history}
        ):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                msg.content = f"**{ROUTE_LABELS['chat']}**\n\n{answer}"
                await msg.update()
    except Exception as exc:
        logger.error("Erreur Chat streaming : %s — fallback router P3", exc)
        # Fallback : utiliser chat_node de Jayson (P3) sans streaming
        try:
            state = RouterState(
                question=question,
                route="chat",
                answer="",
                sources=[],
                history=chat_history,
            )
            result = chat_node(state)
            answer = result["answer"]
            msg.content = f"**{ROUTE_LABELS['chat']}**\n\n{answer}"
            await msg.update()
        except Exception as exc2:
            logger.error("Erreur Chat fallback P3 : %s", exc2)
            answer = f"Erreur Chat : {exc2}"
            msg.content += answer
            await msg.update()

    logger.info("Chat : %d car", len(answer))
    return answer


# ── Upload document dynamique (PDF + DOCX) ───────────────────────────────


async def _integrer_document(element, doc_type: str = "pdf") -> None:
    """Integre un PDF ou DOCX uploade dans le corpus RAG en temps reel."""
    logger.info("Upload %s recu : %s", doc_type.upper(), element.name)

    try:
        from src.config import FAISS_STORE_PATH
        from src.rag.embeddings import charger_vector_store
        from src.rag.loader import decouper_documents

        suffix = f".{doc_type}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(
                element.content
                if isinstance(element.content, bytes)
                else open(element.path, "rb").read()
            )
            tmp_path = tmp.name

        if doc_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(tmp_path)
        elif doc_type == "docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(tmp_path)
        else:
            await cl.Message(content=f"Format non supporte : {doc_type}").send()
            return

        pages = loader.load()
        chunks = decouper_documents(pages)

        vector_store = charger_vector_store()
        if vector_store is not None:
            vector_store.add_documents(chunks)
            vector_store.save_local(FAISS_STORE_PATH)
            logger.info(
                "PDF %s integre : %d pages, %d chunks",
                element.name,
                len(pages),
                len(chunks),
            )
            await cl.Message(
                content=(
                    f"**Document integre** : {element.name}\n"
                    f"- {len(pages)} pages lues\n"
                    f"- {len(chunks)} passages indexes\n"
                    f"- Disponible immediatement pour les recherches"
                )
            ).send()
        else:
            await cl.Message(
                content="Vector store non initialise. Lancez d'abord embeddings.py."
            ).send()

    except Exception as exc:
        logger.error("Erreur integration PDF : %s", exc)
        await cl.Message(
            content=f"Erreur lors de l'integration de {element.name} : {exc}"
        ).send()


# ── Speech-to-text (optionnel — depend de la version Chainlit) ────────────


try:

    @cl.on_audio_chunk
    async def on_audio_chunk(chunk):
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

        logger.info("Audio recu : %d octets", len(audio_buffer))

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
                await cl.Message(content="Aucun texte detecte dans l'audio.").send()

        except ImportError:
            await cl.Message(
                content="STT non disponible. Installez : pip install faster-whisper"
            ).send()
        except Exception as exc:
            logger.error("Erreur STT : %s", exc)
            await cl.Message(content=f"Erreur de transcription : {exc}").send()

except (AttributeError, KeyError):
    logger.info("Speech-to-text non supporte par cette version de Chainlit")
