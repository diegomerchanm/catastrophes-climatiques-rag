"""
P3 — app.py
Interface Chainlit avec router conditionnel LangGraph.
Mémoire conversationnelle via add_exchange() de memory.py (P4 - Xia).
Streaming token par token sur RAG et Chat.
"""
import os
import logging
import chainlit as cl
from dotenv import load_dotenv
from src.config import get_llm
from src.router.router import classify_question, RouterState, RAG_PROMPT, CHAT_PROMPT
from src.memory.memory import add_exchange, get_session_history

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get("id", "default")
    cl.user_session.set("session_id", session_id)
    await cl.Message(
        content="🌍 **Assistant Catastrophes Climatiques**\n\nJe peux répondre à vos questions sur le corpus scientifique, consulter la météo, effectuer des calculs, ou simplement discuter.\n\nComment puis-je vous aider ?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id", "default")
    question = message.content

    # Récupérer l'historique depuis la mémoire de Xia
    history = get_session_history(session_id)
    history_messages = history.messages

    # Étape 1 : classifier la question
    state: RouterState = {
        "question": question,
        "route": "",
        "answer": "",
        "sources": [],
        "history": history_messages
    }
    state = classify_question(state)
    route = state["route"]

    # Badge de route
    route_badge = {
        "rag": "📚 **RAG** — Réponse basée sur le corpus scientifique",
        "agent": "🔧 **Agent** — Outil externe utilisé",
        "chat": "💬 **Chat** — Conversation directe"
    }
    badge = route_badge.get(route, "")

    # Étape 2 : streaming selon la route
    msg = cl.Message(content=f"{badge}\n\n")
    await msg.send()

    if route == "rag":
        try:
            from src.rag.embeddings import charger_vector_store
            from src.rag.retriever import creer_retriever, interroger_rag

            vector_store = charger_vector_store()
            if vector_store is None:
                msg.content += "⚠️ Vector store non disponible. Lancez d'abord embeddings.py."
                await msg.update()
                return

            retriever = creer_retriever(vector_store)
            resultat = interroger_rag(retriever, question)
            contexte = resultat["contexte"]
            documents = resultat["documents"]

            sources = []
            for doc in documents:
                source = os.path.basename(doc.metadata.get("source", "inconnu"))
                page = doc.metadata.get("page", "?")
                sources.append(f"{source} (page {page})")

            llm = get_llm("rag")
            chain = RAG_PROMPT | llm

            # Streaming token par token
            answer_text = ""
            async for chunk in chain.astream({
                "context": contexte,
                "question": question
            }):
                msg.content += chunk.content
                answer_text += chunk.content
                await msg.update()

            # Ajouter les sources à la fin
            if sources:
                msg.content += "\n\n---\n**📎 Sources :**\n"
                for i, src in enumerate(sources, 1):
                    msg.content += f"- [{i}] {src}\n"
                await msg.update()

            # Sauvegarder dans la mémoire de Xia
            add_exchange(session_id, question, answer_text)

        except Exception as e:
            logger.error("Erreur RAG : %s", str(e))
            msg.content += f"⚠️ Erreur RAG : {str(e)}"
            await msg.update()
            add_exchange(session_id, question, str(e))

    elif route == "agent":
        try:
            from src.agents.agent import run_agent
            answer_text = run_agent(question, history_messages)
            msg.content += answer_text
            await msg.update()
            add_exchange(session_id, question, answer_text)
        except Exception as e:
            logger.error("Erreur Agent : %s", str(e))
            error_msg = f"⚠️ Agent non disponible : {str(e)}"
            msg.content += error_msg
            await msg.update()
            add_exchange(session_id, question, error_msg)

    else:  # chat
        llm = get_llm("chat")
        chain = CHAT_PROMPT | llm

        # Streaming token par token
        answer_text = ""
        async for chunk in chain.astream({
            "question": question,
            "history": history_messages
        }):
            msg.content += chunk.content
            answer_text += chunk.content
            await msg.update()

        # Sauvegarder dans la mémoire de Xia
        add_exchange(session_id, question, answer_text)