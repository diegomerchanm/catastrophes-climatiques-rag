import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.router.router import router

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="🌍 **Assistant Catastrophes Climatiques**\n\nJe peux répondre à vos questions sur le corpus scientifique, consulter la météo, effectuer des calculs, ou simplement discuter.\n\nComment puis-je vous aider ?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    question = message.content

    # Indicateur de chargement
    msg = cl.Message(content="⏳ Analyse en cours...")
    await msg.send()

    # Invoke le router LangGraph
    result = router.invoke({
        "question": question,
        "route": "",
        "answer": "",
        "sources": [],
        "history": history
    })

    answer = result["answer"]
    sources = result["sources"]
    route = result["route"]

    # Badge de route
    route_badge = {
        "rag": "📚 **RAG** — Réponse basée sur le corpus scientifique",
        "agent": "🔧 **Agent** — Outil externe utilisé",
        "chat": "💬 **Chat** — Conversation directe"
    }
    badge = route_badge.get(route, "")

    # Construire la réponse complète
    full_response = f"{badge}\n\n{answer}"
    if sources:
        full_response += "\n\n---\n**📎 Sources :**\n"
        for i, src in enumerate(sources, 1):
            full_response += f"- [{i}] {src}\n"

    msg.content = full_response
    await msg.update()

    # Mettre à jour la mémoire
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    cl.user_session.set("history", history)# Point d'entrée principal de l'application Chainlit
