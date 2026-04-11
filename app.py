import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.router.router import router, get_llm, RAG_PROMPT, CHAT_PROMPT

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

    # Étape 1 : classifier la question
    from src.router.router import classify_question, RouterState
    state: RouterState = {
        "question": question,
        "route": "",
        "answer": "",
        "sources": [],
        "history": history
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
            import os

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

            llm = get_llm()
            chain = RAG_PROMPT | llm

            # Streaming token par token
            async for chunk in chain.astream({
                "context": contexte,
                "question": question
            }):
                msg.content += chunk.content
                await msg.update()

            # Ajouter les sources à la fin
            if sources:
                msg.content += "\n\n---\n**📎 Sources :**\n"
                for i, src in enumerate(sources, 1):
                    msg.content += f"- [{i}] {src}\n"
                await msg.update()

            answer = msg.content

        except Exception as e:
            msg.content += f"⚠️ Erreur RAG : {str(e)}"
            await msg.update()
            answer = msg.content

    elif route == "agent":
        try:
            from src.agents.agent import run_agent
            answer_text = run_agent(question, history)
            msg.content += answer_text
            await msg.update()
            answer = answer_text
        except Exception as e:
            msg.content += f"⚠️ Agent non disponible : {str(e)}"
            await msg.update()
            answer = msg.content

    else:  # chat
        llm = get_llm()
        chain = CHAT_PROMPT | llm

        # Streaming token par token
        async for chunk in chain.astream({
            "question": question,
            "history": history
        }):
            msg.content += chunk.content
            await msg.update()

        answer = msg.content

    # Mettre à jour la mémoire
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    cl.user_session.set("history", history)