import os
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# ── State ─────────────────────────────────────────────────────────────────────

class RouterState(TypedDict):
    question: str
    route: str
    answer: str
    sources: list
    history: list

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ── Node 1 : Classifier ───────────────────────────────────────────────────────

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un routeur de questions. Analyse la question et réponds UNIQUEMENT par un seul mot :
- "rag" si la question porte sur les catastrophes climatiques, le réchauffement, les événements météo extrêmes, ou tout sujet scientifique environnemental
- "agent" si la question demande la météo actuelle, une recherche web, ou un calcul mathématique
- "chat" pour toute autre conversation générale (salutations, opinions, questions hors-sujet)

Ne réponds que par : rag, agent, ou chat"""),
    ("human", "{question}")
])

def classify_question(state: RouterState) -> RouterState:
    llm = get_llm()
    chain = CLASSIFIER_PROMPT | llm
    result = chain.invoke({"question": state["question"]})
    route = result.content.strip().lower()
    if route not in ["rag", "agent", "chat"]:
        route = "chat"
    return {**state, "route": route}

def route_decision(state: RouterState) -> str:
    return state["route"]

# ── Node 2 : RAG ──────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant expert en catastrophes climatiques.
Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
Cite tes sources avec [Source: nom_fichier, page X].
Si le contexte est insuffisant, dis-le clairement.

Contexte :
{context}"""),
    ("human", "{question}")
])

def rag_node(state: RouterState) -> RouterState:
    try:
        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import creer_retriever, interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            return {**state, "answer": "Le vector store n'est pas disponible. Lancez d'abord embeddings.py.", "sources": []}

        retriever = creer_retriever(vector_store)
        resultat = interroger_rag(retriever, state["question"])

        contexte = resultat["contexte"]
        documents = resultat["documents"]

        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "inconnu")
            page = doc.metadata.get("page", "?")
            sources.append(f"{source} (page {page})")

        llm = get_llm()
        chain = RAG_PROMPT | llm
        result = chain.invoke({
            "context": contexte,
            "question": state["question"]
        })
        return {**state, "answer": result.content, "sources": sources}

    except Exception as e:
        return {**state, "answer": f"Erreur RAG : {str(e)}", "sources": []}

# ── Node 3 : Agent ────────────────────────────────────────────────────────────

def agent_node(state: RouterState) -> RouterState:
    try:
        from src.agents.agent import run_agent
        answer = run_agent(state["question"], state.get("history", []))
    except Exception as e:
        answer = f"[Agent non disponible] {str(e)}"
    return {**state, "answer": answer, "sources": []}

# ── Node 4 : Chat direct ──────────────────────────────────────────────────────

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant conversationnel sympathique et concis."),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

def chat_node(state: RouterState) -> RouterState:
    llm = get_llm()
    chain = CHAT_PROMPT | llm
    result = chain.invoke({
        "question": state["question"],
        "history": state.get("history", [])
    })
    return {**state, "answer": result.content, "sources": []}

# ── Compilation du graph ──────────────────────────────────────────────────────

def build_router():
    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_question)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_decision, {
        "rag": "rag",
        "agent": "agent",
        "chat": "chat"
    })
    graph.add_edge("rag", END)
    graph.add_edge("agent", END)
    graph.add_edge("chat", END)

    return graph.compile()

router = build_router()