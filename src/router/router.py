"""
P3 — src/router/router.py
Router conditionnel LangGraph — 3 branches : rag / agent / chat
Utilise get_llm() de config.py (migration Groq → Claude)
"""

import logging
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from src.config import get_llm

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────


class RouterState(TypedDict):
    question: str
    route: str
    answer: str
    sources: list
    history: list


# ── Node 1 : Classifier ───────────────────────────────────────────────────────

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un routeur de questions. Analyse la question et reponds UNIQUEMENT par un seul mot :
- "rag" si la question porte sur les catastrophes climatiques, le rechauffement, les evenements meteo extremes, le GIEC, Copernicus, EM-DAT, le corpus, les documents scientifiques, les rapports, ou tout sujet environnemental
- "agent" si la question demande une ACTION : meteo, previsions, recherche web, calcul, envoi d'email, prediction de risque, score de risque, analyse, ou toute tache necessitant un outil
- "chat" UNIQUEMENT pour les salutations simples (bonjour, merci, au revoir) ou les questions sur l'identite du systeme

EN CAS DE DOUTE, reponds "agent".

Exemples :
- "Bonjour" -> chat
- "Qui es-tu ?" -> chat
- "Quelle est la meteo a Paris ?" -> agent
- "Calcule 3+7" -> agent
- "Envoie un email" -> agent
- "Donne-moi les news" -> agent
- "Quel risque a Nice ?" -> agent
- "Que dit le GIEC sur les inondations ?" -> rag
- "Quelles catastrophes en 2023 ?" -> rag
- "Liste les documents du corpus" -> rag
- "Combien de documents" -> rag
- "Quels rapports sont disponibles" -> rag

Ne reponds que par un seul mot : rag, agent, ou chat""",
        ),
        ("human", "{question}"),
    ]
)


AGENT_KEYWORDS = [
    "email", "mail", "envoie", "envoyer", "meteo", "météo", "temps",
    "prevision", "prévision", "forecast", "calcul", "combien",
    "recherche", "cherche", "news", "actualit", "web",
    "risque", "predict", "score", "alerte",
    "pdf", "inventaire", "fichier",
    "temperature", "température", "moyenne", "historique",
    "precipit", "vent", "humidite", "humidité", "pression",
]


RAG_KEYWORDS = [
    "corpus", "document", "rapport", "giec", "ipcc", "copernicus",
    "em-dat", "emdat", "catastrophe", "climatique", "inondation",
    "secheresse", "incendie", "cyclone", "ouragan",
]


# Mots-cles qui forcent agent meme si un mot RAG est present
AGENT_PRIORITY = [
    "recherche", "cherche", "news", "actualit", "web",
    "email", "mail", "envoie", "envoyer", "calcul", "combien",
    "meteo", "météo", "prevision", "prévision", "forecast",
    "temperature", "température", "historique",
]


def classify_question(state: RouterState) -> RouterState:
    question_lower = state["question"].lower()

    # Mots-cles agent prioritaires (recherche web, email, meteo...)
    for kw in AGENT_PRIORITY:
        if kw in question_lower:
            logger.info("Mot-cle prioritaire '%s' detecte -> route agent", kw)
            return {**state, "route": "agent"}

    # Detection rapide par mots-cles -> force "rag"
    for kw in RAG_KEYWORDS:
        if kw in question_lower:
            logger.info("Mot-cle '%s' detecte -> route rag", kw)
            return {**state, "route": "rag"}

    # Detection rapide par mots-cles -> force "agent"
    for kw in AGENT_KEYWORDS:
        if kw in question_lower:
            logger.info("Mot-cle '%s' detecte -> route agent", kw)
            return {**state, "route": "agent"}

    llm = get_llm("orchestrator")
    chain = CLASSIFIER_PROMPT | llm
    result = chain.invoke({"question": state["question"]})
    route = result.content.strip().lower()
    if route not in ["rag", "agent", "chat"]:
        route = "chat"
    logger.info("Question routée vers : %s", route)
    return {**state, "route": route}


def route_decision(state: RouterState) -> str:
    return state["route"]


# ── Node 2 : RAG ──────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant expert en catastrophes climatiques.
Réponds à la question en utilisant UNIQUEMENT le contexte fourni.
Cite tes sources avec [Source: nom_fichier, page X].
Si le contexte est insuffisant, dis-le clairement en expliquant ce que tu as trouvé quand même.

Contexte :
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def rag_node(state: RouterState) -> RouterState:
    try:
        import os
        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import creer_retriever, interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            return {
                **state,
                "answer": "⚠️ Le vector store n'est pas disponible. Lancez d'abord : python -m src.rag.embeddings",
                "sources": [],
            }

        retriever = creer_retriever(vector_store)
        resultat = interroger_rag(retriever, state["question"])

        contexte = resultat["contexte"]
        documents = resultat["documents"]

        # Nettoyer les chemins — afficher uniquement le nom du fichier
        sources = []
        for doc in documents:
            source = os.path.basename(doc.metadata.get("source", "inconnu"))
            page = doc.metadata.get("page", "?")
            sources.append(f"{source} (page {page})")

        llm = get_llm("rag")
        chain = RAG_PROMPT | llm
        result = chain.invoke({"context": contexte, "question": state["question"]})

        logger.info("RAG : %d sources trouvées", len(sources))
        return {**state, "answer": result.content, "sources": sources}

    except Exception as e:
        logger.error("Erreur RAG : %s", str(e))
        return {**state, "answer": f"⚠️ Erreur RAG : {str(e)}", "sources": []}


# ── Node 3 : Agent ────────────────────────────────────────────────────────────


def agent_node(state: RouterState) -> RouterState:
    try:
        from src.agents.agent import run_agent

        answer = run_agent(state["question"], state.get("history", []))
        logger.info("Agent : réponse générée")
    except Exception as e:
        logger.error("Erreur Agent : %s", str(e))
        answer = f"⚠️ Agent non disponible : {str(e)}"
    return {**state, "answer": answer, "sources": []}


# ── Node 4 : Chat direct ──────────────────────────────────────────────────────

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es DooMax, l'IA du systeme SAEARCH. Tu ne t'appelles jamais Claude.
Tu ne mets JAMAIS d'emojis dans tes reponses. Ton ton est professionnel et concis.
Tu te souviens de ce qui a ete dit dans la conversation.
Reponds toujours en francais.
Date et heure actuelles : {current_time}. Adapte tes reponses au contexte temporel.""",
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)


def chat_node(state: RouterState) -> RouterState:
    from datetime import datetime

    llm = get_llm("chat")
    chain = CHAT_PROMPT | llm
    now = datetime.now().strftime("%A %d %B %Y, %H:%M (heure locale)")
    result = chain.invoke(
        {
            "question": state["question"],
            "history": state.get("history", []),
            "current_time": now,
        }
    )
    logger.info("Chat : réponse directe générée")
    return {**state, "answer": result.content, "sources": []}


# ── Compilation du graph ──────────────────────────────────────────────────────


def build_router():
    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_question)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_decision, {"rag": "rag", "agent": "agent", "chat": "chat"}
    )
    graph.add_edge("rag", END)
    graph.add_edge("agent", END)
    graph.add_edge("chat", END)

    return graph.compile()


router = build_router()
