"""
Agent Agentic RAG multi-compétences avec orchestration LangGraph.
L'agent décide seul quels outils appeler et dans quel ordre,
y compris le RAG comme outil pour croiser données corpus + météo + web.

Architecture ReAct : Reason → Act → Observe → Repeat → Answer
LLMOps : logging, monitoring tokens, versioning prompt, fallback.
"""

import logging
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agents.tools import ALL_TOOLS
from src.config import TOKEN_TRACKING, TokenCounter, get_fallback_llm, get_llm

load_dotenv()
logger = logging.getLogger(__name__)

# ── Prompt système de l'agent (versionné) ──────────────────────────────────

PROMPT_VERSION = "v1.0"

AGENT_SYSTEM_PROMPT = """Tu es un assistant expert en catastrophes climatiques et environnement.
Tu disposes de 7 outils que tu peux appeler librement et enchaîner dans l'ordre que tu veux :

1. **search_corpus** : chercher dans le corpus de rapports scientifiques (GIEC, Copernicus,
   EM-DAT, NOAA, JRC, WMO). Utilise-le pour toute question sur les catastrophes climatiques,
   les seuils de risque, les données historiques documentées.

2. **get_weather** : météo actuelle d'une ville (OpenMeteo, temps réel).

3. **get_historical_weather** : météo d'une date passée pour une ville donnée.

4. **get_forecast** : prévisions météo des 7 prochains jours pour une ville.

5. **web_search** : recherche web (Tavily en priorité, DuckDuckGo en fallback) pour des informations récentes ou actualités.

6. **calculator** : calculs mathématiques (statistiques, conversions, projections).

7. **send_email** : envoyer un email d'alerte ou de rapport climatique à un destinataire.

Règles :
- Quand on te pose une question sur les catastrophes climatiques, cherche d'abord dans le
  corpus (search_corpus), puis vérifie les conditions météo historiques ou actuelles des lieux
  et dates concernés, et croise les deux pour donner une analyse complète.
- Pour une analyse de risque, consulte les prévisions météo (get_forecast), compare avec les
  seuils critiques du corpus (search_corpus), et réfère-toi aux événements passés similaires.
- Cite toujours tes sources avec [Source: nom_fichier, Page: X] quand tu utilises le corpus.
- Réponds dans la langue de l'utilisateur. Si la question est en français, réponds en français.
  Si elle est en espagnol, réponds en espagnol. Si elle est en anglais, réponds en anglais.
- Structure tes réponses de façon claire et lisible.
- Si la question est une simple conversation (bonjour, merci, etc.), réponds directement
  sans appeler d'outil.
- Retiens les informations données par l'utilisateur (prénom, contexte) pour les réutiliser
  plus tard dans la conversation.
"""

# ── Compteur de tokens global ──────────────────────────────────────────────

token_counter = TokenCounter()

# ── État du graphe ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Construction du LLM (une seule fois) ──────────────────────────────────

_llm_with_tools = None


def _get_llm_with_tools():
    """Retourne le LLM orchestrateur avec les outils bindés (singleton)."""
    global _llm_with_tools
    if _llm_with_tools is None:
        try:
            llm = get_llm("orchestrator")
            logger.info("LLM orchestrateur initialisé (prompt %s)", PROMPT_VERSION)
        except Exception as exc:
            logger.warning("Fallback LLM activé : %s", exc)
            llm = get_fallback_llm()
        _llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


# ── Nœuds du graphe ────────────────────────────────────────────────────────


def call_model(state: AgentState) -> AgentState:
    """Appelle le LLM avec l'historique complet des messages."""
    llm_with_tools = _get_llm_with_tools()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)

    # Monitoring des tokens (si activé)
    if TOKEN_TRACKING and hasattr(response, "usage_metadata") and response.usage_metadata:
        token_counter.log("orchestrator", response)
        logger.debug(
            "Tokens : in=%d, out=%d",
            response.usage_metadata.get("input_tokens", 0),
            response.usage_metadata.get("output_tokens", 0),
        )

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Décide si on doit appeler un outil ou terminer."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info("Outils appelés : %s", tool_names)
        return "tools"
    logger.info("Fin de la boucle ReAct — réponse finale")
    return END


# ── Construction du graphe LangGraph ──────────────────────────────────────


def build_agent_graph() -> StateGraph:
    """Construit le graphe ReAct : call_model ↔ tools → END."""
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "call_model")

    return graph.compile()


# ── Interface publique ─────────────────────────────────────────────────────

_agent_graph = None


def get_agent():
    """Retourne l'agent compilé (singleton)."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
        logger.info("Agent RAG compilé (prompt %s, %d outils)", PROMPT_VERSION, len(ALL_TOOLS))
    return _agent_graph


def run_agent(question: str, chat_history: list[BaseMessage] | None = None) -> str:
    """
    Exécute l'agent sur une question et retourne la réponse finale.

    Args:
        question:     La question de l'utilisateur.
        chat_history: Historique des messages précédents (optionnel).

    Returns:
        Réponse textuelle finale de l'agent.
    """
    logger.info("Question reçue : %s", question[:100])
    agent = get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    result = agent.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            logger.info("Réponse générée (%d caractères)", len(msg.content))
            return msg.content

    logger.warning("L'agent n'a pas produit de réponse finale")
    return "L'agent n'a pas pu produire de réponse."


def get_token_summary() -> dict:
    """Retourne le résumé de consommation de tokens."""
    return token_counter.summary()


def get_prompt_version() -> str:
    """Retourne la version du prompt système."""
    return PROMPT_VERSION


def reset_token_counter():
    """Remet à zéro le compteur de tokens."""
    token_counter.reset()


# ── Test rapide ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    questions = [
        "Quelles catastrophes climatiques ont touché la Méditerranée en 2023 ?",
        "Quel est le risque d'inondation à Marseille cette semaine ?",
        "Quel temps faisait-il à Paris le 1er janvier 2023 ?",
        "Combien font 3+7*2 ?",
        "Bonjour, je suis Kamila",
    ]

    question = sys.argv[1] if len(sys.argv) > 1 else questions[0]
    print(f"\nQuestion : {question}")
    print("-" * 60)
    answer = run_agent(question)
    print(f"Réponse :\n{answer}")
    print(f"\nTokens : {get_token_summary()}")
    print(f"Prompt version : {get_prompt_version()}")
