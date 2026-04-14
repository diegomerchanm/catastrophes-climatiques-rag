"""
Agent Agentic RAG multi-compétences avec orchestration LangGraph.
L'agent décide seul quels outils appeler et dans quel ordre,
y compris le RAG comme outil pour croiser données corpus + météo + web.

Architecture ReAct : Reason → Act → Observe → Repeat → Answer
LLMOps : logging, monitoring tokens, versioning prompt, fallback.
"""

import logging
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agents.tools import ALL_TOOLS
from src.config import (
    TOKEN_TRACKING,
    TokenCounter,
    get_fallback_llm,
    get_llm,
    get_ollama_fallback,
)
from src.prompts.agent_prompts import CURRENT_VERSION, get_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ── Prompt système de l'agent (versionné — src/prompts/agent_prompts.py) ──

PROMPT_VERSION = CURRENT_VERSION
AGENT_SYSTEM_PROMPT = get_prompt(PROMPT_VERSION)

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
            logger.warning("Fallback Haiku activé : %s", exc)
            try:
                llm = get_fallback_llm()
            except Exception as exc2:
                logger.warning("Fallback Ollama activé : %s", exc2)
                llm = get_ollama_fallback()
        _llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


# ── Nœuds du graphe ────────────────────────────────────────────────────────


def call_model(state: AgentState) -> AgentState:
    """Appelle le LLM avec l'historique complet des messages."""
    llm_with_tools = _get_llm_with_tools()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        now = datetime.now().strftime("%A %d %B %Y, %H:%M (heure locale)")

        prompt_with_time = (
            AGENT_SYSTEM_PROMPT + f"\n\nDate et heure actuelles : {now}. "
            "Adapte tes reponses au contexte temporel "
            "(jour/nuit, saison, jour de la semaine). "
            "Si l'utilisateur te donne son email ou son prenom, retiens-le "
            "pour la suite de la conversation. Quand il dit 'envoie-moi', "
            "utilise l'email qu'il t'a donne precedemment."
        )
        messages = [SystemMessage(content=prompt_with_time)] + messages

    response = llm_with_tools.invoke(messages)

    # Monitoring des tokens (si activé)
    if (
        TOKEN_TRACKING
        and hasattr(response, "usage_metadata")
        and response.usage_metadata
    ):
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
        logger.info(
            "Agent RAG compilé (prompt %s, %d outils)", PROMPT_VERSION, len(ALL_TOOLS)
        )
    return _agent_graph


def run_agent(question: str, chat_history: list[BaseMessage] | None = None) -> str:
    """
    Exécute l'agent sur une question et retourne la réponse finale.
    Tracke la prédiction dans MLflow si disponible.

    Args:
        question:     La question de l'utilisateur.
        chat_history: Historique des messages précédents (optionnel).

    Returns:
        Réponse textuelle finale de l'agent.
    """
    import time

    logger.info("Question reçue : %s", question[:100])
    agent = get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    t0 = time.time()
    result = agent.invoke({"messages": messages})
    duree = round(time.time() - t0, 2)

    answer = "L'agent n'a pas pu produire de réponse."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    # Détecter les outils appelés
    outils_appeles = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                outils_appeles.append(tc["name"])

    logger.info(
        "Réponse générée (%d car, %d outils, %.2fs)",
        len(answer),
        len(outils_appeles),
        duree,
    )

    # Tracking MLflow
    _log_mlflow(question, answer, outils_appeles, duree)

    return answer


def _git_commit_short() -> str:
    """Retourne le SHA court du commit courant, ou var env GITHUB_SHA, ou 'unknown'."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        import os as _os

        return _os.getenv("GITHUB_SHA", "unknown")[:7]


def _log_mlflow(question: str, answer: str, outils: list, duree: float) -> None:
    """Tracke une prédiction dans MLflow si disponible."""
    try:
        import os
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)

        env_label = os.getenv("ENV_LABEL") or (
            "prod" if os.getenv("AZURE_CONTAINER_APP") else "local"
        )
        mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT_AGENT") or "rag-catastrophes-climatiques"
        )

        with mlflow.start_run(run_name=question[:50], nested=True):
            # Tags (traçabilite code + env)
            mlflow.set_tags(
                {
                    "git_commit": _git_commit_short(),
                    "env": env_label,
                    "prompt_version": PROMPT_VERSION,
                }
            )

            # Paramètres
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            mlflow.log_param("question", question[:250])
            mlflow.log_param("nb_outils_appeles", len(outils))
            mlflow.log_param("outils", ", ".join(outils) if outils else "aucun")
            mlflow.log_param("orchestrator_model", "sonnet")

            # Métriques
            mlflow.log_metric("duree_s", duree)
            mlflow.log_metric("longueur_reponse", len(answer))
            mlflow.log_metric("nb_outils", len(outils))

            tokens = token_counter.summary()
            mlflow.log_metric("total_tokens", tokens["total_tokens"])
            mlflow.log_metric("estimated_cost_usd", tokens.get("estimated_cost_usd", 0))

            logger.debug("MLflow run enregistré pour : %s", question[:50])

    except ImportError:
        logger.debug("MLflow non installé, tracking désactivé")
    except Exception as exc:
        logger.debug("MLflow tracking échoué : %s", exc)


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
