"""
P2 — src/agents/agent.py
Agent LangGraph (ReAct pattern) qui orchestre les 3 outils :
  - get_weather  : météo en temps réel
  - web_search   : recherche DuckDuckGo
  - calculator   : calculs mathématiques

Le graphe suit le cycle :  call_model → (tool_node | END)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agents.tools import ALL_TOOLS

load_dotenv()

# ── Prompt système de l'agent ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Tu es un assistant spécialisé dans les catastrophes climatiques et \
l'environnement. Tu as accès à trois outils :

1. **get_weather** : obtenir la météo actuelle d'une ville (OpenMeteo, temps réel).
2. **web_search**  : rechercher des informations récentes sur le web (DuckDuckGo).
3. **calculator**  : effectuer des calculs mathématiques.

Règles :
- Utilise **get_weather** quand on te demande la météo ou les conditions climatiques actuelles d'un lieu.
- Utilise **web_search** pour des informations récentes, des actualités ou des données \
  non présentes dans le corpus scientifique.
- Utilise **calculator** pour des calculs numériques (statistiques, conversions, projections).
- Si tu n'as pas besoin d'outil, réponds directement en français, de façon claire et structurée.
- Cite toujours tes sources quand tu utilises un outil.
"""

# ── État du graphe ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Construction du LLM avec tools bindés ─────────────────────────────────

def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY manquante. Ajoutez-la dans votre fichier .env"
        )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=api_key,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Nœuds du graphe ────────────────────────────────────────────────────────

def call_model(state: AgentState) -> AgentState:
    """Appelle le LLM avec l'historique complet des messages."""
    llm_with_tools = _build_llm()

    # Injecter le system prompt si c'est le premier tour
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Décide si on doit appeler un outil ou terminer."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ── Construction du graphe LangGraph ──────────────────────────────────────

def build_agent_graph() -> StateGraph:
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
    graph.add_edge("tools", "call_model")  # retour au LLM après chaque appel d'outil

    return graph.compile()


# ── Interface publique ─────────────────────────────────────────────────────

# Instance partagée (compilée une seule fois)
_agent_graph = None


def get_agent():
    """Retourne l'agent compilé (singleton)."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(question: str, chat_history: list[BaseMessage] | None = None) -> str:
    """
    Exécute l'agent sur une question et retourne la réponse finale en texte.

    Args:
        question:     La question de l'utilisateur.
        chat_history: Historique des messages précédents (optionnel).

    Returns:
        Réponse textuelle finale de l'agent.
    """
    agent = get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    result = agent.invoke({"messages": messages})

    # Récupérer le dernier message AI sans tool_calls
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content

    return "L'agent n'a pas pu produire de réponse."


# ── Test rapide en ligne de commande ──────────────────────────────────────

if __name__ == "__main__":
    import sys

    questions = [
        "Quelle est la météo à Lyon en ce moment ?",
        "Combien font 1.5 * 3.7 + sqrt(81) ?",
        "Quelles sont les dernières nouvelles sur les inondations en Europe ?",
    ]

    question = sys.argv[1] if len(sys.argv) > 1 else questions[0]
    print(f"\n📨 Question : {question}")
    print("─" * 60)
    answer = run_agent(question)
    print(f"🤖 Réponse :\n{answer}")
