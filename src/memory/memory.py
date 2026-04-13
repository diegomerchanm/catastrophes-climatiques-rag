"""
Mémoire conversationnelle pour le suivi du contexte.
Pattern aligné sur le TP du prof (langchain.ipynb) :
InMemoryChatMessageHistory + RunnableWithMessageHistory.
Fenêtre glissante : tronque les messages anciens au-delà de MEMORY_WINDOW_SIZE.
"""

import logging

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import MEMORY_WINDOW_SIZE

logger = logging.getLogger(__name__)

# ── Stockage des sessions en mémoire ──────────────────────────────────────

_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retourne l'historique d'une session.
    Crée une nouvelle session si elle n'existe pas.
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
        logger.info("Nouvelle session créée : %s", session_id)
    return _store[session_id]


def add_exchange(session_id: str, question: str, reponse: str):
    """
    Ajoute un échange utilisateur/assistant et applique la fenêtre glissante.
    Si l'historique dépasse MEMORY_WINDOW_SIZE paires, supprime les plus anciens.
    """
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(reponse)

    # Fenêtre glissante : garder les N dernières paires (2 messages par paire)
    max_messages = MEMORY_WINDOW_SIZE * 2
    if len(history.messages) > max_messages:
        overflow = len(history.messages) - max_messages
        history.messages = history.messages[overflow:]
        logger.debug(
            "Session %s : troncature de %d messages (fenêtre %d)",
            session_id,
            overflow,
            MEMORY_WINDOW_SIZE,
        )


def wrap_chain_with_memory(chain):
    """
    Enveloppe une chaîne LangChain avec la mémoire conversationnelle.
    Permet au LLM de voir l'historique des échanges précédents.
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def clear_session(session_id: str):
    """Efface l'historique d'une session."""
    if session_id in _store:
        _store[session_id].clear()
        logger.info("Session %s effacée", session_id)


def list_sessions() -> list[str]:
    """Liste toutes les sessions actives."""
    return list(_store.keys())


def get_session_message_count(session_id: str) -> int:
    """Retourne le nombre de messages dans une session."""
    if session_id not in _store:
        return 0
    return len(_store[session_id].messages)
