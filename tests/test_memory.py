"""Tests pour la mémoire conversationnelle."""


def test_get_session_history_creates_new():
    """Vérifie qu'une nouvelle session est créée automatiquement."""
    from src.memory.memory import get_session_history

    history = get_session_history("test_session_new")
    assert history is not None
    assert len(history.messages) == 0


def test_get_session_history_returns_same():
    """Vérifie qu'on récupère la même session."""
    from src.memory.memory import get_session_history

    h1 = get_session_history("test_session_same")
    h2 = get_session_history("test_session_same")
    assert h1 is h2


def test_add_exchange():
    """Vérifie l'ajout d'échanges via add_exchange."""
    from src.memory.memory import add_exchange, get_session_history

    add_exchange("test_exchange", "Je suis Kamila", "Bonjour Kamila !")
    history = get_session_history("test_exchange")
    assert len(history.messages) == 2
    assert history.messages[0].content == "Je suis Kamila"
    assert history.messages[1].content == "Bonjour Kamila !"


def test_memory_window_truncation():
    """Vérifie que la fenêtre glissante tronque les vieux messages."""
    from src.config import MEMORY_WINDOW_SIZE
    from src.memory.memory import add_exchange, get_session_history

    session_id = "test_truncation"
    # Ajouter plus que MEMORY_WINDOW_SIZE échanges
    for i in range(MEMORY_WINDOW_SIZE + 5):
        add_exchange(session_id, f"question_{i}", f"reponse_{i}")

    history = get_session_history(session_id)
    # Doit contenir exactement MEMORY_WINDOW_SIZE * 2 messages
    assert len(history.messages) == MEMORY_WINDOW_SIZE * 2

    # Le premier message doit être question_5 (les 5 premiers supprimés)
    assert history.messages[0].content == "question_5"


def test_clear_session():
    """Vérifie la suppression d'une session."""
    from src.memory.memory import add_exchange, clear_session, get_session_history

    add_exchange("test_clear", "test", "réponse")
    clear_session("test_clear")
    assert len(get_session_history("test_clear").messages) == 0


def test_list_sessions():
    """Vérifie le listing des sessions."""
    from src.memory.memory import get_session_history, list_sessions

    get_session_history("test_list_a")
    get_session_history("test_list_b")
    sessions = list_sessions()
    assert "test_list_a" in sessions
    assert "test_list_b" in sessions


def test_get_session_message_count():
    """Vérifie le comptage des messages."""
    from src.memory.memory import add_exchange, get_session_message_count

    add_exchange("test_count_v2", "un", "deux")
    add_exchange("test_count_v2", "trois", "quatre")
    assert get_session_message_count("test_count_v2") == 4
    assert get_session_message_count("inexistant") == 0
