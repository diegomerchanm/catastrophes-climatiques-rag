"""Tests pour la configuration centralisée."""

import os
import pytest


def test_agent_configs_complete():
    """Vérifie que tous les types d'agents sont configurés."""
    from src.config import AGENT_CONFIGS

    types_attendus = ["orchestrator", "rag", "meteo", "web", "analyst", "chat"]
    for agent_type in types_attendus:
        assert agent_type in AGENT_CONFIGS, f"Agent '{agent_type}' manquant"


def test_agent_configs_have_required_keys():
    """Vérifie que chaque config d'agent contient les clés requises."""
    from src.config import AGENT_CONFIGS

    cles_requises = ["model", "temperature", "max_tokens", "description"]
    for agent_type, config in AGENT_CONFIGS.items():
        for cle in cles_requises:
            assert cle in config, f"Clé '{cle}' manquante pour l'agent '{agent_type}'"


def test_temperature_ranges():
    """Vérifie que les températures sont dans les bornes valides."""
    from src.config import AGENT_CONFIGS

    for agent_type, config in AGENT_CONFIGS.items():
        temp = config["temperature"]
        assert 0 <= temp <= 1, (
            f"Température {temp} hors bornes pour '{agent_type}'"
        )


def test_max_tokens_positive():
    """Vérifie que les max_tokens sont positifs."""
    from src.config import AGENT_CONFIGS

    for agent_type, config in AGENT_CONFIGS.items():
        assert config["max_tokens"] > 0, (
            f"max_tokens négatif pour '{agent_type}'"
        )


def test_get_llm_invalid_type():
    """Vérifie que get_llm lève une erreur pour un type inconnu."""
    from src.config import get_llm

    with pytest.raises(ValueError):
        get_llm("type_inexistant")


def test_get_llm_missing_api_key():
    """Vérifie que get_llm lève une erreur sans clé API."""
    from src.config import get_llm

    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(EnvironmentError):
            get_llm("chat")
    finally:
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_token_counter():
    """Vérifie le fonctionnement du compteur de tokens."""
    from src.config import TokenCounter

    counter = TokenCounter()
    assert counter.total_input == 0
    assert counter.total_output == 0

    class FakeResponse:
        usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    counter.log("rag", FakeResponse())
    summary = counter.summary()
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50
    assert summary["total_tokens"] == 150
    assert summary["calls_by_agent"]["rag"] == 1
    assert summary["tokens_by_agent"]["rag"]["input"] == 100
    assert summary["tokens_by_agent"]["rag"]["output"] == 50

    counter.reset()
    assert counter.total_input == 0
    assert counter.tokens_by_agent == {}


def test_token_counter_cost_estimation():
    """Vérifie l'estimation de coût."""
    from src.config import TokenCounter

    counter = TokenCounter()

    class FakeResponse:
        usage_metadata = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    counter.log("orchestrator", FakeResponse())
    summary = counter.summary()
    # Haiku : input $0.25/M + output $1.25/M = $1.50
    assert summary["estimated_cost_usd"] == pytest.approx(1.50, rel=0.01)


def test_model_pricing_complete():
    """Vérifie que tous les modèles ont un tarif."""
    from src.config import MODEL_HAIKU, MODEL_OPUS, MODEL_PRICING, MODEL_SONNET

    assert MODEL_HAIKU in MODEL_PRICING
    assert MODEL_SONNET in MODEL_PRICING
    assert MODEL_OPUS in MODEL_PRICING


def test_retriever_params():
    """Vérifie les paramètres du retriever."""
    from src.config import RETRIEVER_FETCH_K, RETRIEVER_K

    assert RETRIEVER_K > 0
    assert RETRIEVER_FETCH_K >= RETRIEVER_K


def test_memory_window_size():
    """Vérifie la taille de la fenêtre mémoire."""
    from src.config import MEMORY_WINDOW_SIZE

    assert MEMORY_WINDOW_SIZE > 0
    assert MEMORY_WINDOW_SIZE == 20
