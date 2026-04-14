"""
Configuration centralisée du projet RAG Catastrophes Climatiques.
Hyperparamètres, spécialisation des LLM par agent, monitoring LLMOps.
"""

import os
from langchain_anthropic import ChatAnthropic

# ── Modèles Anthropic disponibles ─────────────────────────────────────────

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# ── Tarifs Anthropic ($ / 1M tokens) pour estimation de coût ──────────────

MODEL_PRICING = {
    MODEL_HAIKU: {"input": 0.25, "output": 1.25},
    MODEL_SONNET: {"input": 3.00, "output": 15.00},
    MODEL_OPUS: {"input": 15.00, "output": 75.00},
}

# ── Configuration par type d'agent ────────────────────────────────────────

AGENT_CONFIGS = {
    "orchestrator": {
        "model": MODEL_SONNET,
        "temperature": 0,
        "max_tokens": 2048,
        "description": "Orchestre les outils et génère la réponse finale",
    },
    "rag": {
        "model": MODEL_SONNET,
        "temperature": 0.2,
        "max_tokens": 2048,
        "description": "Répond en citant les sources du corpus climatique",
    },
    "meteo": {
        "model": MODEL_HAIKU,
        "temperature": 0,
        "max_tokens": 1024,
        "description": "Interprète les données météo (actuel/historique/prévisions)",
    },
    "web": {
        "model": MODEL_HAIKU,
        "temperature": 0.2,
        "max_tokens": 1024,
        "description": "Synthétise les actualités issues de la recherche web",
    },
    "analyst": {
        "model": MODEL_OPUS,
        "temperature": 0.5,
        "max_tokens": 4096,
        "description": "Croise les données pour produire l'analyse de risque",
    },
    "chat": {
        "model": MODEL_HAIKU,
        "temperature": 0.7,
        "max_tokens": 1024,
        "description": "Conversation simple, salutations, hors-sujet",
    },
}

# ── Modèle de fallback ────────────────────────────────────────────────────

FALLBACK_MODEL = MODEL_HAIKU

# ── Paramètres du chunking ────────────────────────────────────────────────

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# ── Paramètres des embeddings ─────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_STORE_PATH = "faiss_store"

# ── Paramètres du retriever ───────────────────────────────────────────────

RETRIEVER_K = 8
RETRIEVER_FETCH_K = 20
RETRIEVER_SEARCH_TYPE = "mmr"

# ── Recherche hybride BM25 + Dense ────────────────────────────────────────

BM25_WEIGHT = 0.5
DENSE_WEIGHT = 0.5

# ── Mémoire conversationnelle ─────────────────────────────────────────────

MEMORY_WINDOW_SIZE = 20

# ── Monitoring LLMOps ─────────────────────────────────────────────────────

TOKEN_TRACKING = True


class TokenCounter:
    """Compteur de tokens par agent, par session, avec estimation de coût."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.calls_by_agent = {}
        self.tokens_by_agent = {}

    def log(self, agent_type: str, response):
        usage = response.usage_metadata
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        self.total_input += input_tokens
        self.total_output += output_tokens
        self.calls_by_agent[agent_type] = self.calls_by_agent.get(agent_type, 0) + 1

        if agent_type not in self.tokens_by_agent:
            self.tokens_by_agent[agent_type] = {"input": 0, "output": 0}
        self.tokens_by_agent[agent_type]["input"] += input_tokens
        self.tokens_by_agent[agent_type]["output"] += output_tokens

    def estimate_cost(self) -> float:
        """Estime le coût total en $ basé sur les tarifs Anthropic."""
        total_cost = 0.0
        for agent_type, tokens in self.tokens_by_agent.items():
            model = AGENT_CONFIGS.get(agent_type, {}).get("model", MODEL_HAIKU)
            pricing = MODEL_PRICING.get(model, MODEL_PRICING[MODEL_HAIKU])
            cost_input = (tokens["input"] / 1_000_000) * pricing["input"]
            cost_output = (tokens["output"] / 1_000_000) * pricing["output"]
            total_cost += cost_input + cost_output
        return total_cost

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "calls_by_agent": self.calls_by_agent,
            "tokens_by_agent": self.tokens_by_agent,
            "estimated_cost_usd": self.estimate_cost(),
        }

    def reset(self):
        self.total_input = 0
        self.total_output = 0
        self.calls_by_agent = {}
        self.tokens_by_agent = {}


# ── API météo OpenMeteo ───────────────────────────────────────────────────

OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1"
OPENMETEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
METEO_FORECAST_DAYS = 7

# ── Chainlit ──────────────────────────────────────────────────────────────

CHAINLIT_HOST = "0.0.0.0"
CHAINLIT_PORT = 8000


# ── Construction du LLM ───────────────────────────────────────────────────


def get_llm(agent_type: str) -> ChatAnthropic:
    """Retourne un LLM configuré pour un type d'agent donné."""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(
            f"Type d'agent inconnu : '{agent_type}'. "
            f"Disponibles : {list(AGENT_CONFIGS.keys())}"
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante. Ajoutez-la dans votre fichier .env"
        )

    config = AGENT_CONFIGS[agent_type]
    return ChatAnthropic(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


def get_fallback_llm() -> ChatAnthropic:
    """LLM de secours Anthropic (Haiku) en cas d'échec du modèle principal."""
    return ChatAnthropic(
        model=FALLBACK_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )


def get_ollama_fallback():
    """
    LLM de secours local via Ollama (open source, gratuit, hors ligne).
    Nécessite Ollama installé + modèle téléchargé : ollama pull mistral
    Utilisé quand l'API Anthropic est indisponible.
    """
    try:
        import langchain_community.llms as community_llms

        ollama_cls = getattr(community_llms, "Ollama")
        return ollama_cls(
            model="mistral",
            temperature=0.2,
            num_predict=1024,
        )
    except (ImportError, AttributeError) as exc:
        raise EnvironmentError(
            "Ollama non disponible. Installez langchain-community et Ollama : "
            "pip install langchain-community && curl -fsSL https://ollama.com/install.sh | sh && ollama pull mistral"
        ) from exc
