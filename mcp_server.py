"""
Serveur MCP (Model Context Protocol) — expose les outils du projet RAG
pour qu'ils soient utilisables directement dans Claude Desktop ou tout client MCP.

Lancement : python mcp_server.py
"""

import logging

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.agents.tools import (
    calculator,
    get_forecast,
    get_historical_weather,
    get_weather,
    search_corpus,
    web_search,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Création du serveur MCP ───────────────────────────────────────────────

mcp = FastMCP("RAG Catastrophes Climatiques")


# ── Exposition des outils ─────────────────────────────────────────────────


@mcp.tool()
def meteo_actuelle(ville: str) -> str:
    """Récupère la météo actuelle d'une ville."""
    return get_weather.invoke({"city": ville})


@mcp.tool()
def meteo_historique(ville: str, date: str) -> str:
    """Récupère la météo d'une date passée (format YYYY-MM-DD)."""
    return get_historical_weather.invoke({"city": ville, "date": date})


@mcp.tool()
def previsions_meteo(ville: str) -> str:
    """Récupère les prévisions météo des 7 prochains jours."""
    return get_forecast.invoke({"city": ville})


@mcp.tool()
def recherche_web(requete: str) -> str:
    """Recherche sur le web (Tavily en prio, DuckDuckGo en fallback)."""
    return web_search.invoke({"query": requete})


@mcp.tool()
def calculatrice(expression: str) -> str:
    """Évalue une expression mathématique."""
    return calculator.invoke({"expression": expression})


@mcp.tool()
def recherche_corpus(question: str) -> str:
    """Recherche hybride (BM25 + Dense) dans le corpus climatique (GIEC, Copernicus, EM-DAT)."""
    return search_corpus.invoke({"question": question})


# ── Lancement ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Démarrage du serveur MCP — RAG Catastrophes Climatiques")
    mcp.run()
