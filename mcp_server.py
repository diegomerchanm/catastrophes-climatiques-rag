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
    calculer_score_risque,
    get_forecast,
    get_historical_weather,
    get_weather,
    list_corpus,
    predict_risk,
    predict_risk_by_type,
    search_corpus,
    send_email,
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


@mcp.tool()
def envoyer_email(destinataire: str, sujet: str, contenu: str) -> str:
    """Envoie un email d'alerte ou de rapport climatique."""
    return send_email.invoke(
        {
            "destinataire": destinataire,
            "sujet": sujet,
            "contenu": contenu,
        }
    )


@mcp.tool()
def prediction_risque(pays: str) -> str:
    """Prédit le niveau d'exposition aux catastrophes climatiques pour un pays (agrégat + détail par type climatique, modèle ML EM-DAT multi-type)."""
    return predict_risk.invoke({"country": pays})


@mcp.tool()
def prediction_risque_par_type(pays: str, type_catastrophe: str) -> str:
    """Prédit l'impact d'un type spécifique (drought, flood, extreme_weather, extreme_temperature, wildfire) pour un pays en 2030."""
    return predict_risk_by_type.invoke(
        {"country": pays, "disaster_type": type_catastrophe}
    )


@mcp.tool()
def score_risque(
    precipitation_prevue_mm: float,
    seuil_critique_mm: float,
    risk_level_ml: str,
    a_precedent_historique: bool,
    corpus_mentionne_risque: bool,
) -> str:
    """Calcule un score de risque agrégé (0-1) croisant météo, ML, corpus et historique."""
    return calculer_score_risque.invoke(
        {
            "precipitation_prevue_mm": precipitation_prevue_mm,
            "seuil_critique_mm": seuil_critique_mm,
            "risk_level_ml": risk_level_ml,
            "a_precedent_historique": a_precedent_historique,
            "corpus_mentionne_risque": corpus_mentionne_risque,
        }
    )


@mcp.tool()
def inventaire_corpus() -> str:
    """Liste tous les documents du corpus SAEARCH avec taille et nombre de pages."""
    return list_corpus.invoke({})


# ── Lancement ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Démarrage du serveur MCP — RAG Catastrophes Climatiques")
    mcp.run()
