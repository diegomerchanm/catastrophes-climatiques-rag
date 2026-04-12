"""
Outils pour l'agent LangGraph Agentic RAG.
6 outils :
  1. get_weather       → météo actuelle (OpenMeteo)
  2. get_historical_weather → météo historique (OpenMeteo Archive)
  3. get_forecast      → prévisions météo 7 jours (OpenMeteo)
  4. web_search        → recherche web (DuckDuckGo)
  5. calculator        → calculs mathématiques
  6. search_corpus     → recherche hybride dans le corpus RAG (BM25 + Dense)
"""

import logging
import math
import os
import re

import requests
from duckduckgo_search import DDGS
from langchain_core.tools import tool

from src.config import (
    METEO_FORECAST_DAYS,
    OPENMETEO_BASE_URL,
    OPENMETEO_GEOCODING_URL,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# UTILITAIRE — Géocodage (partagé entre les 3 outils météo)
# ══════════════════════════════════════════════════════════════════

_WMO_CODES: dict[int, str] = {
    0: "Ciel dégagé",
    1: "Principalement dégagé",
    2: "Partiellement nuageux",
    3: "Couvert",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine modérée",
    55: "Bruine dense",
    61: "Pluie légère",
    63: "Pluie modérée",
    65: "Pluie forte",
    71: "Neige légère",
    73: "Neige modérée",
    75: "Neige forte",
    77: "Grains de neige",
    80: "Averses légères",
    81: "Averses modérées",
    82: "Averses violentes",
    85: "Averses de neige",
    86: "Averses de neige fortes",
    95: "Orage",
    96: "Orage avec grêle",
    99: "Orage violent avec grêle",
}


def _geocode(city: str) -> tuple[float, float, str] | None:
    """Géocode une ville via l'API gratuite Open-Meteo geocoding."""
    try:
        resp = requests.get(
            OPENMETEO_GEOCODING_URL,
            params={"name": city, "count": 1, "language": "fr"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r.get("name", city)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# OUTIL 1 — Météo actuelle (OpenMeteo)
# ══════════════════════════════════════════════════════════════════


@tool
def get_weather(city: str) -> str:
    """
    Récupère la météo actuelle d'une ville via l'API OpenMeteo (gratuit, sans clé).

    Args:
        city: Nom de la ville (ex: "Paris", "Lyon", "Bogota")

    Returns:
        Résumé météo : conditions, température, humidité, vent, précipitations.
    """
    logger.info("Appel get_weather pour %s", city)
    geo = _geocode(city)
    if geo is None:
        return f"Impossible de trouver la ville « {city} »."

    lat, lon, city_name = geo

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m,precipitation,surface_pressure"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        }
        resp = requests.get(
            f"{OPENMETEO_BASE_URL}/forecast", params=params, timeout=10
        )
        resp.raise_for_status()
        cur = resp.json()["current"]
    except requests.RequestException as exc:
        return f"Erreur API météo pour « {city_name} » : {exc}"

    wmo = cur.get("weather_code", 0)
    conditions = _WMO_CODES.get(wmo, f"Code WMO {wmo}")

    return (
        f"Météo actuelle à {city_name}\n"
        f"- Conditions    : {conditions}\n"
        f"- Température   : {cur.get('temperature_2m', 'N/A')} °C "
        f"(ressenti {cur.get('apparent_temperature', 'N/A')} °C)\n"
        f"- Humidité      : {cur.get('relative_humidity_2m', 'N/A')} %\n"
        f"- Vent          : {cur.get('wind_speed_10m', 'N/A')} km/h\n"
        f"- Précipitations: {cur.get('precipitation', 0)} mm\n"
        f"- Pression      : {cur.get('surface_pressure', 'N/A')} hPa"
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 2 — Météo historique (OpenMeteo Archive)
# ══════════════════════════════════════════════════════════════════


@tool
def get_historical_weather(city: str, date: str) -> str:
    """
    Récupère la météo historique d'une ville à une date passée via OpenMeteo Archive.

    Args:
        city: Nom de la ville (ex: "Marseille", "Paris")
        date: Date au format YYYY-MM-DD (ex: "2023-01-15")

    Returns:
        Résumé météo du jour : température min/max, précipitations, vent max.
    """
    logger.info("Appel get_historical_weather pour %s le %s", city, date)
    geo = _geocode(city)
    if geo is None:
        return f"Impossible de trouver la ville « {city} »."

    lat, lon, city_name = geo

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max,weather_code"
            ),
            "timezone": "auto",
        }
        resp = requests.get(
            f"{OPENMETEO_BASE_URL}/archive", params=params, timeout=10
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
    except requests.RequestException as exc:
        return f"Erreur API météo historique pour « {city_name} » le {date} : {exc}"

    if not daily.get("time"):
        return f"Aucune donnée disponible pour « {city_name} » le {date}."

    wmo = daily.get("weather_code", [0])[0]
    conditions = _WMO_CODES.get(wmo, f"Code WMO {wmo}")

    return (
        f"Météo historique à {city_name} le {date}\n"
        f"- Conditions       : {conditions}\n"
        f"- Température max  : {daily.get('temperature_2m_max', ['N/A'])[0]} °C\n"
        f"- Température min  : {daily.get('temperature_2m_min', ['N/A'])[0]} °C\n"
        f"- Précipitations   : {daily.get('precipitation_sum', [0])[0]} mm\n"
        f"- Vent max         : {daily.get('wind_speed_10m_max', ['N/A'])[0]} km/h"
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 3 — Prévisions météo 7 jours (OpenMeteo Forecast)
# ══════════════════════════════════════════════════════════════════


@tool
def get_forecast(city: str) -> str:
    """
    Récupère les prévisions météo des 7 prochains jours via OpenMeteo.

    Args:
        city: Nom de la ville (ex: "Lyon", "Marseille")

    Returns:
        Prévisions jour par jour : température min/max, précipitations, vent.
    """
    logger.info("Appel get_forecast pour %s", city)
    geo = _geocode(city)
    if geo is None:
        return f"Impossible de trouver la ville « {city} »."

    lat, lon, city_name = geo

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max,weather_code"
            ),
            "timezone": "auto",
            "forecast_days": METEO_FORECAST_DAYS,
        }
        resp = requests.get(
            f"{OPENMETEO_BASE_URL}/forecast", params=params, timeout=10
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
    except requests.RequestException as exc:
        return f"Erreur API prévisions pour « {city_name} » : {exc}"

    lines = [f"Prévisions météo à {city_name} ({METEO_FORECAST_DAYS} jours)\n"]
    for i, day in enumerate(daily.get("time", [])):
        wmo = daily.get("weather_code", [0])[i]
        conditions = _WMO_CODES.get(wmo, f"Code WMO {wmo}")
        t_max = daily.get("temperature_2m_max", ["N/A"])[i]
        t_min = daily.get("temperature_2m_min", ["N/A"])[i]
        precip = daily.get("precipitation_sum", [0])[i]
        wind = daily.get("wind_speed_10m_max", ["N/A"])[i]
        lines.append(
            f"  {day} : {conditions}, {t_min}-{t_max}°C, "
            f"pluie {precip}mm, vent {wind}km/h"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# OUTIL 4 — Recherche web (Tavily en prio, DuckDuckGo en fallback)
# ══════════════════════════════════════════════════════════════════


def _search_tavily(query: str, max_results: int = 5) -> str | None:
    """Recherche via Tavily (optimisé LLM). Retourne None si indisponible."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        results = response.get("results", [])

        if not results:
            return None

        lines = [f"Résultats Tavily pour : « {query} »\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Sans titre")
            url = r.get("url", "")
            content = r.get("content", "")
            snippet = content[:300].strip() + ("..." if len(content) > 300 else "")
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")

        logger.info("Recherche Tavily réussie pour : %s", query)
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Tavily indisponible, fallback DuckDuckGo : %s", exc)
        return None


def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Recherche via DuckDuckGo (fallback)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Erreur lors de la recherche web : {exc}"

    if not results:
        return f"Aucun résultat trouvé pour : « {query} »"

    lines = [f"Résultats DuckDuckGo pour : « {query} »\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Sans titre")
        href = r.get("href", "")
        body = r.get("body", "")
        snippet = body[:300].strip() + ("..." if len(body) > 300 else "")
        lines.append(f"{i}. {title}\n   {href}\n   {snippet}\n")

    return "\n".join(lines)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Recherche web pour des informations récentes.
    Utilise Tavily (optimisé LLM) en priorité, DuckDuckGo en fallback.

    Args:
        query: Requête de recherche en langage naturel.
        max_results: Nombre maximum de résultats (défaut : 5).

    Returns:
        Liste des résultats avec titre, URL et extrait.
    """
    logger.info("Appel web_search pour : %s", query)

    # Tavily en priorité
    result = _search_tavily(query, max_results)
    if result is not None:
        return result

    # Fallback DuckDuckGo
    return _search_duckduckgo(query, max_results)


# ══════════════════════════════════════════════════════════════════
# OUTIL 5 — Calculatrice (avec garde-fou regex)
# ══════════════════════════════════════════════════════════════════

_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}

_SAFE_FUNC_NAMES = (
    "sqrt|log|log10|log2|sin|cos|tan|asin|acos|atan|"
    "exp|ceil|floor|abs|round|min|max|sum|pow|int|float|pi|inf"
)


@tool
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique et retourne le résultat.

    Args:
        expression: Expression mathématique (ex: "3+7*2", "sqrt(144)", "log(1000)/log(10)")

    Returns:
        Le résultat numérique ou un message d'erreur.
    """
    logger.info("Appel calculator pour : %s", expression)
    expr = expression.strip()

    # Garde-fou : rejeter les caractères non mathématiques
    cleaned = re.sub(_SAFE_FUNC_NAMES, "", expr)
    if re.search(r"[a-zA-Z]{2,}", cleaned):
        return (
            f"Expression non reconnue : « {expression} ». "
            "Utilisez uniquement des opérateurs mathématiques."
        )

    try:
        result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        elif isinstance(result, float):
            formatted = f"{result:.6g}"
        else:
            formatted = str(result)
        return f"{expression} = {formatted}"
    except ZeroDivisionError:
        return "Erreur : division par zéro."
    except (ValueError, TypeError, OverflowError) as exc:
        return f"Erreur de calcul : {exc}"
    except Exception as exc:
        return f"Expression invalide : {exc}"


# ══════════════════════════════════════════════════════════════════
# OUTIL 6 — Recherche hybride dans le corpus RAG (BM25 + Dense)
# ══════════════════════════════════════════════════════════════════

# Singleton pour ne pas recharger le vector store à chaque appel
_cached_vector_store = None
_cached_chunks = None
_cached_hybrid_retriever = None


def _get_hybrid_retriever():
    """Charge le retriever hybride une seule fois (singleton)."""
    global _cached_vector_store, _cached_chunks, _cached_hybrid_retriever

    if _cached_hybrid_retriever is not None:
        return _cached_hybrid_retriever

    from src.rag.embeddings import charger_vector_store
    from src.rag.hybrid_retriever import creer_hybrid_retriever
    from src.rag.loader import charger_et_decouper

    _cached_vector_store = charger_vector_store()
    if _cached_vector_store is None:
        return None

    _cached_chunks = charger_et_decouper("data/raw")
    _cached_hybrid_retriever = creer_hybrid_retriever(
        _cached_vector_store, _cached_chunks
    )
    logger.info("Retriever hybride BM25+Dense chargé (singleton)")
    return _cached_hybrid_retriever


@tool
def search_corpus(question: str) -> str:
    """
    Recherche hybride (BM25 + Dense) dans le corpus de documents climatiques
    (PDFs GIEC, Copernicus, EM-DAT, etc.). Combine recherche sémantique et
    recherche par mots-clés pour une meilleure couverture.

    Args:
        question: Question en langage naturel sur les catastrophes climatiques.

    Returns:
        Passages pertinents avec citations [Source: fichier, Page: X].
    """
    logger.info("Appel search_corpus pour : %s", question)

    retriever = _get_hybrid_retriever()
    if retriever is None:
        # Fallback sur le retriever dense seul
        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import creer_retriever, interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            return "Le vector store n'est pas disponible. Lancez d'abord embeddings.py."
        retriever_dense = creer_retriever(vector_store)
        resultat = interroger_rag(retriever_dense, question)
        return resultat["contexte"]

    from src.rag.retriever import formater_contexte_avec_citations

    docs = retriever.invoke(question)
    return formater_contexte_avec_citations(docs)


# ══════════════════════════════════════════════════════════════════
# Export : liste des outils pour l'agent LangGraph
# ══════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    get_weather,
    get_historical_weather,
    get_forecast,
    web_search,
    calculator,
    search_corpus,
]
