"""
P2 — src/agents/tools.py
Trois outils pour l'agent LangGraph :
  1. get_weather      → météo en temps réel via OpenMeteo (gratuit, sans clé)
  2. web_search       → recherche web via DuckDuckGo
  3. calculator       → calculs mathématiques simples
"""

import math
import re
import requests
from duckduckgo_search import DDGS
from langchain_core.tools import tool


# ══════════════════════════════════════════════════════════════════
# OUTIL 1 — Météo en temps réel (OpenMeteo, 0 inscription requise)
# ══════════════════════════════════════════════════════════════════

# Codes météo WMO → description humaine
_WMO_CODES: dict[int, str] = {
    0: "Ciel dégagé ☀️",
    1: "Principalement dégagé 🌤️",
    2: "Partiellement nuageux ⛅",
    3: "Couvert ☁️",
    45: "Brouillard 🌫️",
    48: "Brouillard givrant 🌫️",
    51: "Bruine légère 🌦️",
    53: "Bruine modérée 🌦️",
    55: "Bruine dense 🌧️",
    61: "Pluie légère 🌧️",
    63: "Pluie modérée 🌧️",
    65: "Pluie forte 🌧️",
    71: "Neige légère 🌨️",
    73: "Neige modérée 🌨️",
    75: "Neige forte ❄️",
    77: "Grains de neige ❄️",
    80: "Averses légères 🌦️",
    81: "Averses modérées 🌧️",
    82: "Averses violentes ⛈️",
    85: "Averses de neige 🌨️",
    86: "Averses de neige fortes ❄️",
    95: "Orage ⛈️",
    96: "Orage avec grêle ⛈️🧊",
    99: "Orage violent avec grêle ⛈️🧊",
}


def _geocode(city: str) -> tuple[float, float, str] | None:
    """Géocode une ville via l'API gratuite Open-Meteo geocoding."""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        resp = requests.get(url, params={"name": city, "count": 1, "language": "fr"}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r.get("name", city)
    except Exception:
        return None


@tool
def get_weather(city: str) -> str:
    """
    Récupère la météo actuelle d'une ville via l'API OpenMeteo (gratuit, sans clé API).
    Utilise le géocodage automatique, donc accepte n'importe quelle ville du monde.

    Args:
        city: Nom de la ville (ex: "Paris", "Lyon", "Dakar", "New York")

    Returns:
        Résumé météo textuel : conditions, température, humidité, vent, précipitations.
    """
    geo = _geocode(city)
    if geo is None:
        return f"❌ Impossible de trouver la ville « {city} ». Vérifiez l'orthographe."

    lat, lon, city_name = geo

    try:
        url = "https://api.open-meteo.com/v1/forecast"
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
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        cur = resp.json()["current"]
    except requests.RequestException as exc:
        return f"❌ Erreur API météo pour « {city_name} » : {exc}"

    wmo = cur.get("weather_code", 0)
    conditions = _WMO_CODES.get(wmo, f"Conditions inconnues (code WMO {wmo})")

    return (
        f"🌍 Météo actuelle à {city_name}\n"
        f"• Conditions    : {conditions}\n"
        f"• Température   : {cur.get('temperature_2m', 'N/A')} °C "
        f"(ressenti {cur.get('apparent_temperature', 'N/A')} °C)\n"
        f"• Humidité      : {cur.get('relative_humidity_2m', 'N/A')} %\n"
        f"• Vent          : {cur.get('wind_speed_10m', 'N/A')} km/h\n"
        f"• Précipitations: {cur.get('precipitation', 0)} mm\n"
        f"• Pression      : {cur.get('surface_pressure', 'N/A')} hPa"
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 2 — Recherche web (DuckDuckGo, sans clé API)
# ══════════════════════════════════════════════════════════════════

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Effectue une recherche web via DuckDuckGo et retourne les résultats pertinents.
    Utile pour obtenir des informations récentes non présentes dans le corpus PDF.

    Args:
        query: La requête de recherche en langage naturel.
        max_results: Nombre maximum de résultats à retourner (défaut : 5).

    Returns:
        Liste des résultats formatée avec titre, URL et extrait.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"❌ Erreur lors de la recherche web : {exc}"

    if not results:
        return f"Aucun résultat trouvé pour : « {query} »"

    lines = [f"🔍 Résultats de recherche pour : « {query} »\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Sans titre")
        href = r.get("href", "")
        body = r.get("body", "")
        # Tronquer l'extrait pour ne pas surcharger le contexte
        snippet = body[:300].strip() + ("…" if len(body) > 300 else "")
        lines.append(f"{i}. **{title}**\n   {href}\n   {snippet}\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# OUTIL 3 — Calculatrice (expressions mathématiques sûres)
# ══════════════════════════════════════════════════════════════════

# Contexte autorisé pour eval() — uniquement math + builtins sûrs
_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "int": int, "float": float,
    # Fonctions du module math
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e, "inf": math.inf,
}


@tool
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique et retourne le résultat.
    Supporte les opérations de base (+, -, *, /, **), les fonctions trigonométriques,
    logarithmes, racine carrée, etc.

    Args:
        expression: Expression mathématique en texte
                    (ex: "2 + 2", "sqrt(144)", "log(1000) / log(10)", "15 * 1.08").

    Returns:
        Le résultat numérique ou un message d'erreur explicite.
    """
    # Nettoyage minimal : supprimer les espaces superflus
    expr = expression.strip()

    # Garde-fou : rejeter les caractères clairement non mathématiques
    if re.search(r"[a-zA-Z]{2,}", expr.replace("sqrt", "").replace("log", "")
                 .replace("sin", "").replace("cos", "").replace("tan", "")
                 .replace("asin", "").replace("acos", "").replace("atan", "")
                 .replace("exp", "").replace("ceil", "").replace("floor", "")
                 .replace("abs", "").replace("round", "").replace("min", "")
                 .replace("max", "").replace("sum", "").replace("pow", "")
                 .replace("int", "").replace("float", "").replace("inf", "")
                 .replace("pi", "").replace("log2", "").replace("log10", "")):
        return f"❌ Expression non reconnue : « {expression} ». Utilisez uniquement des opérateurs mathématiques."

    try:
        result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307
        # Formatage : entier si possible, sinon 6 décimales max
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        elif isinstance(result, float):
            formatted = f"{result:.6g}"
        else:
            formatted = str(result)
        return f"🧮 {expression} = **{formatted}**"
    except ZeroDivisionError:
        return "❌ Erreur : division par zéro."
    except (ValueError, TypeError, OverflowError) as exc:
        return f"❌ Erreur de calcul : {exc}"
    except Exception as exc:
        return f"❌ Expression invalide : {exc}"


# ══════════════════════════════════════════════════════════════════
# Export : liste des tools pour l'agent LangGraph
# ══════════════════════════════════════════════════════════════════

ALL_TOOLS = [get_weather, web_search, calculator]
