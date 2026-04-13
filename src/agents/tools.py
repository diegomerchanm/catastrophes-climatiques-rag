"""
Outils pour l'agent LangGraph Agentic RAG.
7 outils :
  1. get_weather       → météo actuelle (OpenMeteo)
  2. get_historical_weather → météo historique (OpenMeteo Archive)
  3. get_forecast      → prévisions météo 7 jours (OpenMeteo)
  4. web_search        → recherche web (Tavily + DuckDuckGo fallback)
  5. calculator        → calculs mathématiques
  6. search_corpus     → recherche hybride dans le corpus RAG (BM25 + Dense)
  7. send_email        → envoi d'email d'alerte climatique
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
        resp = requests.get(f"{OPENMETEO_BASE_URL}/forecast", params=params, timeout=10)
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
        resp = requests.get(f"{OPENMETEO_BASE_URL}/archive", params=params, timeout=10)
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
        resp = requests.get(f"{OPENMETEO_BASE_URL}/forecast", params=params, timeout=10)
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
# OUTIL 7 — Envoi d'email (alertes climatiques)
# ══════════════════════════════════════════════════════════════════


@tool
def send_email(destinataire: str, sujet: str, contenu: str) -> str:
    """
    Envoie un email d'alerte ou de rapport climatique.

    Args:
        destinataire: Adresse email du destinataire.
        sujet: Sujet du mail.
        contenu: Corps du mail en texte.

    Returns:
        Confirmation d'envoi ou message d'erreur.
    """
    logger.info("Appel send_email vers %s : %s", destinataire, sujet)

    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_APP_PASSWORD")

    if not email_address or not email_password:
        return (
            "Email non configuré. Ajoutez EMAIL_ADDRESS et EMAIL_APP_PASSWORD dans .env"
        )

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg["From"] = email_address
        msg["To"] = destinataire
        msg["Subject"] = sujet
        msg.attach(MIMEText(contenu, "plain", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)

        logger.info("Email envoyé à %s", destinataire)
        return f"Email envoyé à {destinataire} : {sujet}"
    except Exception as exc:
        logger.error("Erreur envoi email : %s", exc)
        return f"Erreur lors de l'envoi : {exc}"


# ══════════════════════════════════════════════════════════════════
# OUTIL 8 — Prédiction ML du risque catastrophe (modèle EM-DAT)
# ══════════════════════════════════════════════════════════════════

_cached_predictions = None


def _load_predictions() -> dict | None:
    """Charge les prédictions d'impact 2030 pré-calculées (NB10)."""
    global _cached_predictions
    if _cached_predictions is not None:
        return _cached_predictions

    predictions_path = os.path.join("outputs", "NB10_predictions_2030.csv")
    if not os.path.exists(predictions_path):
        logger.warning("Prédictions ML non trouvées : %s", predictions_path)
        return None

    import csv

    _cached_predictions = {}
    with open(predictions_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _cached_predictions[row["country"].lower()] = {
                "country": row["country"],
                "continent": row["continent"],
                "deaths_pred": float(row["deaths_pred"]),
                "risk_pred": row["risk_pred"],
            }
    logger.info("Prédictions ML chargées : %d pays", len(_cached_predictions))
    return _cached_predictions


@tool
def predict_risk(country: str) -> str:
    """
    Prédit le niveau d'exposition aux catastrophes climatiques pour un pays,
    basé sur un modèle ML entraîné sur les données EM-DAT (1900-2020).
    Retourne l'impact humain moyen annuel prédit pour la décennie 2030
    et le niveau de risque.

    Args:
        country: Nom du pays en anglais (ex: "France", "India", "Bangladesh")

    Returns:
        Prédiction : impact humain moyen/an, niveau de risque, continent.
    """
    logger.info("Appel predict_risk pour %s", country)
    predictions = _load_predictions()

    if predictions is None:
        return (
            "Modèle ML non disponible. Lancez d'abord le notebook NB10 "
            "pour générer les prédictions (outputs/NB10_predictions_2030.csv)."
        )

    key = country.strip().lower()
    if key not in predictions:
        # Recherche approximative
        matches = [k for k in predictions if key in k or k in key]
        if matches:
            key = matches[0]
        else:
            pays_dispo = ", ".join(
                sorted(set(p["country"] for p in predictions.values()))[:20]
            )
            return (
                f"Pays « {country} » non trouvé dans les prédictions ML. "
                f"Exemples disponibles : {pays_dispo}..."
            )

    pred = predictions[key]
    niveau = pred["risk_pred"]
    victimes = pred["deaths_pred"]

    return (
        f"Prédiction ML 2030 pour {pred['country']} ({pred['continent']})\n"
        f"- Victimes moyennes annuelles prédites : {victimes:.0f}\n"
        f"- Niveau d'exposition : {niveau}\n"
        f"- Source : modèle entraîné sur EM-DAT (1900-2020), "
        f"pipeline sklearn (régression + classification)\n"
        f"- Limite : prédiction statistique basée sur les tendances passées, "
        f"ne tient pas compte des politiques d'adaptation"
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 9 — Score de risque agrégé multi-sources
# ══════════════════════════════════════════════════════════════════

_RISK_LEVELS = {
    "Aucun": 0.0,
    "Faible": 0.25,
    "Modéré": 0.50,
    "Élevé": 0.75,
    "Critique": 1.0,
}

_POIDS = {
    "meteo": 0.35,
    "corpus": 0.25,
    "ml_predict": 0.25,
    "historique": 0.15,
}


@tool
def calculer_score_risque(
    precipitation_prevue_mm: float,
    seuil_critique_mm: float,
    risk_level_ml: str,
    a_precedent_historique: bool,
    corpus_mentionne_risque: bool,
) -> str:
    """
    Calcule un score de risque agrégé (0-1) en croisant 4 sources :
    météo (précipitations vs seuil), ML (prédiction EM-DAT),
    corpus (GIEC mentionne un risque), historique (précédent connu).

    Args:
        precipitation_prevue_mm: Précipitations prévues en mm.
        seuil_critique_mm: Seuil critique du GIEC en mm.
        risk_level_ml: Niveau de risque ML (Aucun/Faible/Modéré/Élevé/Critique).
        a_precedent_historique: True si un événement similaire a eu lieu dans le passé.
        corpus_mentionne_risque: True si le corpus GIEC mentionne un risque pour la zone.

    Returns:
        Score agrégé avec détail par source et décision GO/NO-GO.
    """
    logger.info(
        "Calcul score risque : %.0fmm vs %.0fmm seuil, ML=%s",
        precipitation_prevue_mm,
        seuil_critique_mm,
        risk_level_ml,
    )

    # Score météo : ratio précipitations / seuil (plafonné à 1.0)
    if seuil_critique_mm > 0:
        score_meteo = min(precipitation_prevue_mm / seuil_critique_mm, 1.0)
    else:
        score_meteo = 0.5

    # Score ML
    score_ml = _RISK_LEVELS.get(risk_level_ml, 0.5)

    # Score corpus
    score_corpus = 0.8 if corpus_mentionne_risque else 0.2

    # Score historique
    score_historique = 0.9 if a_precedent_historique else 0.1

    # Agrégation pondérée
    score_final = (
        _POIDS["meteo"] * score_meteo
        + _POIDS["ml_predict"] * score_ml
        + _POIDS["corpus"] * score_corpus
        + _POIDS["historique"] * score_historique
    )

    # Décision
    if score_final >= 0.75:
        decision = "CRITIQUE — Alerte immédiate recommandée"
    elif score_final >= 0.50:
        decision = "ÉLEVÉ — Vigilance renforcée"
    elif score_final >= 0.25:
        decision = "MODÉRÉ — Surveillance normale"
    else:
        decision = "FAIBLE — Pas d'action requise"

    return (
        f"Score de risque agrégé : {score_final:.2f} / 1.00\n"
        f"\nDétail par source :\n"
        f"  - Météo      ({_POIDS['meteo']:.0%}) : {score_meteo:.2f} "
        f"({precipitation_prevue_mm:.0f}mm / {seuil_critique_mm:.0f}mm seuil)\n"
        f"  - ML EM-DAT  ({_POIDS['ml_predict']:.0%}) : {score_ml:.2f} "
        f"(niveau {risk_level_ml})\n"
        f"  - Corpus GIEC ({_POIDS['corpus']:.0%}) : {score_corpus:.2f} "
        f"({'risque mentionné' if corpus_mentionne_risque else 'pas de mention'})\n"
        f"  - Historique  ({_POIDS['historique']:.0%}) : {score_historique:.2f} "
        f"({'précédent connu' if a_precedent_historique else 'pas de précédent'})\n"
        f"\nDécision : {decision}"
    )


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
    send_email,
    predict_risk,
    calculer_score_risque,
]
