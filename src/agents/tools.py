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
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "auto",
        }
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        daily = data["daily"]
        hourly = data.get("hourly", {})
    except requests.RequestException as exc:
        return f"Erreur API météo historique pour « {city_name} » le {date} : {exc}"

    if not daily.get("time"):
        return f"Aucune donnée disponible pour « {city_name} » le {date}."

    wmo = daily.get("weather_code", [0])[0]
    conditions = _WMO_CODES.get(wmo, f"Code WMO {wmo}")

    result = (
        f"Météo historique à {city_name} le {date}\n"
        f"- Conditions       : {conditions}\n"
        f"- Température max  : {daily.get('temperature_2m_max', ['N/A'])[0]} °C\n"
        f"- Température min  : {daily.get('temperature_2m_min', ['N/A'])[0]} °C\n"
        f"- Précipitations   : {daily.get('precipitation_sum', [0])[0]} mm\n"
        f"- Vent max         : {daily.get('wind_speed_10m_max', ['N/A'])[0]} km/h"
    )

    # Ajouter les donnees horaires si disponibles
    temps = hourly.get("temperature_2m", [])
    if temps:
        result += "\n\nDonnees horaires (temperature) :"
        hours = hourly.get("time", [])
        for i, (h, t) in enumerate(zip(hours, temps)):
            heure = h.split("T")[1] if "T" in h else f"{i:02d}:00"
            result += f"\n  {heure} : {t} °C"

    return result


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

    _cached_vector_store = charger_vector_store()
    if _cached_vector_store is None:
        return None

    # Extraction des chunks directement depuis le FAISS docstore.
    # Avantages : (1) marche en container sans data/raw/, (2) plus rapide
    # (pas de re-parsing PDFs), (3) garantit coherence chunks BM25 = chunks Dense.
    try:
        _cached_chunks = list(_cached_vector_store.docstore._dict.values())
        logger.info(
            "Chunks extraits du FAISS docstore : %d (sans recharger PDFs)",
            len(_cached_chunks),
        )
    except Exception as exc:
        logger.warning("Echec extraction chunks FAISS, fallback PDFs : %s", exc)
        from src.rag.loader import charger_et_decouper

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

    # Resoudre le nom en email si besoin
    if "@" not in destinataire:
        resolved = TEAM_DIRECTORY.get(destinataire.strip().lower())
        if resolved:
            destinataire = resolved

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
# OUTIL 7b — Envoi d'email groupé (alertes climatiques)
# ══════════════════════════════════════════════════════════════════

# Emails equipe charges depuis env (jamais dans le code public).
# Format attendu dans .env :
#   TEAM_DIRECTORY_JSON={"kamila":"xxx@mail.com","xia":"xxx@mail.com",...}
# TEAM_EMAILS est deduit automatiquement des valeurs du dict.
import json as _json

_directory_json = os.getenv("TEAM_DIRECTORY_JSON", "{}")
try:
    TEAM_DIRECTORY = _json.loads(_directory_json)
except _json.JSONDecodeError:
    logger.warning("TEAM_DIRECTORY_JSON mal formate, fallback vide")
    TEAM_DIRECTORY = {}

TEAM_EMAILS = list(TEAM_DIRECTORY.values())


@tool
def send_bulk_email(destinataires: str, sujet: str, contenu: str) -> str:
    """
    Envoie un email a plusieurs destinataires en meme temps.
    Utilise cet outil pour les alertes climatiques groupees,
    les rapports envoyes a une equipe, ou les notifications de masse.

    Args:
        destinataires: Adresses email separees par des virgules (ex: "a@test.com, b@test.com").
        sujet: Sujet du mail.
        contenu: Corps du mail en texte.

    Returns:
        Confirmation d'envoi avec le nombre de destinataires.
    """
    logger.info("Appel send_bulk_email : %s", sujet)

    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_APP_PASSWORD")

    if not email_address or not email_password:
        return (
            "Email non configure. Ajoutez EMAIL_ADDRESS et EMAIL_APP_PASSWORD dans .env"
        )

    # Parser les destinataires
    dest_lower = destinataires.lower()
    if any(
        kw in dest_lower for kw in ["equipe", "team", "tous", "tout le monde", "all"]
    ):
        emails = TEAM_EMAILS
    else:
        # Resoudre les noms en emails via le repertoire
        emails = []
        for part in destinataires.split(","):
            part = part.strip()
            if "@" in part:
                emails.append(part)
            else:
                # Chercher par nom dans le repertoire
                resolved = TEAM_DIRECTORY.get(part.lower())
                if resolved:
                    emails.append(resolved)
    if not emails:
        noms_dispo = ", ".join(TEAM_DIRECTORY.keys())
        return f"Aucune adresse trouvee. Noms disponibles : {noms_dispo}"

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        envoyes = 0
        erreurs = []

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_address, email_password)

            for dest in emails:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = email_address
                    msg["To"] = dest
                    msg["Subject"] = sujet
                    msg.attach(MIMEText(contenu, "plain", "utf-8"))
                    server.send_message(msg)
                    envoyes += 1
                    logger.info("Email envoye a %s", dest)
                except Exception as exc:
                    erreurs.append(f"{dest}: {exc}")
                    logger.error("Erreur envoi a %s : %s", dest, exc)

        result = f"Email envoye a {envoyes}/{len(emails)} destinataires : {sujet}"
        if erreurs:
            result += f"\nErreurs : {'; '.join(erreurs)}"
        return result
    except Exception as exc:
        logger.error("Erreur envoi email groupe : %s", exc)
        return f"Erreur lors de l'envoi groupe : {exc}"


# ══════════════════════════════════════════════════════════════════
# OUTIL 7c — Programmation d'email differe (scheduling)
# ══════════════════════════════════════════════════════════════════

_scheduler = None


def _get_scheduler():
    """Retourne l'instance BackgroundScheduler (lazy init + persistance SQLite)."""
    global _scheduler
    if _scheduler is None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

            # Persistance : jobs survivent au restart (SQLite local dedie)
            jobstores = {
                "default": SQLAlchemyJobStore(url="sqlite:///scheduler_jobs.db")
            }
            # misfire_grace_time 1h : rattrape les jobs manques au redemarrage
            job_defaults = {"misfire_grace_time": 3600, "coalesce": True}
            _scheduler = BackgroundScheduler(
                jobstores=jobstores,
                job_defaults=job_defaults,
                timezone="Europe/Paris",
            )
            _scheduler.start()
            logger.info(
                "APScheduler demarre (timezone Europe/Paris, "
                "persistance SQLite scheduler_jobs.db)"
            )
        except ImportError as e:
            logger.warning("APScheduler/SQLAlchemy non installe : %s", e)
            return None
    return _scheduler


def _envoyer_email_differe(destinataire: str, sujet: str, contenu: str) -> None:
    """Job interne appele par le scheduler : envoie reellement l'email."""
    logger.info("Scheduler declenche : envoi a %s", destinataire)
    # Appel direct de la logique send_email (sans le decorateur @tool)
    send_email.invoke(
        {"destinataire": destinataire, "sujet": sujet, "contenu": contenu}
    )


@tool
def schedule_email(
    destinataire: str, sujet: str, contenu: str, delay_minutes: int
) -> str:
    """
    Programme l'envoi d'un email dans X minutes (envoi differe).
    Utile pour : rappels, alertes planifiees, notifications differees.
    L'utilisateur peut demander "envoie dans 2 minutes", "rappelle-moi dans 1h",
    "programme un mail dans 30 minutes".

    Args:
        destinataire: Adresse email ou nom (Alice, P1...).
        sujet: Sujet du mail.
        contenu: Corps du mail en texte.
        delay_minutes: Delai en minutes avant l'envoi (1 a 10080, soit 7 jours max).

    Returns:
        Confirmation avec l'heure prevue d'envoi.
    """
    logger.info("Appel schedule_email vers %s dans %d min", destinataire, delay_minutes)

    if delay_minutes < 1 or delay_minutes > 10080:
        return "Le delai doit etre entre 1 minute et 7 jours (10080 min)."

    # Resoudre le nom en email si besoin
    if "@" not in destinataire:
        resolved = TEAM_DIRECTORY.get(destinataire.strip().lower())
        if resolved:
            destinataire = resolved

    scheduler = _get_scheduler()
    if scheduler is None:
        return (
            "Scheduler indisponible (APScheduler non installe). "
            "Executez : pip install apscheduler"
        )

    from datetime import datetime, timedelta

    run_time = datetime.now() + timedelta(minutes=delay_minutes)
    scheduler.add_job(
        _envoyer_email_differe,
        "date",
        run_date=run_time,
        args=[destinataire, sujet, contenu],
        id=f"email_{run_time.timestamp()}",
    )

    heure_envoi = run_time.strftime("%H:%M:%S")
    return (
        f"Email programme pour {heure_envoi} (dans {delay_minutes} min) "
        f"-> {destinataire} : {sujet}"
    )


@tool
def cancel_scheduled_emails() -> str:
    """
    Annule tous les emails programmes en attente dans le scheduler.
    Utile si l'utilisateur a programme un email par erreur ou veut tout arreter.

    Returns:
        Nombre d'emails annules et confirmation.
    """
    logger.info("Appel cancel_scheduled_emails")
    scheduler = _get_scheduler()
    if scheduler is None:
        return "Scheduler indisponible (APScheduler non installe)."

    jobs = scheduler.get_jobs()
    if not jobs:
        return "Aucun email programme en attente."

    nb = len(jobs)
    details = []
    for job in jobs:
        details.append(f"  - {job.id} (prevu {job.next_run_time})")
        job.remove()

    return (
        f"{nb} email(s) programme(s) annule(s) :\n"
        + "\n".join(details)
        + "\nTous les envois ont ete supprimes."
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 8 — Prédiction ML du risque catastrophe (modèle EM-DAT multi-type)
# ══════════════════════════════════════════════════════════════════

_cached_predictions_country = None
_cached_predictions_detail = None

CLIMATE_TYPES = [
    "drought",
    "flood",
    "extreme_weather",
    "extreme_temperature",
    "wildfire",
]


def _load_predictions() -> dict | None:
    """Charge les prédictions agrégées par pays (NB10).
    L'agregat est recalcule a partir des predictions detail clippees
    pour coherence (si clipping applique en detail, l'agregat est clippe aussi)."""
    global _cached_predictions_country
    if _cached_predictions_country is not None:
        return _cached_predictions_country

    # Recompose l'agregat depuis le detail clippe
    detail = _load_predictions_detail()
    if detail is not None:
        _cached_predictions_country = {}
        for key, data in detail.items():
            total = sum(t["impact_pred"] for t in data["types"].values())
            _cached_predictions_country[key] = {
                "country": data["country"],
                "continent": data["continent"],
                "impact_pred": total,
            }
        logger.info(
            "Prédictions ML (agrégat clippe) recomposees : %d pays",
            len(_cached_predictions_country),
        )
        return _cached_predictions_country

    # Fallback : CSV brut si detail indisponible
    path = os.path.join("outputs", "NB10_predictions_2030_country.csv")
    if not os.path.exists(path):
        logger.warning("Prédictions ML non trouvées : %s", path)
        return None

    import csv

    _cached_predictions_country = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _cached_predictions_country[row["country"].lower()] = {
                "country": row["country"],
                "continent": row["continent"],
                "impact_pred": float(row["impact_pred"]),
            }
    logger.info(
        "Prédictions ML (agrégat) chargées : %d pays",
        len(_cached_predictions_country),
    )
    return _cached_predictions_country


def _load_historical_max() -> dict:
    """Charge le max historique decadal par (pays, type) depuis EM-DAT.
    Sert a clipper les predictions 2030 aberrantes (extrapolation warming trop forte).
    """
    path = os.path.join("data", "decadal-deaths-disasters-by-type.csv")
    if not os.path.exists(path):
        return {}
    import csv

    maxima = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                country = row.get("entity", "").lower()
                for t in CLIMATE_TYPES:
                    col = f"total_dead_{t}_decadal"
                    try:
                        v = float(row.get(col) or 0)
                    except ValueError:
                        v = 0
                    key = (country, t)
                    if v > maxima.get(key, 0):
                        maxima[key] = v
    except Exception as exc:
        logger.warning("Impossible de charger max historique : %s", exc)
    return maxima


def _load_predictions_detail() -> dict | None:
    """Charge les prédictions détaillées par (pays, type) (NB10)."""
    global _cached_predictions_detail
    if _cached_predictions_detail is not None:
        return _cached_predictions_detail

    path = os.path.join("outputs", "NB10_predictions_2030_detail.csv")
    if not os.path.exists(path):
        logger.warning("Prédictions ML détaillées non trouvées : %s", path)
        return None

    import csv

    # Charge le max historique pour clipping des predictions aberrantes
    hist_max = _load_historical_max()
    clip_multiplier = 1.5  # tolerer +50% vs pire decennie historique

    _cached_predictions_detail = {}
    clip_count = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["country"].lower()
            if key not in _cached_predictions_detail:
                _cached_predictions_detail[key] = {
                    "country": row["country"],
                    "continent": row["continent"],
                    "types": {},
                }
            # Clipping anti-aberration : max 1.5x pire decennie historique
            impact = float(row["impact_pred"])
            hist_key = (key, row["disaster_type"])
            hist_decadal_max = hist_max.get(hist_key, 0)
            if hist_decadal_max > 0:
                cap_annuel = (hist_decadal_max / 10) * clip_multiplier
                if impact > cap_annuel:
                    impact = cap_annuel
                    clip_count += 1
            _cached_predictions_detail[key]["types"][row["disaster_type"]] = {
                "impact_pred": impact,
                "risk_pred": row["risk_pred"],
            }
    if clip_count > 0:
        logger.info(
            "Clipping anti-aberration : %d predictions plafonnees (1.5x max historique)",
            clip_count,
        )
    logger.info(
        "Prédictions ML (détail) chargées : %d pays × %d types",
        len(_cached_predictions_detail),
        len(CLIMATE_TYPES),
    )
    return _cached_predictions_detail


def _resolve_country(country: str, predictions: dict) -> str | None:
    """Résout un nom de pays (tolère casse et correspondance partielle)."""
    key = country.strip().lower()
    if key in predictions:
        return key
    matches = [k for k in predictions if key in k or k in key]
    return matches[0] if matches else None


@tool
def predict_risk(country: str) -> str:
    """
    Prédit l'impact humain agrégé des catastrophes climatiques pour un pays
    à l'horizon 2030, avec le détail par type (sécheresse, inondation,
    tempête, canicule, feux de forêt). Basé sur un modèle ML entraîné sur
    les données EM-DAT (1900-2020) en format multi-type, enrichi d'une
    variable exogène de réchauffement global (NASA GISS).

    Args:
        country: Nom du pays en anglais (ex: "France", "India", "Bangladesh").

    Returns:
        Prédiction : impact total 2030 + détail par type de catastrophe
        climatique + niveau de risque associé.
    """
    logger.info("Appel predict_risk pour %s", country)
    agg = _load_predictions()
    detail = _load_predictions_detail()

    if agg is None or detail is None:
        return (
            "Modèle ML non disponible. Lancez d'abord le notebook NB10 "
            "pour générer outputs/NB10_predictions_2030_country.csv "
            "et outputs/NB10_predictions_2030_detail.csv."
        )

    key = _resolve_country(country, agg)
    if key is None:
        pays_dispo = ", ".join(sorted(set(p["country"] for p in agg.values()))[:20])
        return (
            f"Pays « {country} » non trouvé dans les prédictions ML. "
            f"Exemples disponibles : {pays_dispo}..."
        )

    pred = agg[key]
    types_data = detail.get(key, {}).get("types", {})

    lines = [
        f"Prédiction ML 2030 pour {pred['country']} ({pred['continent']})",
        f"- Impact total (tous types climatiques) : {pred['impact_pred']:.0f} décès/an",
        "- Détail par type :",
    ]
    for typ in CLIMATE_TYPES:
        if typ in types_data:
            t = types_data[typ]
            lines.append(
                f"    {typ:<22} {t['impact_pred']:>8.0f} décès/an "
                f"(risque : {t['risk_pred']})"
            )

    lines.append(
        "- Source : modèle sklearn multi-type (5 types climatiques) entraîné "
        "sur EM-DAT 1900-2020, avec feature exogène NASA GISS (réchauffement)."
    )
    lines.append(
        "- Limite : prédiction statistique, ne tient pas compte des politiques "
        "d'adaptation ni des projections climatiques physiques."
    )
    return "\n".join(lines)


@tool
def predict_risk_by_type(country: str, disaster_type: str) -> str:
    """
    Prédit l'impact humain d'un type spécifique de catastrophe climatique
    pour un pays à l'horizon 2030. Utile pour répondre aux questions
    ciblées du style « quel est le risque d'inondation au Bangladesh ? ».

    Args:
        country: Nom du pays en anglais (ex: "Bangladesh").
        disaster_type: Type de catastrophe parmi drought, flood,
            extreme_weather, extreme_temperature, wildfire.

    Returns:
        Impact prédit 2030 pour ce type, niveau de risque, comparaison
        avec les autres types climatiques du pays.
    """
    logger.info("Appel predict_risk_by_type %s / %s", country, disaster_type)
    detail = _load_predictions_detail()

    if detail is None:
        return (
            "Modèle ML non disponible. Lancez d'abord le notebook NB10 "
            "pour générer outputs/NB10_predictions_2030_detail.csv."
        )

    typ = disaster_type.strip().lower().replace(" ", "_")
    if typ not in CLIMATE_TYPES:
        return (
            f"Type « {disaster_type} » non reconnu. "
            f"Types disponibles : {', '.join(CLIMATE_TYPES)}."
        )

    key = _resolve_country(country, detail)
    if key is None:
        return f"Pays « {country} » non trouvé dans les prédictions ML."

    pays_data = detail[key]
    if typ not in pays_data["types"]:
        return f"Pas de prédiction disponible pour {typ} en {pays_data['country']}."

    t = pays_data["types"][typ]
    # Ranking du type parmi les autres types du pays
    tous = sorted(
        pays_data["types"].items(),
        key=lambda x: x[1]["impact_pred"],
        reverse=True,
    )
    rang = [i for i, (nom, _) in enumerate(tous) if nom == typ][0] + 1

    return (
        f"Prédiction ML 2030 — {typ} en {pays_data['country']} "
        f"({pays_data['continent']})\n"
        f"- Impact prédit : {t['impact_pred']:.0f} décès/an\n"
        f"- Niveau de risque : {t['risk_pred']}\n"
        f"- Rang dans le pays : {rang}/{len(tous)} types climatiques\n"
        f"- Source : modèle multi-type EM-DAT 1900-2020 + exogène "
        f"réchauffement NASA GISS."
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

# Ponderation par defaut (horizon "standard" = 7 jours).
# Voir _POIDS_PAR_HORIZON pour les variantes court-terme / long-terme.
_POIDS = {
    "meteo": 0.35,
    "corpus": 0.25,
    "ml_predict": 0.25,
    "historique": 0.15,
}

# Ponderation dynamique selon l'horizon temporel de la decision.
# Court-terme (24-48h) : la meteo prime, ML et histo sont secondaires
# Standard (7 jours) : equilibre entre signal frais et contexte
# Long-terme (1-5 ans) : les projections ML + le corpus scientifique priment
_POIDS_PAR_HORIZON = {
    "court_terme": {
        "meteo": 0.50,
        "ml_predict": 0.20,
        "corpus": 0.20,
        "historique": 0.10,
    },
    "standard": _POIDS,  # 35/25/25/15 (defaut)
    "long_terme": {
        "meteo": 0.10,
        "ml_predict": 0.40,
        "corpus": 0.40,
        "historique": 0.10,
    },
}


@tool
def calculer_score_risque(
    precipitation_prevue_mm: float,
    seuil_critique_mm: float,
    risk_level_ml: str,
    a_precedent_historique: bool,
    corpus_mentionne_risque: bool,
    horizon: str = "standard",
) -> str:
    """
    Calcule un score de risque agrégé (0-1) en croisant 4 sources :
    météo (précipitations vs seuil), ML (prédiction EM-DAT),
    corpus (GIEC mentionne un risque), historique (précédent connu).

    La pondération s'adapte à l'horizon temporel de la décision :
    - "court_terme" (24-48h) : météo 50 % + ML 20 % + corpus 20 % + histo 10 %
    - "standard" (7 jours, défaut) : 35/25/25/15
    - "long_terme" (1-5 ans) : ML 40 % + corpus 40 % + météo 10 % + histo 10 %

    Args:
        precipitation_prevue_mm: Précipitations prévues en mm.
        seuil_critique_mm: Seuil critique du GIEC en mm.
        risk_level_ml: Niveau de risque ML (Aucun/Faible/Modéré/Élevé/Critique).
        a_precedent_historique: True si un événement similaire a eu lieu dans le passé.
        corpus_mentionne_risque: True si le corpus GIEC mentionne un risque pour la zone.
        horizon: "court_terme" / "standard" / "long_terme" (défaut "standard").

    Returns:
        Score agrégé avec détail par source et décision GO/NO-GO.
    """
    logger.info(
        "Calcul score risque : %.0fmm vs %.0fmm seuil, ML=%s, horizon=%s",
        precipitation_prevue_mm,
        seuil_critique_mm,
        risk_level_ml,
        horizon,
    )

    # Selection de la ponderation selon l'horizon (fallback standard si inconnu)
    poids = _POIDS_PAR_HORIZON.get(horizon, _POIDS)

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

    # Agrégation pondérée selon l'horizon
    score_final = (
        poids["meteo"] * score_meteo
        + poids["ml_predict"] * score_ml
        + poids["corpus"] * score_corpus
        + poids["historique"] * score_historique
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
        f"Score de risque agrégé : {score_final:.2f} / 1.00 "
        f"(horizon : {horizon})\n"
        f"\nDétail par source (pondération {horizon}) :\n"
        f"  - Météo      ({poids['meteo']:.0%}) : {score_meteo:.2f} "
        f"({precipitation_prevue_mm:.0f}mm / {seuil_critique_mm:.0f}mm seuil)\n"
        f"  - ML EM-DAT  ({poids['ml_predict']:.0%}) : {score_ml:.2f} "
        f"(niveau {risk_level_ml})\n"
        f"  - Corpus GIEC ({poids['corpus']:.0%}) : {score_corpus:.2f} "
        f"({'risque mentionné' if corpus_mentionne_risque else 'pas de mention'})\n"
        f"  - Historique  ({poids['historique']:.0%}) : {score_historique:.2f} "
        f"({'précédent connu' if a_precedent_historique else 'pas de précédent'})\n"
        f"\nDécision : {decision}"
    )


# ══════════════════════════════════════════════════════════════════
# OUTIL 10 — Inventaire du corpus RAG
# ══════════════════════════════════════════════════════════════════


@tool
def list_corpus() -> str:
    """
    Liste tous les documents du corpus RAG avec leurs metadonnees.
    Utilise cet outil quand l'utilisateur demande combien de documents
    il y a, quels sont les documents disponibles, ou veut un inventaire
    du corpus complet.

    Returns:
        Liste complete des fichiers PDF du corpus avec taille et nombre de pages.
    """
    logger.info("Appel list_corpus")

    # Lire le CSV pre-genere (rapide)
    csv_path = os.path.join("outputs", "corpus_inventory.csv")
    if os.path.exists(csv_path):
        import csv

        lines = ["Corpus SAEARCH :\n"]
        total = 0
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                lines.append(
                    f"{total}. {row['fichier']} "
                    f"({row['taille_mo']} Mo, {row['pages']} pages)"
                )
        lines.append(f"\nTotal : {total} documents")
        return "\n".join(lines)

    # Fallback : scan du dossier (sans compter les pages)
    corpus_dir = os.path.join("data", "raw")
    if not os.path.exists(corpus_dir):
        return "Le dossier corpus (data/raw/) n'existe pas."

    files = sorted(
        f
        for f in os.listdir(corpus_dir)
        if f.lower().endswith((".pdf", ".docx", ".txt"))
    )

    if not files:
        return "Aucun document trouve dans le corpus."

    lines = [f"Corpus SAEARCH : {len(files)} documents\n"]
    for i, f in enumerate(files, 1):
        path = os.path.join(corpus_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines.append(f"{i}. {f} ({size_mb:.1f} Mo)")

    return "\n".join(lines)


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
    send_bulk_email,
    schedule_email,
    cancel_scheduled_emails,
    predict_risk,
    predict_risk_by_type,
    calculer_score_risque,
    list_corpus,
]


# Initialisation EAGER du scheduler : au chargement du module tools,
# demarre le BackgroundScheduler. Sans cela, les jobs persistes dans
# scheduler_jobs.db ne s'executent pas tant que personne n'appelle
# schedule_email (donc les jobs en retard restent orphelins).
try:
    _get_scheduler()
    logger.info("Scheduler initialise au chargement de tools.py (eager)")
except Exception as _e:
    logger.warning("Init eager scheduler impossible : %s", _e)
