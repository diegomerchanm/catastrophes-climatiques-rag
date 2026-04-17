"""
Point d'entrée Chainlit — Interface conversationnelle.
Combine le router conditionnel (P3 — P3) avec l'agent agentic RAG (P4 — P4).
- Route "rag" et "chat" : streaming token par token via chain.astream() (pattern P3)
- Route "agent" : agent ReAct 9 outils avec astream_events (pattern P4)
- Fallback : si l'agent ou le streaming échoue, le router compilé P3 prend le relais
Fonctionnalités : streaming, badges de route, sources, STT, upload PDF/DOCX, monitoring LLMOps.
"""

import logging
import os
import tempfile

import chainlit as cl
import chainlit.data as cl_data
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.agent import (
    get_agent,
    get_prompt_version,
    get_token_summary,
    run_agent,
)
from src.ui.donut_chart import generer_message_avec_donut
from src.ui.data_layer import SQLiteDataLayer

# Activer le data layer SQLite pour la sidebar
cl_data._data_layer = SQLiteDataLayer()
from src.memory.memory import add_exchange, get_session_history
from src.router.router import (
    RAG_PROMPT,
    CHAT_PROMPT,
    classify_question,
    router as jayson_router,
    RouterState,
    rag_node,
    chat_node,
)
from src.config import get_llm

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Badges de route ──────────────────────────────────────────────────────

ROUTE_LABELS = {
    "rag": "RAG -- Reponse basee sur le corpus scientifique",
    "agent": "Agent -- Outils externes utilises",
    "chat": "Chat -- Conversation directe",
}

TOOL_BADGES = {
    "search_corpus": "RAG",
    "list_corpus": "RAG",
    "get_weather": "Meteo",
    "get_historical_weather": "Meteo",
    "get_forecast": "Meteo",
    "web_search": "Web",
    "calculator": "Calcul",
    "send_email": "Email",
    "send_bulk_email": "Email",
    "schedule_email": "Email",
    "predict_risk": "ML",
    "predict_risk_by_type": "ML",
    "calculer_score_risque": "Scoring",
}


def _detecter_outils_appeles(messages: list) -> tuple[set, list, list]:
    """Detecte les outils appeles et les sources RAG dans les messages de l'agent."""
    outils = set()
    outils_bruts = []
    sources = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                nom = tc["name"]
                outils_bruts.append(nom)
                if nom in TOOL_BADGES:
                    outils.add(TOOL_BADGES[nom])

        if hasattr(msg, "content") and "[Source:" in str(msg.content):
            for ligne in str(msg.content).split("\n"):
                if "[Source:" in ligne and ligne.strip() not in sources:
                    sources.append(ligne.strip())

    return outils, sources, outils_bruts


# ── Authentification ──────────────────────────────────────────────────────


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authentification simple par mot de passe.
    Les comptes utilisateurs sont charges depuis la variable d'environnement
    USER_ACCOUNTS_JSON (JSON serialise). Voir .env.example pour le format.
    En production : migrer vers OAuth + DB avec hash bcrypt.
    """
    import json

    users_json = os.getenv("USER_ACCOUNTS_JSON", "{}")
    try:
        users = json.loads(users_json)
    except json.JSONDecodeError:
        logger.error("USER_ACCOUNTS_JSON mal formate dans .env")
        users = {}

    # Fallback minimum : compte demo pour que l'app reste utilisable
    # meme sans .env configure (pour tests basiques)
    if not users:
        users = {
            "demo@saearch.ai": {"password": "demo", "name": "Demo", "role": "user"}
        }

    user = users.get(username.lower())
    if user and user["password"] == password:
        return cl.User(
            identifier=user["name"],
            metadata={"email": username, "role": user["role"]},
        )
    return None


# ── Initialisation de la session ──────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start():
    """Initialise une nouvelle session de chat."""
    session_id = cl.user_session.get("id", "default")
    cl.user_session.set("session_id", session_id)

    # Recuperer l'utilisateur connecte
    user = cl.user_session.get("user")
    user_name = user.identifier if user else "utilisateur"
    user_email = user.metadata.get("email", "") if user else ""
    cl.user_session.set("user_email", user_email)

    # Geolocalisation par IP (une seule fois au login)
    user_location = ""
    try:
        import requests as _req

        geo = _req.get("http://ip-api.com/json/?lang=fr", timeout=5).json()
        if geo.get("status") == "success":
            city = geo.get("city", "")
            country = geo.get("country", "")
            region = geo.get("regionName", "")
            lat = geo.get("lat", "")
            lon = geo.get("lon", "")
            user_location = f"{city}, {region}, {country} ({lat}, {lon})"
            logger.info("Geolocalisation : %s", user_location)
    except Exception as exc:
        logger.warning("Geolocalisation echouee : %s", exc)
    cl.user_session.set("user_location", user_location)
    logger.info("Nouvelle session Chainlit : %s (user: %s)", session_id, user_name)

    # Donut d'accueil personnalise
    greeting = f"Bonjour <b>{user_name}</b>" if user else "Bonjour"
    donut_accueil = generer_message_avec_donut(
        answer=(
            f"{greeting}, je suis <b>DooMax</b>, votre IA dans <b>SAEARCH</b>, "
            "le Systeme Agentique d'Evaluation et d'Anticipation "
            "des Risques Climatiques et Hydrologiques.<br><br>"
            "Comment puis-je t'aider ?"
        ),
        outils_appeles=[],
        route="chat",
    )
    await cl.Message(content=donut_accueil).send()

    # Mode decisionnel : 3 boutons d'aide a la decision pour profils metier
    # (organisateur, assureur, decideur public). Clique -> dialogue guide ->
    # prompt force format GO/NO-GO + score + recommandations.
    decision_actions = [
        cl.Action(
            name="decide_event",
            payload={"mode": "event"},
            label="🟢 Organisation d'evenement",
            tooltip="GO/NO-GO meteo pour evenement en exterieur",
        ),
        cl.Action(
            name="decide_insurance",
            payload={"mode": "insurance"},
            label="🛡️ Analyse assurance",
            tooltip="Previsibilite d'un sinistre pour contentieux juridique",
        ),
        cl.Action(
            name="decide_public",
            payload={"mode": "public"},
            label="🚨 Decision publique",
            tooltip="Recommandation evacuation/alerte pour autorites",
        ),
        cl.Action(
            name="decide_tourist",
            payload={"mode": "tourist"},
            label="🏖️ Particulier / Vacances",
            tooltip="Evaluation risque climatique pour un sejour touristique",
        ),
    ]
    await cl.Message(
        content=(
            "<b>Mode Aide à la Décision</b> — Pour préparer vos déplacements, "
            "vos événements, choisissez un profil pour avoir une aide à la "
            "décision structurée en 30 secondes :"
        ),
        actions=decision_actions,
    ).send()


# ── Mode decisionnel : boutons action avec dialogue guide ─────────────────


async def _run_decision_agent(prompt: str, route_label: str = "Agent") -> None:
    """Execute l'agent en mode decisionnel et affiche la reponse avec donut.

    Ajoute un workflow Human-in-the-Loop : apres chaque decision, 3 boutons
    (approuver / modifier / rejeter) permettent a l'utilisateur d'engager sa
    responsabilite. L'IA propose, l'humain dispose.
    """
    session_id = cl.user_session.get("session_id")
    history = get_session_history(session_id)
    chat_history = list(history.messages) if history else []

    msg = cl.Message(content=f"**{route_label}**\n\n_Analyse en cours..._")
    await msg.send()

    try:
        # Reuse existing agent runner
        answer = run_agent(prompt, chat_history=chat_history)
    except Exception as exc:
        logger.error("Erreur mode decisionnel : %s", exc, exc_info=True)
        msg.content = f"**Erreur**\n\n{exc}"
        await msg.update()
        return

    # Ajout a la memoire session
    if session_id:
        add_exchange(session_id, prompt, answer)

    # Affichage final avec donut
    tokens = get_token_summary()
    tokens_info = ""
    if tokens["total_tokens"] > 0:
        cost = tokens.get("estimated_cost_usd", 0)
        tokens_info = (
            f"Tokens : {tokens['total_tokens']} "
            f"(in: {tokens['total_input_tokens']}, "
            f"out: {tokens['total_output_tokens']}) -- "
            f"${cost:.4f} -- {get_prompt_version()}"
        )

    # Tools detection (best-effort simple)
    outils_bruts = []
    try:
        from src.agents.agent import get_last_tools_called

        outils_bruts = get_last_tools_called() or []
    except Exception:
        pass

    msg.content = generer_message_avec_donut(
        answer=answer,
        outils_appeles=outils_bruts,
        route="agent",
        sources=[],
        tokens_info=tokens_info,
    )
    await msg.update()

    # Stocker la decision courante dans la session pour les callbacks HITL
    cl.user_session.set("last_decision_prompt", prompt)
    cl.user_session.set("last_decision_answer", answer)

    # Human-in-the-Loop : AskActionMessage rend des boutons modaux XXL,
    # impossibles a louper sur mobile (contrairement au Message+actions
    # qui rendait des pilules minuscules).
    # Human-in-the-Loop via message + actions classiques (pas AskActionMessage
    # qui bloque le flux et laisse un spinner actif).
    hitl_actions = [
        cl.Action(
            name="hitl_approve",
            payload={"status": "approved"},
            label="✅ J'approuve",
        ),
        cl.Action(
            name="hitl_modify",
            payload={"status": "modified"},
            label="⚠️ Enrichir",
        ),
        cl.Action(
            name="hitl_reject",
            payload={"status": "rejected"},
            label="❌ Rejeter",
        ),
    ]
    await cl.Message(
        content=(
            "<b>🧑‍⚖️ L'IA propose, vous disposez.</b><br>"
            "Cette recommandation doit etre <b>validee par un humain</b> "
            "avant toute action operationnelle."
        ),
        actions=hitl_actions,
    ).send()


# ── Callbacks Human-in-the-Loop ────────────────────────────────────────────


def _log_hitl_feedback(status: str, comment: str = "") -> None:
    """Logge le feedback humain dans MLflow pour tracabilite."""
    try:
        import mlflow as _mlflow

        tracking = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
        _mlflow.set_tracking_uri(tracking)
        _mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT_AGENT") or "rag-catastrophes-climatiques"
        )
        with _mlflow.start_run(run_name=f"hitl_{status}", nested=True):
            _mlflow.set_tag("hitl_status", status)
            _mlflow.set_tag("hitl_comment", comment[:250] if comment else "")
            _mlflow.log_metric("hitl_approved", 1 if status == "approved" else 0)
            _mlflow.log_metric("hitl_rejected", 1 if status == "rejected" else 0)
            _mlflow.log_metric("hitl_modified", 1 if status == "modified" else 0)
        logger.info("HITL feedback logge : %s", status)
    except Exception as exc:
        logger.debug("HITL MLflow log impossible : %s", exc)


@cl.action_callback("hitl_approve")
async def _hitl_apply_approved(action: cl.Action) -> None:
    """Traitement quand l'utilisateur clique Approuver."""
    await action.remove()
    _log_hitl_feedback("approved")
    await cl.Message(
        content=(
            "✅ <b>Decision approuvee et archivee.</b><br>"
            "Le statut <code>approved</code> a ete logge dans MLflow "
            "(tag <code>hitl_status=approved</code>). "
            "Vous pouvez maintenant engager les actions operationnelles."
        ),
    ).send()


@cl.action_callback("hitl_modify")
async def _hitl_apply_modified(action: cl.Action) -> None:
    """Traitement quand l'utilisateur clique Enrichir."""
    await action.remove()
    res_contexte = await cl.AskUserMessage(
        content=(
            "⚠️ <b>Contexte supplementaire ?</b><br>"
            "Ex : 'zone cotiere exposee avec afflux touristes samedi' "
            "ou 'precedent similaire en 2023 avec degats'. "
            "L'agent relancera une analyse enrichie."
        ),
        timeout=180,
    ).send()
    if not res_contexte:
        return
    contexte = _clean_input(res_contexte)
    _log_hitl_feedback("modified", contexte)

    original_prompt = cl.user_session.get("last_decision_prompt", "")
    new_prompt = (
        f"{original_prompt}\n\n"
        f"[CONTEXTE TERRAIN AJOUTE PAR L'UTILISATEUR]\n{contexte}\n\n"
        f"Tiens compte de ce contexte pour affiner ta recommandation "
        f"(meme format de reponse)."
    )
    await _run_decision_agent(new_prompt, "Mode decisionnel — Revision enrichie")


@cl.action_callback("hitl_reject")
async def _hitl_apply_rejected(action: cl.Action) -> None:
    """Traitement quand l'utilisateur clique Rejeter."""
    await action.remove()
    res_raison = await cl.AskUserMessage(
        content=(
            "❌ <b>Pourquoi rejetez-vous cette recommandation ?</b><br>"
            "Votre retour permettra d'ameliorer le systeme (prompt engineering, "
            "seuils, outils) dans la prochaine version."
        ),
        timeout=180,
    ).send()
    if not res_raison:
        return
    raison = _clean_input(res_raison)
    _log_hitl_feedback("rejected", raison)
    await cl.Message(
        content=(
            "🚫 <b>Decision rejetee et feedback archive.</b><br>"
            f"Raison enregistree dans MLflow : <i>{raison[:200]}</i><br>"
            "Merci pour votre retour — il sera analyse pour affiner les prompts "
            "decisionnels et les seuils de `calculer_score_risque`."
        ),
    ).send()


def _prompt_decisionnel_event(lieu: str, date: str, type_evt: str) -> str:
    """Construit un prompt force format decision pour evenement en exterieur."""
    return (
        f"[MODE DECISIONNEL GO/NO-GO — ORGANISATION EVENEMENT]\n"
        f"Profil utilisateur : organisateur evenementiel\n"
        f"Evenement : {type_evt} en exterieur a {lieu} le {date}\n\n"
        f"PROTOCOLE OBLIGATOIRE :\n"
        f"1. Appelle get_forecast({lieu}) pour les previsions meteo\n"
        f"2. Appelle search_corpus pour verifier les seuils de risque saison\n"
        f"3. Appelle calculer_score_risque avec les 4 sources\n\n"
        f"FORMAT DE REPONSE STRICT :\n"
        f"Commence OBLIGATOIREMENT par ce bloc de legende (copie-le tel quel) :\n"
        f"---\n"
        f"**Guide de lecture** : DECISION = verdict operationnel | HORIZON = fenetre temporelle qui determine la ponderation des sources (court_terme : meteo 50%, standard : equilibre, long_terme : ML+corpus 80%) | SCORE = indice de risque 0-1 (0 = aucun danger, 1 = danger maximum) | RISQUE = traduction qualitative du score | CONFIANCE = qualite des sources\n"
        f"---\n\n"
        f"Puis enchaine avec les champs suivants :\n"
        f"**DECISION** : [GO / VIGILANCE / NO-GO]\n"
        f"**HORIZON** : [COURT TERME / STANDARD / LONG TERME] + ponderation appliquee\n"
        f"**SCORE DE RISQUE** : [0.00-1.00] (0 = aucun danger, 1 = danger maximum)\n"
        f"**RISQUE** : [TRES FAIBLE / FAIBLE / MODERE / ELEVE / CRITIQUE]\n"
        f"**CONFIANCE** : [ELEVEE / MOYENNE / FAIBLE]\n"
        f"**JUSTIFICATION** : 3 lignes maximum\n"
        f"**RECOMMANDATIONS** : 3 bullets concrets\n"
        f"**SOURCES** : [Source: fichier.pdf, Page: X]\n\n"
        f"Rappel : cette recommandation est une aide a la decision. "
        f"La decision finale reste a l'humain responsable.\n"
    )


def _prompt_decisionnel_insurance(lieu: str, date_evt: str, type_sinistre: str) -> str:
    """Construit un prompt pour analyse de previsibilite assurantielle."""
    return (
        f"[MODE DECISIONNEL — ANALYSE ASSURANCE]\n"
        f"Profil utilisateur : assureur evaluant un contentieux\n"
        f"Sinistre : {type_sinistre} a {lieu} le {date_evt}\n\n"
        f"PROTOCOLE OBLIGATOIRE :\n"
        f"1. search_corpus : chercher alertes et seuils documentes a l'epoque\n"
        f"2. get_historical_weather({lieu}, {date_evt}) pour conditions reelles\n"
        f"3. web_search : articles presse pour alertes officielles emises\n"
        f"4. Conclure si le sinistre etait PREVISIBLE ou non\n\n"
        f"FORMAT DE REPONSE STRICT :\n"
        f"Commence OBLIGATOIREMENT par ce bloc de legende (copie-le tel quel) :\n"
        f"---\n"
        f"**Guide de lecture** : DECISION = previsibilite du sinistre | CERTITUDE = force du faisceau de preuves | CONFIANCE = qualite des sources\n"
        f"---\n\n"
        f"Puis enchaine avec les champs suivants :\n"
        f"**DECISION** : [PREVISIBLE / PARTIELLEMENT PREVISIBLE / NON PREVISIBLE]\n"
        f"**NIVEAU DE CERTITUDE** : [FAIBLE / MOYEN / ELEVE]\n"
        f"**CONFIANCE** : [ELEVEE / MOYENNE / FAIBLE]\n"
        f"**PREUVES CLES** : 3-5 bullets sources\n"
        f"**IMPLICATIONS JURIDIQUES** : 2 lignes max (pas de conseil juridique)\n"
        f"**SOURCES** : [Source: fichier.pdf, Page: X] pour chaque preuve\n\n"
        f"Rappel : cette analyse est une aide a l'expertise. "
        f"Elle ne remplace pas l'avis d'un expert climatologue ni d'un avocat.\n"
    )


def _prompt_decisionnel_public(lieu: str, type_decision: str) -> str:
    """Construit un prompt pour decision publique (maire/prefet/SDIS)."""
    return (
        f"[MODE DECISIONNEL — DECISION PUBLIQUE]\n"
        f"Profil utilisateur : autorite publique (maire/prefet/SDIS)\n"
        f"Decision a prendre : {type_decision} pour {lieu} dans les 7 jours\n\n"
        f"PROTOCOLE OBLIGATOIRE :\n"
        f"1. get_forecast({lieu}) : previsions 7 jours\n"
        f"2. search_corpus : seuils critiques de la region\n"
        f"3. predict_risk({lieu}) ou predict_risk_by_type\n"
        f"4. calculer_score_risque : agregation 4 sources\n\n"
        f"FORMAT DE REPONSE STRICT :\n"
        f"Commence OBLIGATOIREMENT par ce bloc de legende (copie-le tel quel) :\n"
        f"---\n"
        f"**Guide de lecture** : RECOMMANDATION = action preconisee | HORIZON = ponderation temporelle des sources | SCORE = indice de risque 0-1 (0 = aucun danger, 1 = danger maximum) | RISQUE = traduction qualitative | URGENCE = fenetre d'action | CONFIANCE = qualite des sources\n"
        f"---\n\n"
        f"Puis enchaine avec les champs suivants :\n"
        f"**RECOMMANDATION** : [DECLENCHEMENT / VIGILANCE / STANDBY]\n"
        f"**HORIZON** : [COURT TERME / STANDARD / LONG TERME] + ponderation appliquee\n"
        f"**SCORE DE RISQUE** : [0.00-1.00] (0 = aucun danger, 1 = danger maximum)\n"
        f"**RISQUE** : [TRES FAIBLE / FAIBLE / MODERE / ELEVE / CRITIQUE]\n"
        f"**URGENCE** : [IMMEDIATE / 24-48H / 7 JOURS / ROUTINE]\n"
        f"**CONFIANCE** : [ELEVEE / MOYENNE / FAIBLE]\n"
        f"**JUSTIFICATION CHIFFREE** : metriques precises\n"
        f"**MESURES CONCRETES** : 3-5 bullets operationnels\n"
        f"**SOURCES** : [Source: fichier.pdf, Page: X]\n\n"
        f"Rappel : cette recommandation est une aide a la decision. "
        f"La responsabilite legale de toute mesure (evacuation, alerte, fermeture) "
        f"incombe a l'autorite publique competente.\n"
    )


def _prompt_decisionnel_tourist(
    destination: str, date_arrivee: str, duree: str, type_activite: str
) -> str:
    """Construit un prompt pour conseil touristique (particulier en vacances)."""
    return (
        f"[MODE DECISIONNEL — SEJOUR TOURISTIQUE]\n"
        f"Profil utilisateur : particulier en visite ou vacances\n"
        f"Destination : {destination}\n"
        f"Date d'arrivee : {date_arrivee}\n"
        f"Duree du sejour : {duree}\n"
        f"Type d'activite prevue : {type_activite}\n\n"
        f"PROTOCOLE OBLIGATOIRE :\n"
        f"1. Appelle get_forecast({destination}) pour les previsions 7 jours\n"
        f"2. Appelle search_corpus pour identifier les risques saisonniers de la region\n"
        f"3. Appelle predict_risk ou predict_risk_by_type selon le type d'activite\n"
        f"4. Appelle calculer_score_risque avec horizon='court_terme' si sejour <=7 jours, "
        f"sinon horizon='standard'\n\n"
        f"REGLES STRICTES DE FORMAT :\n"
        f"- Tu commences DIRECTEMENT par **RECOMMANDATION** (pas de bandeau, "
        f"pas d'introduction narrative, pas de 'ATTENTION', pas de preambule).\n"
        f"- Si vous detectez une incoherence (ex : destination inadaptee a l'activite), "
        f"mets la remarque UNIQUEMENT dans le champ **CONSEILS PRATIQUES** (premier bullet).\n"
        f"- Respecte exactement les 7 champs ci-dessous, dans cet ordre, sans rien "
        f"ajouter avant ni apres.\n\n"
        f"FORMAT DE REPONSE STRICT (7 champs, commence directement par le premier) :\n"
        f"Commence OBLIGATOIREMENT par ce bloc de legende (copie-le tel quel) :\n"
        f"---\n"
        f"**Guide de lecture** : RECOMMANDATION = conseil sejour | HORIZON = ponderation temporelle (court_terme pour sejour <=7j, standard sinon) | SCORE = indice de risque 0-1 (0 = aucun danger, 1 = danger maximum) | RISQUE = traduction qualitative | CONFIANCE = qualite des sources\n"
        f"---\n\n"
        f"Puis enchaine avec les champs suivants :\n"
        f"**RECOMMANDATION** : [PARTEZ TRANQUILLE / VIGILANCE / REPORTEZ LE SEJOUR]\n"
        f"**HORIZON** : [COURT TERME / STANDARD] + ponderation appliquee\n"
        f"**SCORE DE RISQUE** : [0.00-1.00] (0 = aucun danger, 1 = danger maximum)\n"
        f"**RISQUE** : [TRES FAIBLE / FAIBLE / MODERE / ELEVE / CRITIQUE]\n"
        f"**CONFIANCE** : [ELEVEE / MOYENNE / FAIBLE]\n"
        f"**CONSEILS PRATIQUES** : 3-5 bullets concrets adaptes au touriste "
        f"(materiel a prevoir, activites a privilegier/eviter, numeros d'urgence locaux, "
        f"assurance voyage conseillee, points de repli en cas d'alerte). "
        f"Si incoherence detectee (ex : destination inadaptee a l'activite), "
        f"le premier bullet la signale factuellement, sans bandeau.\n"
        f"**ALERTES A SURVEILLER** : 2-3 signaux meteo/actualite a verifier avant le depart\n"
        f"**SOURCES** : [Source: fichier.pdf, Page: X]\n"
    )


def _clean_input(res) -> str:
    """Extrait le texte d'une reponse AskUserMessage, trime, retourne str vide si None."""
    if not res:
        return ""
    if isinstance(res, dict):
        raw = res.get("output", "")
    else:
        raw = getattr(res, "content", "") or ""
    return (raw or "").strip()


async def _confirmer_recap(champs: list[tuple[str, str]]) -> bool:
    """Affiche un recap des saisies et demande confirmation avant de lancer l'agent.

    Args:
        champs: liste de (label, valeur) saisis.

    Returns:
        True si l'utilisateur confirme, False s'il annule/recommence.
    """
    recap_html = "<b>📋 Récapitulatif de vos saisies :</b><br><br>"
    for label, valeur in champs:
        val_affichee = valeur if valeur else "<i>(vide)</i>"
        recap_html += f"• <b>{label}</b> : {val_affichee}<br>"
    recap_html += (
        "<br>Si une info est inversée ou mal saisie, cliquez "
        "<b>Recommencer</b>. Sinon lancez l'analyse."
    )

    res = await cl.AskActionMessage(
        content=recap_html,
        actions=[
            cl.Action(
                name="confirm_recap",
                payload={"value": "ok"},
                label="✅ Lancer l'analyse",
            ),
            cl.Action(
                name="redo_recap",
                payload={"value": "redo"},
                label="🔄 Recommencer",
            ),
        ],
        timeout=180,
    ).send()

    if not res:
        return False
    value = res.get("payload", {}).get("value") if isinstance(res, dict) else None
    return value == "ok"


@cl.action_callback("decide_event")
async def on_decide_event(action: cl.Action):
    """Dialogue guide pour decision d'organisation d'evenement."""
    await action.remove()
    while True:
        res_lieu = await cl.AskUserMessage(
            content="📍 **Lieu** de l'evenement (ville ou region) ?",
            timeout=180,
        ).send()
        if not res_lieu:
            return
        res_date = await cl.AskUserMessage(
            content="📅 **Date** de l'evenement (ex : samedi 19 avril 2026) ?",
            timeout=180,
        ).send()
        if not res_date:
            return
        res_type = await cl.AskUserMessage(
            content="🎪 **Type** d'evenement (concert, festival, mariage, sport, reunion publique...) ?",
            timeout=180,
        ).send()
        if not res_type:
            return

        lieu = _clean_input(res_lieu)
        date = _clean_input(res_date)
        type_evt = _clean_input(res_type)

        if await _confirmer_recap(
            [
                ("Lieu", lieu),
                ("Date", date),
                ("Type d'evenement", type_evt),
            ]
        ):
            break
        await cl.Message(content="🔄 On recommence la saisie...").send()

    prompt = _prompt_decisionnel_event(lieu, date, type_evt)
    await _run_decision_agent(prompt, "Mode decisionnel — Evenement")


@cl.action_callback("decide_insurance")
async def on_decide_insurance(action: cl.Action):
    """Dialogue guide pour analyse de previsibilite assurantielle."""
    await action.remove()
    while True:
        res_lieu = await cl.AskUserMessage(
            content="📍 **Lieu** du sinistre ?",
            timeout=180,
        ).send()
        if not res_lieu:
            return
        res_date = await cl.AskUserMessage(
            content="📅 **Date** du sinistre (format AAAA-MM-JJ) ?",
            timeout=180,
        ).send()
        if not res_date:
            return
        res_type = await cl.AskUserMessage(
            content="💧 **Type** de sinistre (inondation, tempete, incendie, canicule...) ?",
            timeout=180,
        ).send()
        if not res_type:
            return

        lieu = _clean_input(res_lieu)
        date = _clean_input(res_date)
        type_sin = _clean_input(res_type)

        if await _confirmer_recap(
            [
                ("Lieu du sinistre", lieu),
                ("Date du sinistre", date),
                ("Type de sinistre", type_sin),
            ]
        ):
            break
        await cl.Message(content="🔄 On recommence la saisie...").send()

    prompt = _prompt_decisionnel_insurance(lieu, date, type_sin)
    await _run_decision_agent(prompt, "Mode decisionnel — Assurance")


@cl.action_callback("decide_public")
async def on_decide_public(action: cl.Action):
    """Dialogue guide pour recommandation publique (maire/prefet/SDIS)."""
    await action.remove()
    while True:
        res_lieu = await cl.AskUserMessage(
            content="📍 **Commune ou region** concernee ?",
            timeout=180,
        ).send()
        if not res_lieu:
            return
        res_type = await cl.AskUserMessage(
            content="🚨 **Type de decision** (evacuation, alerte rouge, pre-alerte, fermeture ecoles, restriction route...) ?",
            timeout=180,
        ).send()
        if not res_type:
            return

        lieu = _clean_input(res_lieu)
        type_dec = _clean_input(res_type)

        if await _confirmer_recap(
            [
                ("Commune / region", lieu),
                ("Type de decision", type_dec),
            ]
        ):
            break
        await cl.Message(content="🔄 On recommence la saisie...").send()

    prompt = _prompt_decisionnel_public(lieu, type_dec)
    await _run_decision_agent(prompt, "Mode decisionnel — Decision publique")


@cl.action_callback("decide_tourist")
async def on_decide_tourist(action: cl.Action):
    """Dialogue guide pour un particulier en visite ou vacances."""
    await action.remove()
    while True:
        res_dest = await cl.AskUserMessage(
            content="🏖️ **Destination** (ville, region ou pays) ?",
            timeout=180,
        ).send()
        if not res_dest:
            return
        res_date = await cl.AskUserMessage(
            content="📅 **Date d'arrivee** (ex : 20 avril 2026) ?",
            timeout=180,
        ).send()
        if not res_date:
            return
        res_duree = await cl.AskUserMessage(
            content="⏱️ **Duree du sejour** (ex : 1 semaine, 10 jours, 1 mois) ?",
            timeout=180,
        ).send()
        if not res_duree:
            return
        res_type = await cl.AskUserMessage(
            content=(
                "🎒 **Type d'activite prevue** (plage, randonnee en montagne, "
                "visite urbaine, sports nautiques, camping, ski, safari...) ?"
            ),
            timeout=180,
        ).send()
        if not res_type:
            return

        dest = _clean_input(res_dest)
        date = _clean_input(res_date)
        duree = _clean_input(res_duree)
        type_act = _clean_input(res_type)

        if await _confirmer_recap(
            [
                ("Destination", dest),
                ("Date d'arrivee", date),
                ("Duree du sejour", duree),
                ("Type d'activite", type_act),
            ]
        ):
            break
        await cl.Message(content="🔄 On recommence la saisie...").send()

    prompt = _prompt_decisionnel_tourist(dest, date, duree, type_act)
    await _run_decision_agent(prompt, "Mode decisionnel — Sejour touristique")


# ── Traitement des messages texte + fichiers ──────────────────────────────


@cl.on_message
async def on_message(message: cl.Message):
    """Traite un message utilisateur : routing, streaming, badges, monitoring."""
    try:
        logger.info("Message recu : %s", message.content[:50])
        session_id = cl.user_session.get("session_id")
        question = message.content
    except Exception as exc:
        logger.error("CRASH DEBUT on_message : %s", exc)
        await cl.Message(content=f"Erreur : {exc}").send()
        return

    # Upload fichiers (PDF, DOCX, images)
    image_data = None
    doc_text = ""
    if message.elements:
        for element in message.elements:
            if element.name.endswith(".pdf"):
                await _integrer_document(element, "pdf")
                # Extraire le texte pour l'envoyer au LLM
                extracted = await _extraire_texte(element, "pdf")
                if extracted:
                    doc_text += f"\n\n[Contenu du document {element.name}]\n{extracted}"
            elif element.name.endswith(".docx"):
                await _integrer_document(element, "docx")
                extracted = await _extraire_texte(element, "docx")
                if extracted:
                    doc_text += f"\n\n[Contenu du document {element.name}]\n{extracted}"
            elif element.name.endswith(".txt"):
                try:
                    txt_bytes = (
                        element.content
                        if isinstance(element.content, bytes)
                        else open(element.path, "rb").read()
                    )
                    texte = txt_bytes.decode("utf-8", errors="ignore")[:5000]
                    doc_text += f"\n\n[Contenu du fichier {element.name}]\n{texte}"
                    logger.info("TXT recu : %s (%d car)", element.name, len(texte))
                except Exception as exc:
                    logger.error("Erreur lecture TXT : %s", exc)
            elif element.name.endswith(".doc"):
                extracted = await _extraire_texte(element, "docx")
                if extracted:
                    doc_text += f"\n\n[Contenu du document {element.name}]\n{extracted}"
            elif element.name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".webp")
            ):
                import base64

                img_bytes = (
                    element.content
                    if isinstance(element.content, bytes)
                    else open(element.path, "rb").read()
                )
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                ext = element.name.rsplit(".", 1)[-1].lower()
                mime = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "webp": "image/webp",
                }.get(ext, "image/png")
                image_data = {"base64": img_b64, "mime": mime}
                logger.info("Image recue : %s (%s)", element.name, mime)

        # Enrichir la question avec le texte du document
        if doc_text:
            if not question or not question.strip():
                question = "Resume ce document."
            question = question + doc_text

        if not question or not question.strip():
            if image_data:
                question = "Decris cette image."
            else:
                return

    # Recuperer l'historique de la session
    history = get_session_history(session_id)
    chat_history = history.messages

    # Si image uploadee, forcer route agent (multimodal)
    if image_data:
        route = "agent"
        logger.info("Image detectee -> route agent (multimodal)")
    else:
        # Etape 1 : classifier la question via le router de P3 (P3)
        try:
            state = RouterState(
                question=question,
                route="",
                answer="",
                sources=[],
                history=chat_history,
            )
            state = classify_question(state)
            route = state["route"]
            logger.info("Route detectee : %s", route)
        except Exception as exc:
            logger.error("Erreur classification : %s — fallback chat", exc)
            route = "chat"

    # Afficher le badge de route en debut de message (pattern P3)
    badge = ROUTE_LABELS.get(route, "")
    try:
        msg = cl.Message(content=f"**{badge}**\n\n")
        await msg.send()

        answer = ""
        sources = []
        agent_outils = set()
        outils_bruts = []

        # Etape 2 : traitement selon la route
        if route == "rag":
            answer, sources = await _handle_rag(msg, question)
            outils_bruts = ["search_corpus"]
        elif route == "agent":
            # Injecter le contexte utilisateur (prenom + email + localisation)
            user = cl.user_session.get("user")
            user_name = user.identifier if user else ""
            user_email = cl.user_session.get("user_email", "")
            user_location = cl.user_session.get("user_location", "")
            ctx_parts = []
            if user_name:
                ctx_parts.append(
                    f"le profil connecte est {user_name}, mais si l'utilisateur "
                    "a donne un autre prenom dans la conversation, utilise celui-la"
                )
            if any(
                kw in question.lower()
                for kw in ["email", "mail", "envoie", "envoyer", "rappel"]
            ):
                if user_email:
                    ctx_parts.append(f"l'email du profil connecte est {user_email}")
                # Repertoire contacts : charge depuis TEAM_DIRECTORY_JSON
                # (variable d'env / HF Spaces secret) — JAMAIS hardcode dans le code.
                try:
                    import json as _json

                    team_json = os.getenv("TEAM_DIRECTORY_JSON", "").strip()
                    team = _json.loads(team_json) if team_json else {}
                except Exception:
                    team = {}
                if team:
                    repertoire = ", ".join(
                        f"{prenom.capitalize()}={email}"
                        for prenom, email in team.items()
                    )
                    ctx_parts.append(
                        f"Repertoire contacts : {repertoire}. "
                        "Quand l'utilisateur dit 'envoie-moi', utilise l'email "
                        "correspondant au prenom qu'il a donne dans la conversation. "
                        "Ne fabrique JAMAIS d'adresse email."
                    )
                else:
                    ctx_parts.append(
                        "Aucun repertoire contacts configure (TEAM_DIRECTORY_JSON vide). "
                        "Demande a l'utilisateur l'adresse email exacte avant tout envoi."
                    )
            if user_location:
                ctx_parts.append(
                    f"il se trouve a {user_location}. "
                    "Utilise cette ville par defaut pour la meteo"
                )
            if ctx_parts:
                question_enrichie = (
                    f"{question}\n[Info systeme: {'; '.join(ctx_parts)}]"
                )
            else:
                question_enrichie = question
            answer, agent_outils, sources, outils_bruts = await _handle_agent(
                msg, question_enrichie, chat_history, image_data=image_data
            )
            if not outils_bruts:
                outils_bruts = ["__agent__"]
        else:
            # Injecter le prenom pour le chat aussi
            user = cl.user_session.get("user")
            if user and user.identifier:
                question_chat = (
                    f"{question}\n[Info systeme: le profil connecte est "
                    f"{user.identifier}, mais si l'utilisateur a donne "
                    "un autre prenom dans la conversation, utilise celui-la]"
                )
            else:
                question_chat = question
            answer = await _handle_chat(msg, question_chat, chat_history)

        # Mettre a jour la memoire
        add_exchange(session_id, question, answer)

        # Monitoring tokens
        tokens_info = ""
        tokens = get_token_summary()
        if tokens["total_tokens"] > 0:
            cost = tokens.get("estimated_cost_usd", 0)
            tokens_info = (
                f"Tokens : {tokens['total_tokens']} "
                f"(in: {tokens['total_input_tokens']}, "
                f"out: {tokens['total_output_tokens']}) -- "
                f"${cost:.4f} -- {get_prompt_version()}"
            )

        # Message final avec donut a gauche
        # On reprefix le badge de route car generer_message_avec_donut ecrase
        # tout le contenu (sinon le badge disparait au moment du swap vers le donut).
        badge_final = ROUTE_LABELS.get(route, "")
        donut_html = generer_message_avec_donut(
            answer=answer,
            outils_appeles=outils_bruts,
            route=route,
            sources=sources,
            tokens_info=tokens_info,
        )
        msg.content = (
            f"**{badge_final}**\n\n{donut_html}" if badge_final else donut_html
        )
        await msg.update()

    except Exception as exc:
        logger.error("CRASH on_message : %s", exc, exc_info=True)
        await cl.Message(content=f"Erreur : {exc}").send()


# ── Route RAG : streaming via chain.astream (pattern P3) ─────────────────


# Pont lexical FR/ES/DE -> EN pour le retrieval (le corpus est en anglais).
# La REPONSE reste dans la langue originale via le prompt RAG.
_BRIDGE_TERMS = {
    "GIEC": "IPCC",
    "giec": "IPCC",
    "inondation": "flood",
    "inondations": "floods",
    "secheresse": "drought",
    "sécheresse": "drought",
    "canicule": "heatwave",
    "incendie": "wildfire",
    "incendies": "wildfires",
    "feu de foret": "wildfire",
    "tempete": "storm",
    "tempête": "storm",
    "ouragan": "hurricane",
    "cyclone": "cyclone",
    "rechauffement": "warming",
    "réchauffement": "warming",
    "catastrophe": "disaster",
    "catastrophes": "disasters",
    "risque": "risk",
    "risques": "risks",
    "OMM": "WMO",
    "ONU": "UN",
    "Union europeenne": "European Union",
    "Union européenne": "European Union",
}


def _enrichir_query_multilingue(question: str) -> str:
    """Ajoute les equivalents anglais des termes FR pour ameliorer le retrieval."""
    ajouts = []
    for fr, en in _BRIDGE_TERMS.items():
        if fr in question and en not in question:
            ajouts.append(en)
    if ajouts:
        return f"{question} [{' '.join(ajouts)}]"
    return question


async def _handle_rag(msg, question: str) -> tuple[str, list]:
    """Route RAG : recherche dans le corpus puis streaming de la reponse."""
    try:
        # Enrichissement lexical pour le retrieval (GIEC->IPCC, inondations->floods, ...)
        question_retrieval = _enrichir_query_multilingue(question)
        if question_retrieval != question:
            logger.info(
                "RAG query enrichie : '%s' -> '%s'", question, question_retrieval
            )

        question_lower = question.lower()

        # Detection : l'utilisateur veut l'inventaire complet du corpus
        corpus_keywords = [
            "liste",
            "combien de doc",
            "inventaire",
            "quels doc",
        ]
        # "résume/résumer" = demande de resume de contenu, pas d'inventaire
        wants_summary = any(
            kw in question_lower
            for kw in ["résume", "resume", "résumer", "resumer", "summary", "summarize"]
        )
        is_corpus_listing = (
            any(kw in question_lower for kw in corpus_keywords) and not wants_summary
        )
        # "résume tous / chaque / l'ensemble" = resume exhaustif de tous les docs
        wants_full_summary = wants_summary and any(
            kw in question_lower
            for kw in ["tous", "chaque", "ensemble", "all", "every"]
        )

        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            error_msg = "Vector store non disponible. Lancez d'abord embeddings.py."
            msg.content += error_msg
            await msg.update()
            return error_msg, []

        # Merge eventuel FAISS session-scoped (uploads du user) avec corpus officiel
        session_store = cl.user_session.get("uploaded_docs_store")
        if session_store is not None:
            vector_store.merge_from(session_store)
            logger.info(
                "RAG : fusion corpus + %d docs session", session_store.index.ntotal
            )

        # Si inventaire demande : court-circuit retrieval, lister directement
        # les sources uniques du vectorstore (inclut les petits PDFs)
        if is_corpus_listing:
            sources_uniques = {}
            for doc in vector_store.docstore._dict.values():
                src = os.path.basename(doc.metadata.get("source", "?"))
                sources_uniques[src] = sources_uniques.get(src, 0) + 1

            inventaire_lignes = [
                f"{i+1}. {src} ({n} chunks indexes)"
                for i, (src, n) in enumerate(
                    sorted(sources_uniques.items(), key=lambda x: -x[1])
                )
            ]
            inventaire_texte = (
                f"Corpus complet : {len(sources_uniques)} documents, "
                f"{sum(sources_uniques.values())} chunks indexes.\n\n"
                + "\n".join(inventaire_lignes)
            )
            msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{inventaire_texte}"
            await msg.update()
            sources = [f"[Source: {src}]" for src in sources_uniques.keys()]
            logger.info("RAG inventaire direct : %d docs", len(sources_uniques))
            return inventaire_texte, sources

        else:
            # Fetch LARGE (k=30) puis filtrage diversite par source (max 3/PDF)
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 30, "fetch_k": 60, "lambda_mult": 0.7},
            )

        # Retrieval avec query enrichie (GIEC->IPCC etc.), mais reponse LLM sur question originale
        resultat = interroger_rag(retriever, question_retrieval)
        contexte_raw = resultat["contexte"]
        documents_raw = resultat["documents"]

        # Si resume exhaustif demande : garantir au moins 2 chunks par PDF du corpus
        if wants_full_summary:
            sources_dans_top = {
                os.path.basename(d.metadata.get("source", "?")) for d in documents_raw
            }
            # Trouver les PDFs absents du top-k et ajouter des chunks representatifs
            from collections import defaultdict

            par_source = defaultdict(list)
            for doc in vector_store.docstore._dict.values():
                src = os.path.basename(doc.metadata.get("source", "?"))
                par_source[src].append(doc)

            pdfs_manquants = set(par_source.keys()) - sources_dans_top
            for pdf in pdfs_manquants:
                # Prendre les 2 premiers chunks du PDF (souvent intro/resume)
                documents_raw.extend(par_source[pdf][:2])
            logger.info(
                "RAG resume exhaustif : %d PDFs manquants ajoutes",
                len(pdfs_manquants),
            )

        # Filtrage : max 3 chunks par PDF source pour forcer la diversite
        # et eviter qu'un seul gros document (Forest_Fires, GAR) monopolise le top-k
        if not is_corpus_listing:
            from collections import defaultdict

            # Pour resume exhaustif : 2 chunks/source, cap 24 (10 PDFs x 2 = 20)
            # Sinon : 3 chunks/source, cap 12 (diversite + pertinence)
            MAX_PAR_SOURCE = 2 if wants_full_summary else 3
            CAP_TOTAL = 24 if wants_full_summary else 12
            compteur = defaultdict(int)
            documents = []
            for doc in documents_raw:
                src = os.path.basename(doc.metadata.get("source", "inconnu"))
                if compteur[src] < MAX_PAR_SOURCE:
                    documents.append(doc)
                    compteur[src] += 1
                if len(documents) >= CAP_TOTAL:
                    break
            logger.info(
                "RAG diversite : %d -> %d chunks apres limite %d/source (cap %d)",
                len(documents_raw),
                len(documents),
                MAX_PAR_SOURCE,
                CAP_TOTAL,
            )
            # Reformater contexte avec les docs filtres
            from src.rag.retriever import formater_contexte_avec_citations

            contexte = formater_contexte_avec_citations(documents)
        else:
            documents = documents_raw
            contexte = contexte_raw

        # Sources nettoyees (pattern P3 — os.path.basename)
        sources = []
        for doc in documents:
            source = os.path.basename(doc.metadata.get("source", "inconnu"))
            page = doc.metadata.get("page", "?")
            sources.append(f"[Source: {source}, Page: {page}]")

        # Streaming token par token (pattern P3 — chain.astream)
        llm = get_llm("rag")
        chain = RAG_PROMPT | llm
        answer = ""

        async for chunk in chain.astream({"context": contexte, "question": question}):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{answer}"
                await msg.update()

        logger.info("RAG : %d sources, %d car", len(sources), len(answer))
        return answer, sources

    except Exception as exc:
        logger.error("Erreur RAG streaming : %s — fallback router P3", exc)
        # Fallback : utiliser rag_node de P3 (P3) sans streaming
        try:
            state = RouterState(
                question=question,
                route="rag",
                answer="",
                sources=[],
                history=[],
            )
            result = rag_node(state)
            answer = result["answer"]
            sources = [f"[Source: {s}]" for s in result.get("sources", [])]
            msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{answer}"
            await msg.update()
            return answer, sources
        except Exception as exc2:
            logger.error("Erreur RAG fallback P3 : %s", exc2)
            error_msg = f"Erreur RAG : {exc2}"
            msg.content += error_msg
            await msg.update()
            return error_msg, []


# ── Inventaire corpus complet (bypass retriever) ─────────────────────────


async def _handle_corpus_inventory(msg, question: str) -> tuple[str, list]:
    """Parcourt tous les PDFs du corpus et genere un resume par document."""
    import csv

    csv_path = os.path.join("outputs", "corpus_inventory.csv")
    corpus_dir = os.path.join("data", "raw")

    # Lire l'inventaire CSV
    docs_info = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs_info.append(row)

    if not docs_info:
        # Fallback : scanner le dossier
        for f in sorted(os.listdir(corpus_dir)):
            if f.lower().endswith(".pdf"):
                size_mb = round(
                    os.path.getsize(os.path.join(corpus_dir, f)) / (1024 * 1024), 1
                )
                docs_info.append({"fichier": f, "taille_mo": size_mb, "pages": "?"})

    # Extraire la premiere page de chaque PDF pour le resume
    summaries = []
    for doc in docs_info:
        path = os.path.join(corpus_dir, doc["fichier"])
        try:
            from langchain_community.document_loaders import PyPDFLoader

            pages = PyPDFLoader(path).load()
            first_page = pages[0].page_content[:500] if pages else ""
        except Exception:
            first_page = ""
        summaries.append(
            f"{doc['fichier']} ({doc['taille_mo']} Mo, {doc['pages']} pages) : "
            f"{first_page[:200]}..."
        )

    # Construire le contexte et streamer la reponse
    contexte = "\n\n".join(
        f"Document {i+1}/{len(summaries)} : {s}" for i, s in enumerate(summaries)
    )

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es DooMax, l'IA du systeme SAEARCH. "
                "Voici l'integralite des {nb_docs} documents de votre corpus. "
                "Resume chacun en 1-2 phrases. "
                "Cite la source [Source: fichier, Page: 1] pour chaque document.\n\n"
                "Documents :\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    llm = get_llm("rag")
    chain = prompt | llm
    answer = ""

    async for chunk in chain.astream(
        {"context": contexte, "question": question, "nb_docs": len(summaries)}
    ):
        if hasattr(chunk, "content") and chunk.content:
            answer += chunk.content
            msg.content = f"**{ROUTE_LABELS['rag']}**\n\n{answer}"
            await msg.update()

    sources = [f"[Source: {d['fichier']}, Page: 1]" for d in docs_info]
    logger.info("Inventaire corpus : %d docs, %d car", len(docs_info), len(answer))
    return answer, sources


# ── Route Agent : agent ReAct 9 outils (pattern P4) ──────────────────────


async def _handle_agent(
    msg, question: str, chat_history: list, image_data: dict = None
) -> tuple[str, set, list, list]:
    """Route Agent : agent ReAct avec streaming astream_events."""
    agent = get_agent()

    # Construire le message (texte ou multimodal)
    if image_data:
        human_content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_data['mime']};base64,{image_data['base64']}"
                },
            },
        ]
        human_msg = HumanMessage(content=human_content)
    else:
        human_msg = HumanMessage(content=question)

    messages = (chat_history or []) + [human_msg]

    answer = ""
    all_messages = []
    # Capture directe des tool_calls via les events on_tool_* (fallback robuste)
    tools_called_streaming = []

    event_kinds_seen = set()
    try:
        async for event in agent.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]
            event_kinds_seen.add(kind)

            # Capture tool calls dans les chunks AIMessage du stream
            if kind == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output and hasattr(output, "tool_calls") and output.tool_calls:
                    for tc in output.tool_calls:
                        tname = (
                            tc.get("name", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "")
                        )
                        if tname:
                            tools_called_streaming.append(tname)
                            logger.info(
                                "Tool detecte via on_chat_model_end : %s", tname
                            )

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        answer += "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    else:
                        answer += content
                    msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
                    await msg.update()

            # Capture des appels d'outils pendant le streaming
            if kind == "on_tool_start":
                tool_name = event.get("name", "")
                if tool_name:
                    tools_called_streaming.append(tool_name)
                    logger.info("Tool streaming detecte : %s", tool_name)

            if kind == "on_chain_end" and "messages" in event.get("data", {}).get(
                "output", {}
            ):
                all_messages = event["data"]["output"]["messages"]
                # Scanner aussi pour tool_calls direct
                for m in all_messages:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            tname = (
                                tc.get("name", "")
                                if isinstance(tc, dict)
                                else getattr(tc, "name", "")
                            )
                            if tname and tname not in tools_called_streaming:
                                tools_called_streaming.append(tname)
                                logger.info(
                                    "Tool detecte via on_chain_end messages : %s", tname
                                )

    except Exception as exc:
        logger.error("Erreur Agent streaming : %s", exc)

    # Fallback 1 : appel synchrone agent P4
    if not answer:
        try:
            from src.agents.agent import run_agent

            answer = run_agent(question, chat_history=chat_history)
            msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
            await msg.update()
        except Exception as exc:
            logger.warning("Agent P4 echoue : %s — fallback router P3", exc)
            # Fallback 2 : router complet de P3 (P3)
            try:
                state = RouterState(
                    question=question,
                    route="agent",
                    answer="",
                    sources=[],
                    history=chat_history or [],
                )
                result = jayson_router.invoke(state)
                answer = result["answer"]
                msg.content = f"**{ROUTE_LABELS['agent']}**\n\n{answer}"
                await msg.update()
            except Exception as exc2:
                logger.error("Erreur Router P3 fallback : %s", exc2)
                answer = f"Erreur Agent : {exc2}"
                msg.content += answer
                await msg.update()

    # Log debug des events vus (pour diagnostiquer detection outils)
    logger.info("Events LangGraph vus : %s", sorted(event_kinds_seen))

    # Detecter uniquement les outils/sources DU TOUR courant
    # (all_messages contient chat_history complet + nouvelles reponses, on slice)
    nouveaux_messages = all_messages[len(messages) :] if all_messages else []
    agent_outils, sources, outils_bruts = _detecter_outils_appeles(nouveaux_messages)

    # Fallback : si detection via messages a echoue, utiliser la capture streaming
    if not outils_bruts and tools_called_streaming:
        outils_bruts = tools_called_streaming
        agent_outils = {
            TOOL_BADGES[t] for t in tools_called_streaming if t in TOOL_BADGES
        }
        logger.info("Fallback streaming : %d outils recuperes", len(outils_bruts))

    logger.info("Agent : %d outils, %d car", len(agent_outils), len(answer))
    return answer, agent_outils, sources, outils_bruts


# ── Route Chat : streaming via chain.astream (pattern P3) ────────────────


async def _handle_chat(msg, question: str, chat_history: list) -> str:
    """Route Chat : conversation directe avec streaming."""
    from datetime import datetime

    llm = get_llm("chat")
    chain = CHAT_PROMPT | llm
    now = datetime.now().strftime("%A %d %B %Y, %H:%M (heure locale)")
    answer = ""

    try:
        async for chunk in chain.astream(
            {"question": question, "history": chat_history, "current_time": now}
        ):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                msg.content = f"**{ROUTE_LABELS['chat']}**\n\n{answer}"
                await msg.update()
    except Exception as exc:
        logger.error("Erreur Chat streaming : %s — fallback router P3", exc)
        # Fallback : utiliser chat_node de P3 (P3) sans streaming
        try:
            state = RouterState(
                question=question,
                route="chat",
                answer="",
                sources=[],
                history=chat_history,
            )
            result = chat_node(state)
            answer = result["answer"]
            msg.content = f"**{ROUTE_LABELS['chat']}**\n\n{answer}"
            await msg.update()
        except Exception as exc2:
            logger.error("Erreur Chat fallback P3 : %s", exc2)
            answer = f"Erreur Chat : {exc2}"
            msg.content += answer
            await msg.update()

    logger.info("Chat : %d car", len(answer))
    return answer


# ── Upload document dynamique (PDF + DOCX) ───────────────────────────────


async def _extraire_texte(element, doc_type: str = "pdf") -> str:
    """Extrait le texte brut d'un PDF ou DOCX pour l'envoyer au LLM."""
    try:
        suffix = f".{doc_type}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(
                element.content
                if isinstance(element.content, bytes)
                else open(element.path, "rb").read()
            )
            tmp_path = tmp.name

        if doc_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(tmp_path)
        elif doc_type == "docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(tmp_path)
        else:
            return ""

        pages = loader.load()
        # Limiter a 5000 caracteres pour ne pas exploser le contexte
        texte = "\n".join(p.page_content for p in pages)[:5000]
        logger.info(
            "Texte extrait de %s : %d car (%d pages)",
            element.name,
            len(texte),
            len(pages),
        )
        return texte
    except Exception as exc:
        logger.error("Erreur extraction texte %s : %s", element.name, exc)
        return ""


async def _integrer_document(element, doc_type: str = "pdf") -> None:
    """Integre un PDF ou DOCX uploade dans le corpus RAG en temps reel."""
    logger.info("Upload %s recu : %s", doc_type.upper(), element.name)

    try:
        from src.config import FAISS_STORE_PATH
        from src.rag.embeddings import charger_vector_store
        from src.rag.loader import decouper_documents

        suffix = f".{doc_type}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(
                element.content
                if isinstance(element.content, bytes)
                else open(element.path, "rb").read()
            )
            tmp_path = tmp.name

        if doc_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(tmp_path)
        elif doc_type == "docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(tmp_path)
        else:
            await cl.Message(content=f"Format non supporte : {doc_type}").send()
            return

        pages = loader.load()
        chunks = decouper_documents(pages)

        # Ajout dans un FAISS SESSION-SCOPED (en memoire uniquement)
        # pour ne pas polluer le corpus officiel faiss_store/ sur disque
        for chunk in chunks:
            chunk.metadata["source"] = element.name  # vrai nom au lieu du tmpXXX
            chunk.metadata["session_upload"] = True

        session_store = cl.user_session.get("uploaded_docs_store")
        if session_store is None:
            from langchain_community.vectorstores import FAISS as FAISS_cls
            from langchain_huggingface import HuggingFaceEmbeddings
            from src.config import EMBEDDING_MODEL

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            session_store = FAISS_cls.from_documents(chunks, embeddings)
        else:
            session_store.add_documents(chunks)
        cl.user_session.set("uploaded_docs_store", session_store)

        logger.info(
            "%s %s integre en SESSION : %d pages, %d chunks (pas de save disque)",
            doc_type.upper(),
            element.name,
            len(pages),
            len(chunks),
        )
        await cl.Message(
            content=(
                f"**Document integre (session uniquement)** : {element.name}\n"
                f"- {len(pages)} pages lues\n"
                f"- {len(chunks)} passages indexes\n"
                f"- Disponible pour cette session, non persiste sur disque"
            )
        ).send()

    except Exception as exc:
        logger.error("Erreur integration PDF : %s", exc)
        await cl.Message(
            content=f"Erreur lors de l'integration de {element.name} : {exc}"
        ).send()


# ── Speech-to-text (optionnel — depend de la version Chainlit) ────────────


try:

    @cl.on_audio_start
    async def on_audio_start():
        """Demarre l'enregistrement audio."""
        return True

    @cl.on_audio_chunk
    async def on_audio_chunk(chunk):
        """Accumule les chunks audio."""
        if chunk.isStart:
            cl.user_session.set("audio_buffer", b"")
            cl.user_session.set("audio_mime", chunk.mimeType)

        buffer = cl.user_session.get("audio_buffer")
        cl.user_session.set("audio_buffer", buffer + chunk.data)

    @cl.on_audio_end
    async def on_audio_end():
        """Transcrit l'audio puis traite comme un message texte."""
        audio_buffer = cl.user_session.get("audio_buffer")
        if not audio_buffer:
            return

        logger.info("Audio recu : %d octets", len(audio_buffer))

        try:
            from faster_whisper import WhisperModel

            # Chainlit envoie du PCM16 brut — convertir en WAV
            import wave

            audio_mime = cl.user_session.get("audio_mime", "unknown")
            logger.info("Format audio : %s", audio_mime)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                with wave.open(tmp.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                    wav_file.setframerate(24000)  # sample_rate du config
                    wav_file.writeframes(audio_buffer)
                tmp_path = tmp.name

            model = WhisperModel("small", device="cpu", compute_type="int8")
            # language=None → détection automatique (FR, EN, ES, DE, etc.)
            segments, info = model.transcribe(tmp_path, language=None)
            logger.info("Langue detectee par Whisper : %s (proba %.0f%%)", info.language, info.language_probability * 100)
            text = " ".join(segment.text for segment in segments)

            logger.info("Transcription : %s", text[:100])

            if text.strip():
                await cl.Message(
                    content=f"*Transcription : {text}*", author="user"
                ).send()
                fake_message = cl.Message(content=text)
                await on_message(fake_message)
            else:
                await cl.Message(content="Aucun texte detecte dans l'audio.").send()

        except ImportError:
            await cl.Message(
                content="STT non disponible. Installez : pip install faster-whisper"
            ).send()
        except Exception as exc:
            logger.error("Erreur STT : %s", exc)
            await cl.Message(content=f"Erreur de transcription : {exc}").send()

except (AttributeError, KeyError):
    logger.info("Speech-to-text non supporte par cette version de Chainlit")
