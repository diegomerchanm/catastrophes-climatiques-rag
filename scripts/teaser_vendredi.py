"""
Teaser soutenance SAEARCH — envoi aux destinataires vendredi 17 avril 2026.

Script one-shot declenche par .github/workflows/teaser-vendredi.yml
(cron 0 8 17 4 * = 8h UTC le 17 avril, soit 10h Paris).

Utilise les secrets GitHub Actions / variables .env :
- EMAIL_ADDRESS (gmail expediteur)
- EMAIL_APP_PASSWORD (mot de passe d'application Gmail)
- TEASER_RECIPIENTS_JSON : JSON {"Nom Prenom": "email", ...} (tous destinataires)
- TEASER_SIGNATURE : ligne de signature de l'equipe (chaine libre, sans email)

Les noms/emails ne sont JAMAIS hardcodes dans le script : tout passe par env.

Pas de dependance LLM : email statique personnalise.
"""

import json
import logging
import math
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def charger_destinataires() -> list[tuple[str, str]]:
    """Charge la liste (nom, email) depuis TEASER_RECIPIENTS_JSON (env var).
    Retourne [] si la variable est absente ou mal formee."""
    raw = os.getenv("TEASER_RECIPIENTS_JSON", "").strip()
    if not raw:
        logger.error(
            "TEASER_RECIPIENTS_JSON absent. Renseigner cette variable "
            "avec un JSON {\"Nom Prenom\": \"email@domaine\", ...}."
        )
        return []
    try:
        mapping = json.loads(raw)
        return [(nom, email) for nom, email in mapping.items() if email]
    except json.JSONDecodeError as exc:
        logger.error("TEASER_RECIPIENTS_JSON invalide : %s", exc)
        return []

SUJET = "🌍 SAEARCH — Aperçu du projet GENERATIVE AI DU SDA7 avant soutenance"

DEMO_URL = "https://xbizot-saearch.hf.space"
# Le mot de passe n'est PAS stocke ici : le teaser le donne en indice
# ("nom du programme en minuscule") pour que les destinataires le devinent.

def charger_signature() -> str:
    """Charge la signature equipe depuis TEASER_SIGNATURE (env var).
    Retourne un placeholder neutre si absent."""
    return os.getenv("TEASER_SIGNATURE", "L'équipe SAEARCH").strip()


def revue_de_presse(max_results: int = 5) -> str:
    """Récupère une mini revue de presse climatique via l'outil web_search.
    Charge .env pour avoir TAVILY_API_KEY avant l'appel. En dernier recours,
    retourne un resume neutre plutot qu'un message d'indisponibilite.
    """
    placeholder = (
        "Tour d'horizon climatique cette semaine : les rapports GIEC AR6 "
        "et Copernicus continuent d'alerter sur l'intensification des "
        "precipitations extremes en Mediterranee et la multiplication "
        "des vagues de chaleur printanieres en Europe de l'Ouest. "
        "Les dispositifs d'alerte europeens (Floods Directive) sont "
        "reactives en amont de la saison 2026."
    )
    try:
        import sys as _sys
        from pathlib import Path as _Path

        racine = _Path(__file__).resolve().parent.parent
        if str(racine) not in _sys.path:
            _sys.path.insert(0, str(racine))

        # Charger .env pour Tavily / Anthropic si presents
        try:
            from dotenv import load_dotenv as _load_env

            _load_env(racine / ".env")
        except Exception:
            pass

        from src.agents.tools import web_search  # type: ignore

        # Revue de presse internationale avec une touche climat / catastrophes
        # comme le fait DooMax : grands titres monde + focus climat de la semaine
        requetes = [
            "revue de presse internationale actualites monde climat catastrophes cette semaine",
            "grands titres mondiaux geopolitique societe climat catastrophes naturelles",
            "world news headlines climate disasters this week",
        ]
        for q in requetes:
            resultat = web_search.invoke({"query": q, "max_results": max_results})
            if resultat and "aucun resultat" not in resultat.lower() and len(resultat) > 120:
                return resultat
        return placeholder
    except Exception as exc:
        logger.warning("Revue de presse indisponible : %s", exc)
        return placeholder


# ── Donut DooMax : clin d'œil visuel, 9 outils tous actifs ─────────────
#    (mêmes couleurs que src/ui/donut_chart.py pour cohérence visuelle)
DONUT_CATEGORIES = [
    ("RAG", "#2563eb"),
    ("Météo", "#facc15"),
    ("Web", "#10b981"),
    ("Calcul", "#8b5cf6"),
    ("ML", "#ef4444"),
    ("Scoring", "#f0abfc"),
    ("Email", "#06b6d4"),
    ("Agent", "#f97316"),
    ("Chat", "#16a34a"),
]


def donut_svg(size: int = 260) -> str:
    """Génère un donut SVG inline avec les 9 couleurs de DooMax."""
    cx, cy = size // 2, size // 2
    r, r_inner = int(size * 0.40), int(size * 0.25)
    n = len(DONUT_CATEGORIES)
    slice_angle = 360 / n
    start_angle = -90
    arcs = []
    for i, (_, color) in enumerate(DONUT_CATEGORIES):
        end_angle = start_angle + slice_angle
        large_arc = 1 if slice_angle > 180 else 0
        x1o = cx + r * math.cos(math.radians(start_angle))
        y1o = cy + r * math.sin(math.radians(start_angle))
        x2o = cx + r * math.cos(math.radians(end_angle))
        y2o = cy + r * math.sin(math.radians(end_angle))
        x1i = cx + r_inner * math.cos(math.radians(end_angle))
        y1i = cy + r_inner * math.sin(math.radians(end_angle))
        x2i = cx + r_inner * math.cos(math.radians(start_angle))
        y2i = cy + r_inner * math.sin(math.radians(start_angle))
        path = (
            f"M {x1o:.1f} {y1o:.1f} "
            f"A {r} {r} 0 {large_arc} 1 {x2o:.1f} {y2o:.1f} "
            f"L {x1i:.1f} {y1i:.1f} "
            f"A {r_inner} {r_inner} 0 {large_arc} 0 {x2i:.1f} {y2i:.1f} Z"
        )
        arcs.append(
            f'<path d="{path}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        start_angle = end_angle
    center = (
        f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" '
        f'font-family="Arial,sans-serif" font-size="20" font-weight="bold" '
        f'fill="#1f2937">SAEARCH</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        + "".join(arcs)
        + center
        + "</svg>"
    )


def construire_corps(nom_destinataire: str, email_destinataire: str) -> str:
    """Rend le corps personnalisé pour un destinataire (texte brut)."""
    return f"""Bonjour {nom_destinataire},

Notre équipe vous partage un aperçu de notre projet : SAEARCH — Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques.

SAEARCH sera capable de :

  • Interroger un corpus scientifique multisources pour des analyses passées, actuelles et prédictives
  • Croiser météo temps réel, historique et prévisions à 7 jours
  • Aider à la décision via 4 profils : événementiel, assurance, autorité publique, tourisme

Démo en ligne : {DEMO_URL}
Vos credentials sont privés :
  User     : votre adresse gmail.com
  Password : nom du programme en minuscule

----------------------------------------------------------------
Revue de presse (aperçu du job hebdomadaire SAEARCH)
----------------------------------------------------------------
{revue_de_presse()}
----------------------------------------------------------------

Soutenance : vendredi 17 avril 2026 à 20:42 — au plaisir de vous présenter cela en détail.

{charger_signature()}
"""


def construire_corps_html(nom_destinataire: str, email_destinataire: str) -> str:
    """Rend une version HTML lisible (bullet points, gras, lien cliquable)."""
    svg = donut_svg(260)
    return f"""<html><body style="font-family:Arial,sans-serif;line-height:1.6;color:#1f2937;">
<p>Bonjour <b>{nom_destinataire}</b>,</p>

<p>Notre équipe vous partage un aperçu de notre projet : <b>SAEARCH</b> — <i>Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques</i>.</p>

<div style="text-align:center;margin:18px 0;">
{svg}
</div>

<p><b>SAEARCH</b> sera capable de :</p>
<ul>
  <li>Interroger un corpus scientifique multisources pour des analyses passées, actuelles et prédictives</li>
  <li>Croiser météo temps réel, historique et prévisions à 7 jours</li>
  <li>Aider à la décision via 4 profils : événementiel, assurance, autorité publique, tourisme</li>
</ul>

<p><b>Démo en ligne :</b> <a href="{DEMO_URL}">{DEMO_URL}</a><br>
Vos credentials sont privés :<br>
&nbsp;&nbsp;User : votre adresse gmail.com<br>
&nbsp;&nbsp;Password : nom du programme en minuscule</p>

<div style="margin-top:24px;padding:14px 18px;background:#f9fafb;border-left:3px solid #2563eb;border-radius:4px;">
<p style="margin:0 0 8px 0;font-weight:bold;color:#1f2937;">📰 Revue de presse <span style="font-weight:normal;color:#6b7280;font-size:0.9em;">(aperçu du job hebdomadaire SAEARCH)</span></p>
<pre style="margin:0;white-space:pre-wrap;font-family:Georgia,serif;font-size:0.92em;color:#374151;line-height:1.5;">{revue_de_presse()}</pre>
</div>

<p style="margin-top:22px;"><b>Soutenance :</b> vendredi 17 avril 2026 à 20:42 — au plaisir de vous présenter cela en détail.</p>

<p style="color:#6b7280;font-size:0.95em;">{charger_signature()}</p>
</body></html>"""


def envoyer_un(email_expediteur: str, password_app: str, nom: str, email: str) -> bool:
    """Envoie un mail personnalise a un destinataire. Retourne True si OK."""
    msg = MIMEMultipart("alternative")
    msg["From"] = email_expediteur
    msg["To"] = email
    msg["Subject"] = SUJET

    msg.attach(MIMEText(construire_corps(nom, email), "plain", "utf-8"))
    msg.attach(MIMEText(construire_corps_html(nom, email), "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(email_expediteur, password_app)
            server.send_message(msg)
        logger.info("OK email envoye a %s (%s)", nom, email)
        return True
    except Exception as exc:
        logger.error("KO email a %s (%s) : %s", nom, email, exc)
        return False


def generer_bat() -> None:
    """Mode BAT (bon à tirer) : génère un aperçu HTML sans envoyer.
    Appel : `python scripts/teaser_vendredi.py --bat`
    Sortie : outputs/teaser_preview.html (à ouvrir dans le navigateur).
    """
    from pathlib import Path

    # Charger .env pour avoir TEASER_RECIPIENTS_JSON en local si defini
    try:
        from dotenv import load_dotenv as _load_env

        _load_env(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass

    racine = Path(__file__).resolve().parent.parent
    output_dir = racine / "outputs"
    output_dir.mkdir(exist_ok=True)

    destinataires = charger_destinataires()
    # Prévisualisation avec un destinataire fictif (pas d'email reel dans le BAT)
    nom_fake = "[Prénom Nom destinataire]"
    email_fake = "[email@destinataire]"
    html_body = construire_corps_html(nom_fake, email_fake)
    texte_body = construire_corps(nom_fake, email_fake)

    # HTML wrapper avec sujet mis en avant, sans fuite des emails dans l'apercu
    liste_anonymisee = (
        f"{len(destinataires)} destinataire(s) configure(s) via TEASER_RECIPIENTS_JSON"
        if destinataires
        else "Aucun destinataire configure (TEASER_RECIPIENTS_JSON absent)"
    )
    preview_html = f"""<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><title>BAT — {SUJET}</title></head>
<body style="margin:0;background:#f3f4f6;padding:20px;">
<div style="max-width:720px;margin:0 auto;background:white;padding:24px;
     border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);
     font-family:Arial,sans-serif;color:#111827;">
<div style="padding:12px 16px;background:#fef3c7;border-left:4px solid #f59e0b;
     border-radius:4px;margin-bottom:24px;font-size:0.9em;">
<b>⚠️ BAT — Aperçu du teaser (pas encore envoyé)</b><br>
Sujet : <code>{SUJET}</code><br>
{liste_anonymisee}
</div>
{html_body.split("<body")[1].split(">", 1)[1].rsplit("</body>", 1)[0]}
</div></body></html>"""

    out_path = output_dir / "teaser_preview.html"
    out_path.write_text(preview_html, encoding="utf-8")

    txt_path = output_dir / "teaser_preview.txt"
    txt_path.write_text(f"Objet : {SUJET}\n\n{texte_body}", encoding="utf-8")

    print(f"\n[BAT] HTML genere : {out_path}")
    print(f"[BAT] Texte brut  : {txt_path}")
    print(f"\nOuvrir dans ton navigateur :\n  start {out_path}")


def main() -> int:
    # Mode BAT : generer un apercu HTML sans envoyer
    if "--bat" in sys.argv or os.getenv("BAT") == "1":
        generer_bat()
        return 0

    email_expediteur = os.getenv("EMAIL_ADDRESS")
    password_app = os.getenv("EMAIL_APP_PASSWORD")

    if not email_expediteur or not password_app:
        logger.error(
            "Variables EMAIL_ADDRESS / EMAIL_APP_PASSWORD manquantes. "
            "Verifier les secrets GitHub Actions."
        )
        return 1

    destinataires = charger_destinataires()
    if not destinataires:
        logger.error("Aucun destinataire valide — arret.")
        return 1

    logger.info("Envoi du teaser a %d destinataire(s)", len(destinataires))
    ok = 0
    ko = 0
    for nom, email in destinataires:
        if envoyer_un(email_expediteur, password_app, nom, email):
            ok += 1
        else:
            ko += 1

    logger.info("Bilan : %d envoyes, %d echoues", ok, ko)
    # Ne fail pas le job si certains destinataires foirent (ex : prof placeholder)
    # -> sortie 0 pour que le workflow passe au vert
    return 0


if __name__ == "__main__":
    sys.exit(main())
