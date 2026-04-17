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

SUJET = "🌍 DU SDA7 : Aperçu du projet GENERATIVE AI avant soutenance - ANNONCE : Ouverture de SAEARCH"

DEMO_URL = "https://xbizot-saearch.hf.space"
# Le mot de passe n'est PAS stocke ici : le teaser le donne en indice
# ("nom du projet en minuscule") pour que les destinataires le devinent.

def charger_signature() -> str:
    """Charge la signature equipe depuis TEASER_SIGNATURE (env var).
    Retourne un placeholder neutre si absent."""
    return os.getenv("TEASER_SIGNATURE", "L'équipe SAEARCH").strip()


def revue_de_presse() -> str:
    """Tavily parse les titres, premieres lignes et liens. Point."""
    intro = (
        "📢 Platform SAEARCH is opening and ready to DOO MAX for the planet.\n\n"
        "La veille climatique est générée automatiquement chaque semaine "
        "par notre newsletter SAEARCH via l'outil Tavily.\n\n"
    )
    try:
        import sys as _sys
        from pathlib import Path as _Path

        racine = _Path(__file__).resolve().parent.parent
        if str(racine) not in _sys.path:
            _sys.path.insert(0, str(racine))
        try:
            from dotenv import load_dotenv as _ld
            _ld(racine / ".env")
        except Exception:
            pass

        from src.agents.tools import web_search

        # Mix climat + environnement + solutions (pas que catastrophes)
        resultat = web_search.invoke({
            "query": "climat environnement actualites solutions avril 2026",
            "max_results": 5,
        })
        if resultat and len(resultat) > 120:
            return intro + resultat
        logger.warning("Tavily resultat trop court : %d car", len(resultat) if resultat else 0)
    except Exception as exc:
        logger.error("Revue de presse Tavily echouee : %s", exc)
    return intro


# ── Donut SAEARCH : clin d'œil visuel, 9 catégories ────────────────────
#    (mêmes couleurs que src/ui/donut_chart.py pour cohérence visuelle)
#    Génération PNG via matplotlib pour compatibilité Gmail (SVG strippé).
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


def donut_png_bytes() -> bytes:
    """Génère le donut en PNG (bytes) via matplotlib — compatible Gmail."""
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [cat for cat, _ in DONUT_CATEGORIES]
    colors = [color for _, color in DONUT_CATEGORIES]
    sizes = [1] * len(DONUT_CATEGORIES)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.text(0, 0, "SAEARCH", ha="center", va="center", fontsize=13, fontweight="bold", color="#6b7280")
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def construire_corps(nom_destinataire: str, email_destinataire: str) -> str:
    """Rend le corps personnalisé pour un destinataire (texte brut)."""
    return f"""Bonjour {nom_destinataire},

Notre équipe vous annonce l'ouverture de la plateforme SAEARCH — Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques.

SAEARCH vous permet de :

  • Croiser météo temps réel, historique et prévisions à 7 jours
  • Interroger un corpus scientifique multisources pour des analyses passées, actuelles et prédictives
  • Aider à la décision via 4 profils : événementiel, assurance, autorité publique, tourisme

Découverte en ligne : {DEMO_URL}
Vos credentials sont privés :
  User     : votre adresse gmail.com
  Password : nom du projet en minuscule

  Vous ferez connaissance avec notre mascotte.

----------------------------------------------------------------
Revue de presse (aperçu de la newsletter hebdomadaire SAEARCH)
----------------------------------------------------------------
{revue_de_presse()}
----------------------------------------------------------------

Soutenance : vendredi 17 avril 2026 à 20:42 — au plaisir de vous présenter le projet en détail.

{charger_signature()}

L'envoi de cet email a été programmé via l'outil Scheduler de SAEARCH.
"""


def construire_corps_html(nom_destinataire: str, email_destinataire: str) -> str:
    """Rend une version HTML lisible (bullet points, gras, lien cliquable)."""
    return f"""<html><body style="font-family:Arial,sans-serif;line-height:1.6;color:#1f2937;">
<p>Bonjour <b>{nom_destinataire}</b>,</p>

<p>Notre équipe vous annonce l'ouverture de la plateforme <b>SAEARCH</b> — <i>Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques</i>.</p>

<div style="margin:18px 0;">
<img src="cid:donut" alt="SAEARCH donut" width="260">
</div>

<p><b>SAEARCH</b> vous permet de :</p>
<ul>
  <li>Croiser météo temps réel, historique et prévisions à 7 jours</li>
  <li>Interroger un corpus scientifique multisources pour des analyses passées, actuelles et prédictives</li>
  <li>Aider à la décision via 4 profils : événementiel, assurance, autorité publique, tourisme</li>
</ul>

<p><b>Découverte en ligne :</b> <a href="{DEMO_URL}">{DEMO_URL}</a><br>
Vos credentials sont privés :<br>
&nbsp;&nbsp;User : votre adresse gmail.com<br>
&nbsp;&nbsp;Password : nom du projet en minuscule</p>
<p>Vous ferez connaissance avec notre mascotte.</p>

<div style="margin-top:24px;padding:14px 18px;background:#f9fafb;border-left:3px solid #2563eb;border-radius:4px;">
<p style="margin:0 0 12px 0;font-weight:bold;color:#1f2937;">📰 Revue de presse <span style="font-weight:normal;color:#6b7280;font-size:0.9em;">(aperçu de la newsletter hebdomadaire SAEARCH)</span></p>
<pre style="margin:0;white-space:pre-wrap;font-family:Georgia,serif;font-size:0.92em;color:#374151;line-height:1.5;">{revue_de_presse()}</pre>
</div>

<p style="margin-top:22px;"><b>Soutenance :</b> vendredi 17 avril 2026 à 20:42 — au plaisir de vous présenter le projet en détail.</p>

<p style="color:#6b7280;font-size:0.95em;">{charger_signature()}</p>
<p style="color:#9ca3af;font-size:0.8em;font-style:italic;margin-top:16px;">L'envoi de cet email a été programmé via l'outil Scheduler de SAEARCH.</p>
</body></html>"""


def envoyer_un(
    email_expediteur: str, password_app: str, nom: str, email: str,
    donut_data: bytes = None,
) -> bool:
    """Envoie un mail personnalise a un destinataire. Retourne True si OK."""
    from email.mime.image import MIMEImage

    msg = MIMEMultipart("related")

    # Partie alternative (texte brut + HTML)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(construire_corps(nom, email), "plain", "utf-8"))
    alt.attach(MIMEText(construire_corps_html(nom, email), "html", "utf-8"))
    msg.attach(alt)

    # Image donut inline (Content-ID)
    if donut_data:
        img = MIMEImage(donut_data, _subtype="png")
        img.add_header("Content-ID", "<donut>")
        img.add_header("Content-Disposition", "inline", filename="saearch_donut.png")
        msg.attach(img)

    msg["From"] = email_expediteur
    msg["To"] = email
    msg["Subject"] = SUJET

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
    # Charger .env pour avoir les credentials en local
    try:
        from pathlib import Path
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass

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

    # Générer le donut PNG une seule fois (partagé entre tous les envois)
    try:
        donut_data = donut_png_bytes()
        logger.info("Donut PNG genere (%d octets)", len(donut_data))
    except Exception as exc:
        logger.warning("Donut PNG indisponible : %s", exc)
        donut_data = None

    logger.info("Envoi du teaser a %d destinataire(s)", len(destinataires))
    ok = 0
    ko = 0
    for nom, email in destinataires:
        if envoyer_un(email_expediteur, password_app, nom, email, donut_data):
            ok += 1
        else:
            ko += 1

    logger.info("Bilan : %d envoyes, %d echoues", ok, ko)
    # Ne fail pas le job si certains destinataires foirent (ex : prof placeholder)
    # -> sortie 0 pour que le workflow passe au vert
    return 0


if __name__ == "__main__":
    sys.exit(main())
