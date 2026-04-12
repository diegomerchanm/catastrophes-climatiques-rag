"""
Rapport hebdomadaire automatique — exécuté par GitHub Actions cron le lundi matin.
1. Recherche web : actualités catastrophes climatiques de la semaine
2. Prévisions météo : villes à risque pour la semaine à venir
3. Croisement avec les seuils du corpus GIEC
4. Envoi du rapport par email
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from src.agents.tools import get_forecast, search_corpus, send_email, web_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Villes surveillées ────────────────────────────────────────────────────

VILLES_SURVEILLEES = [
    "Marseille",
    "Nice",
    "Lyon",
    "Paris",
    "Bordeaux",
]

# ── Seuils d'alerte (mm de précipitations sur 24h) ───────────────────────

SEUIL_PLUIE_ALERTE = 80  # mm — alerte modérée
SEUIL_PLUIE_CRITIQUE = 100  # mm — alerte critique


def generer_rapport() -> str:
    """Génère le rapport hebdomadaire complet."""
    date_rapport = datetime.now().strftime("%d/%m/%Y")
    sections = [f"RAPPORT HEBDOMADAIRE — Semaine du {date_rapport}\n{'=' * 60}\n"]

    # 1. Revue de presse
    logger.info("1. Revue de presse web...")
    actualites = web_search.invoke({
        "query": "catastrophes climatiques Europe semaine dernière",
        "max_results": 5,
    })
    sections.append(f"\n1. REVUE DE PRESSE\n{'-' * 40}\n{actualites}\n")

    # 2. Prévisions par ville
    logger.info("2. Prévisions météo par ville...")
    alertes = []
    sections.append(f"\n2. PRÉVISIONS MÉTÉO (7 jours)\n{'-' * 40}\n")

    for ville in VILLES_SURVEILLEES:
        previsions = get_forecast.invoke({"city": ville})
        sections.append(f"\n{ville} :\n{previsions}\n")

        # Vérifier les seuils dans les prévisions
        if "mm" in previsions:
            for ligne in previsions.split("\n"):
                if "pluie" in ligne.lower():
                    try:
                        mm_str = ligne.split("pluie")[1].split("mm")[0].strip()
                        mm = float(mm_str)
                        if mm >= SEUIL_PLUIE_CRITIQUE:
                            alertes.append(
                                f"CRITIQUE — {ville} : {mm}mm prévus (seuil {SEUIL_PLUIE_CRITIQUE}mm)"
                            )
                        elif mm >= SEUIL_PLUIE_ALERTE:
                            alertes.append(
                                f"ALERTE — {ville} : {mm}mm prévus (seuil {SEUIL_PLUIE_ALERTE}mm)"
                            )
                    except (ValueError, IndexError):
                        pass

    # 3. Croisement avec le corpus GIEC
    logger.info("3. Croisement avec le corpus GIEC...")
    if alertes:
        contexte_giec = search_corpus.invoke({
            "question": "Quels sont les seuils critiques de précipitations pour les inondations en Méditerranée et en France selon le GIEC ?"
        })
        sections.append(f"\n3. CROISEMENT CORPUS GIEC\n{'-' * 40}\n{contexte_giec}\n")

    # 4. Alertes
    sections.append(f"\n4. ALERTES\n{'-' * 40}\n")
    if alertes:
        for alerte in alertes:
            sections.append(f"  {alerte}")
    else:
        sections.append("  Aucune alerte cette semaine.")

    rapport = "\n".join(sections)
    return rapport


def envoyer_rapport(rapport: str) -> None:
    """Envoie le rapport par email."""
    destinataire = os.getenv("EMAIL_RECIPIENT")
    if not destinataire:
        logger.warning("EMAIL_RECIPIENT non configuré, rapport non envoyé")
        print(rapport)
        return

    date_rapport = datetime.now().strftime("%d/%m/%Y")
    sujet = f"Rapport climatique hebdomadaire — {date_rapport}"

    result = send_email.invoke({
        "destinataire": destinataire,
        "sujet": sujet,
        "contenu": rapport,
    })
    logger.info("Résultat envoi : %s", result)


if __name__ == "__main__":
    logger.info("Génération du rapport hebdomadaire...")
    rapport = generer_rapport()
    print(rapport)
    envoyer_rapport(rapport)
    logger.info("Rapport terminé.")
