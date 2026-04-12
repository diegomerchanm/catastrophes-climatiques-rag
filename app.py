"""
Point d'entrée Chainlit — Interface conversationnelle.
Branche l'agent agentic RAG multi-compétences avec la mémoire conversationnelle.
LLMOps : monitoring tokens, estimation coût, logging.
"""

import logging

import chainlit as cl
from dotenv import load_dotenv

from src.agents.agent import get_prompt_version, get_token_summary, run_agent
from src.memory.memory import add_exchange, get_session_history

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start():
    """Initialise une nouvelle session de chat."""
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    logger.info("Nouvelle session Chainlit : %s", session_id)

    await cl.Message(
        content=(
            "**Assistant Catastrophes Climatiques**\n\n"
            "Je peux :\n"
            "- Répondre à vos questions sur le corpus scientifique "
            "(GIEC, Copernicus, EM-DAT...)\n"
            "- Consulter la météo actuelle, historique ou les prévisions\n"
            "- Effectuer des calculs\n"
            "- Rechercher des actualités sur le web\n"
            "- Croiser toutes ces sources pour une analyse de risque\n\n"
            "Comment puis-je vous aider ?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Traite un message utilisateur."""
    session_id = cl.user_session.get("session_id")
    question = message.content

    # Récupérer l'historique de la session
    history = get_session_history(session_id)
    chat_history = history.messages

    # Indicateur de chargement
    msg = cl.Message(content="")
    await msg.send()

    # Exécuter l'agent avec l'historique
    answer = run_agent(question, chat_history=chat_history)

    # Mettre à jour la mémoire (avec fenêtre glissante)
    add_exchange(session_id, question, answer)

    # Afficher la réponse
    msg.content = answer
    await msg.update()

    # Afficher le monitoring LLMOps
    tokens = get_token_summary()
    if tokens["total_tokens"] > 0:
        cost = tokens.get("estimated_cost_usd", 0)
        await cl.Message(
            content=(
                f"*Tokens : {tokens['total_tokens']} "
                f"(in: {tokens['total_input_tokens']}, "
                f"out: {tokens['total_output_tokens']}) — "
                f"Coût estimé : ${cost:.4f} — "
                f"Prompt {get_prompt_version()}*"
            ),
            author="system",
        ).send()
