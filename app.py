"""
Point d'entrée Chainlit — Interface conversationnelle.
Combine le router conditionnel (P3 — Jayson) avec l'agent agentic RAG (P4 — Xia).
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

from src.agents.agent import get_agent, get_prompt_version, get_token_summary
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
    "get_weather": "Agent",
    "get_historical_weather": "Agent",
    "get_forecast": "Agent",
    "web_search": "Agent",
    "calculator": "Agent",
    "send_email": "Agent",
    "predict_risk": "ML",
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
    """Authentification simple par mot de passe."""
    # Utilisateurs autorises (en prod, utiliser une base de donnees)
    users = {
        "xiabizot@gmail.com": {"password": "saearch", "name": "Xia", "role": "admin"},
        "kamilakare@gmail.com": {"password": "prof2026", "name": "Kamila", "role": "admin"},
        "diegomerchanm@gmail.com": {"password": "saearch", "name": "Diego", "role": "user"},
        "jaysonphannguyenpro@gmail.com": {"password": "saearch", "name": "Jayson", "role": "user"},
        "camille.koenig@gmail.com": {"password": "saearch", "name": "Camille", "role": "user"},
        "demo@saearch.ai": {"password": "demo", "name": "Demo", "role": "user"},
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
    logger.info(
        "Nouvelle session Chainlit : %s (user: %s)", session_id, user_name
    )

    # Donut d'accueil personnalise
    greeting = f"Bonjour <b>{user_name}</b>" if user else "Bonjour"
    donut_accueil = generer_message_avec_donut(
        answer=(
            f"{greeting}, je suis <b>DooMax</b>, ton IA dans <b>SAEARCH</b>, "
            "le Systeme Agentique d'Evaluation et d'Anticipation "
            "des Risques Climatiques et Hydrologiques.<br><br>"
            "Comment puis-je t'aider ?"
        ),
        outils_appeles=[],
        route="chat",
    )
    await cl.Message(content=donut_accueil).send()


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
        # Etape 1 : classifier la question via le router de Jayson (P3)
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
            if any(kw in question.lower() for kw in ["email", "mail", "envoie", "envoyer", "rappel"]):
                if user_email:
                    ctx_parts.append(
                        f"l'email du profil connecte est {user_email}"
                    )
                ctx_parts.append(
                    "Repertoire contacts : Kamila=kamilakare@gmail.com, "
                    "Xia=xiabizot@gmail.com, Camille=camille.koenig@gmail.com, "
                    "Diego=diegomerchanm@gmail.com, Jayson=jaysonphannguyenpro@gmail.com. "
                    "Quand l'utilisateur dit 'envoie-moi', utilise l'email correspondant "
                    "au prenom qu'il a donne dans la conversation"
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
        msg.content = generer_message_avec_donut(
            answer=answer,
            outils_appeles=outils_bruts,
            route=route,
            sources=sources,
            tokens_info=tokens_info,
        )
        await msg.update()

    except Exception as exc:
        logger.error("CRASH on_message : %s", exc, exc_info=True)
        await cl.Message(content=f"Erreur : {exc}").send()


# ── Route RAG : streaming via chain.astream (pattern P3) ─────────────────


async def _handle_rag(msg, question: str) -> tuple[str, list]:
    """Route RAG : recherche dans le corpus puis streaming de la reponse."""
    try:
        question_lower = question.lower()

        # Detection : l'utilisateur veut l'inventaire complet du corpus
        corpus_keywords = ["liste", "combien de doc", "inventaire", "tous les doc",
                           "resume le corpus", "resumer le corpus", "quels doc"]
        is_corpus_listing = any(kw in question_lower for kw in corpus_keywords)

        from src.rag.embeddings import charger_vector_store
        from src.rag.retriever import interroger_rag

        vector_store = charger_vector_store()
        if vector_store is None:
            error_msg = "Vector store non disponible. Lancez d'abord embeddings.py."
            msg.content += error_msg
            await msg.update()
            return error_msg, []

        # Si inventaire demande, k=50 pour couvrir tous les docs
        if is_corpus_listing:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 50, "fetch_k": 100},
            )
        else:
            from src.rag.retriever import creer_retriever
            retriever = creer_retriever(vector_store)

        resultat = interroger_rag(retriever, question)
        contexte = resultat["contexte"]
        documents = resultat["documents"]

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
        # Fallback : utiliser rag_node de Jayson (P3) sans streaming
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
                "Voici l'integralite des {nb_docs} documents de ton corpus. "
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

    try:
        async for event in agent.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]

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

            if kind == "on_chain_end" and "messages" in event.get("data", {}).get(
                "output", {}
            ):
                all_messages = event["data"]["output"]["messages"]

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
            # Fallback 2 : router complet de Jayson (P3)
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

    # Detecter les outils appeles et les sources
    agent_outils, sources, outils_bruts = _detecter_outils_appeles(all_messages)
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
        # Fallback : utiliser chat_node de Jayson (P3) sans streaming
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

        vector_store = charger_vector_store()
        if vector_store is not None:
            vector_store.add_documents(chunks)
            vector_store.save_local(FAISS_STORE_PATH)
            logger.info(
                "PDF %s integre : %d pages, %d chunks",
                element.name,
                len(pages),
                len(chunks),
            )
            await cl.Message(
                content=(
                    f"**Document integre** : {element.name}\n"
                    f"- {len(pages)} pages lues\n"
                    f"- {len(chunks)} passages indexes\n"
                    f"- Disponible immediatement pour les recherches"
                )
            ).send()
        else:
            await cl.Message(
                content="Vector store non initialise. Lancez d'abord embeddings.py."
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
            import struct
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
            segments, _ = model.transcribe(tmp_path, language="fr")
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
