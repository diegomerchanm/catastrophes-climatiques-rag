# CLAUDE.md — catastrophes-climatiques-rag

> **Version 1.1** — Refonte du 17/04/2026, alignée sur le code. Version précédente conservée dans [`docs/archive/CLAUDE_v1.0.md`](docs/archive/CLAUDE_v1.0.md).

> Fichier de contexte chargé automatiquement par Claude Code et autres assistants IA travaillant sur ce repo. Pour la doc utilisateur, voir [`README.md`](README.md). Pour l'architecture détaillée, voir [`docs/architecture.md`](docs/architecture.md).

## Projet

**SAEARCH** (*Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques*), nom UI **DooMax** : système agentique multi-compétences combinant RAG scientifique, agents à outils, machine learning prédictif et aide à la décision humaine. Cible : analyse et anticipation des risques climatiques pour 4 profils utilisateurs (événementiel, assurance, autorité publique, tourisme).

## Dataset / Corpus

- **Dossier** : `data/raw/` (ignoré par git, ne jamais y commiter de fichiers)
- **10 rapports PDF** : GIEC AR6 (Synthesis + WG II + WG III), Copernicus 2023, UNDRR 2022, EM-DAT 2023, WMO 2024, NOAA Atlantic 2023, JRC Forest Fires 2024, EU Floods Directive 2021
- **Vector store** : `faiss_store/` (1 889 chunks, 5 MB, **versionné** dans le repo pour cold start instantané — voir commit `37cb00f`)

## Stack technique (vérité = code)

| Brique | Technologie | Source de vérité |
|---|---|---|
| **Langage** | Python 3.11 | `Dockerfile`, `requirements.txt` |
| **LLM principal** | Anthropic Claude — **Sonnet 4** (orchestrateur), **Opus 4** (analyste), **Haiku 4.5** (météo/web/chat/classifier) | `src/config.py:11-13`, `AGENT_CONFIGS` |
| **Fallback LLM** | 3 niveaux : Sonnet 4 → Haiku 4.5 → Ollama Mistral local | `src/agents/agent.py:64-78` |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (50+ langues, local) | `src/config.py:75` |
| **Vector store** | FAISS | `src/rag/embeddings.py` |
| **Framework agent** | LangChain + LangGraph (boucle ReAct) | `src/agents/agent.py` |
| **Retrieval** | Hybride BM25 + Dense via `EnsembleRetriever` 50/50 + reranking CrossEncoder MS-MARCO + placement stratégique anti-LITM | `src/rag/hybrid_retriever.py` |
| **Paramètres MMR** | k=12, fetch_k=40, lambda_mult=0.7 | `src/config.py:80-83` |
| **UI** | Chainlit (streaming, auth, donut SVG, multi-conversations, traduction 25 langues) | `app.py`, `src/ui/` |
| **Multimodal** | Upload PDF/DOCX session-scoped (FAISS in-memory) + image Claude vision (base64) + audio Faster-Whisper local | `app.py` |
| **STT** | Faster-Whisper modèle `small`, CPU, int8, **autodétection de langue**, capture PCM16 → WAV | `app.py` |
| **TTS** | Web Speech API navigateur (bouton flottant 🔊 vert, bas-gauche), priorité voix Google FR puis FR-FR | `public/tts.js`, `public/geoloc.js` |
| **Géolocalisation** | `navigator.geolocation` (GPS natif) + `ip-api.com` en fallback, stockée `sessionStorage` | `public/geoloc.js` |
| **API météo** | OpenMeteo (gratuit, sans clé, 3 outils : actuel / historique / forecast) | `src/agents/tools.py` |
| **Recherche web** | Tavily (prioritaire) + DuckDuckGo (fallback) | `src/agents/tools.py` |
| **Calculator** | Sandbox AST whitelist (anti-injection) | `src/agents/tools.py` |
| **Email** | SMTP Gmail + APScheduler SQLite (`schedule_email` + `cancel_scheduled_emails`) | `src/agents/tools.py` |
| **ML prédictif** | Gradient Boosting Quantile Regression (median) + classification multi-classe + clipping 1.5× max historique | `scripts/train_nb10.py`, `outputs/NB10_*.joblib` |
| **MLOps** | MLflow SQLite (`mlflow.db`) + Model Registry (`NB10_regression v1+`) + grille comparative 16 modèles par entraînement | `scripts/train_nb10.py` |
| **LLMOps** | 1 run MLflow par requête utilisateur (prompt_version, tokens in/out, coût USD, durée, outils) | `src/agents/agent.py` |
| **Prompts** | Versionnés v1.0 + v2.0, tag MLflow `prompt_version` pour A/B testing | `src/prompts/agent_prompts.py` |
| **Mémoire** | `InMemoryChatMessageHistory`, fenêtre glissante 20 tours par session | `src/memory/memory.py` |
| **HITL** | Mode décisionnel 4 profils, validation humaine Approuver / Enrichir / Rejeter, log MLflow | `app.py` |
| **Guide de lecture** | Légende auto-explicative en fin de chaque réponse décisionnelle (DECISION / HORIZON / SCORE / RISQUE / CONFIANCE) | `app.py:465+` |
| **Style adressage** | Vouvoiement par défaut (ton professionnel cohérent avec public décideur) | prompts |
| **Exposition MCP** | FastMCP, 11 des 13 outils exposés en français pour Claude Desktop / Cursor / Continue | `mcp_server.py` |
| **Conteneurisation** | Docker `python:3.11-slim` + ffmpeg, port 8000, `--no-cache` systématique en CI | `Dockerfile`, `.github/workflows/github-docker-cicd.yaml` |
| **CI/CD** | GitHub Actions 2 jobs : `ci_pipeline` (black + pylint + pytest 43 tests) + `cd_pipeline` (Docker build → Docker Hub → trigger HF Spaces) | `.github/workflows/` |
| **Déploiement live** | Hugging Face Spaces (`xbizot-saearch.hf.space`, gratuit) | `huggingface.co` |
| **Registry alt** | Docker Hub (`xbizot/rag-catastrophes:latest`) + AWS ECR (artefact, pas de runtime payé) | — |
| **Déploiement plan B** | Azure Container Apps (`workflow_dispatch` only, pas activé) | `.github/workflows/azure.yml` |
| **Crons** | `weekly-report.yml` (lundi 8h UTC) + `teaser-vendredi.yml` (one-shot soutenance) | `.github/workflows/` |

## Structure du projet

```
catastrophes-climatiques-rag/
├── app.py                          # UI Chainlit (auth, routing, HITL, multimodal, STT)
├── mcp_server.py                   # Serveur MCP FastMCP (11 outils exposés)
├── scheduled_report.py             # Rapport hebdomadaire (cron lundi 8h UTC)
├── Dockerfile                      # python:3.11-slim + ffmpeg, port 8000
├── README.md                       # Vue d'ensemble pour utilisateurs
├── CLAUDE.md                       # Ce fichier — contexte pour assistants IA
├── .env.example
├── requirements.txt
│
├── data/
│   ├── raw/                        # 10 PDFs corpus (non versionnés)
│   └── decadal-*.csv               # Données EM-DAT décadal
│
├── faiss_store/                    # Vector store FAISS (versionné, 5 MB)
│
├── src/
│   ├── config.py                   # AGENT_CONFIGS, modèles, hyperparams, TokenCounter
│   ├── rag/
│   │   ├── loader.py               # Chargement PDFs, chunking 1500/150
│   │   ├── embeddings.py           # Création FAISS (paraphrase-multilingual)
│   │   ├── retriever.py            # MMR k=12, fetch_k=40, lambda_mult=0.7
│   │   └── hybrid_retriever.py     # EnsembleRetriever BM25+Dense + reranking
│   ├── agents/
│   │   ├── tools.py                # 13 outils @tool LangGraph + complément cancel_scheduled
│   │   └── agent.py                # Agent ReAct LangGraph + fallback 3 niveaux + MLflow
│   ├── memory/memory.py            # Fenêtre glissante 20 tours
│   ├── router/router.py            # Routing conditionnel RAG / Agent / Chat
│   ├── prompts/agent_prompts.py    # v1.0 + v2.0 versionnés
│   └── ui/
│       ├── donut_chart.py          # SVG donut inline (10 catégories)
│       └── data_layer.py           # SQLite threads Chainlit
│
├── notebooks/                      # 12 notebooks (NB1-3 P1/P2/P3, NB4-NB10 P4) + template
├── scripts/
│   ├── train_nb10.py               # Pipeline ML reproductible (CI/CD)
│   └── teaser_vendredi.py          # Teaser soutenance (cron one-shot)
├── tests/                          # 43 tests pytest (outils, calculator, géocodage, prompts)
├── outputs/                        # 7 joblibs ML + plots + prédictions 2030
├── docs/                           # Architecture, ADR, innovations, slides
│   └── archive/                    # Versions précédentes des docs (traçabilité)
├── .github/workflows/              # CI Docker + Azure (plan B) + 2 crons
├── public/                         # Assets statiques Chainlit (background, CSS, JS)
├── mlflow.db                       # Backend SQLite MLflow
└── scheduler_jobs.db               # Persistance APScheduler
```

## Construction par phases

Le projet a été construit en quatre phases. Les innovations P4 **étendent** les briques P1/P2/P3, sans les remplacer.

| Phase | Périmètre | Apports principaux |
|---|---|---|
| **P1** | Corpus & RAG | Pipeline ingestion PDF, chunking, embeddings FAISS, retriever MMR, métadonnées source/page |
| **P2** | Agents & Tools | Agent LangGraph ReAct, premiers outils (météo, web, calcul) |
| **P3** | Router & UI | Interface Chainlit streaming, routing conditionnel 3 voies, traduction UI |
| **P4** | Agentic RAG avancé | Mémoire, MLOps, observabilité, MCP, ML prédictif, multimodal, CI/CD, déploiement |

## Conventions de code

### Langue
- **Variables, fonctions, commentaires, docstrings** : en français
- **Noms de modules et fichiers** : en français ou anglais technique (`snake_case`)

### Nommage
- **Fonctions et variables** : `snake_case` (ex : `creer_retriever`, `chat_history`)
- **Constantes** : `UPPER_CASE` (ex : `CHUNK_SIZE`, `MODEL_SONNET`)
- **Classes** : `PascalCase` (ex : `TokenCounter`, `AgentState`)
- **Fichiers** : `snake_case.py` (ex : `hybrid_retriever.py`)

### Docstrings
- **Format Google** pour les fonctions publiques (description, Args, Returns)
- En français

### Imports
- Groupés par catégorie, séparés par une ligne vide :
  1. Stdlib (`os`, `logging`, `math`)
  2. Third-party (`langchain`, `requests`, `faiss`)
  3. Locaux (`from src.config import ...`)

### Type hints
- Obligatoires sur les signatures de fonctions publiques
- Optionnels sur les variables locales

### Logging
- Utiliser `logging` (pas `print()`)
- Un logger par module : `logger = logging.getLogger(__name__)`

### Emojis
- Pas d'emojis dans le code source ni dans les retours d'outils
- Emojis autorisés uniquement dans l'interface Chainlit (`app.py`)

### Formattage et qualité
- **Formatteur** : `black` (longueur max 88 caractères) — appliqué dès la frappe, pas en post-traitement
- **Linter** : `pylint --disable=R,C` (exit-zero pour les warnings)
- **Tests** : `pytest -vv` (43 tests doivent passer)
- Pas d'imports inutilisés
- Pas de code mort (variables / fonctions jamais appelées)

### Git
- **Commits** : Conventional Commits
  - `feat:` nouvelle fonctionnalité
  - `fix:` correction de bug
  - `chore:` tâche technique (dépendances, config)
  - `docs:` documentation
  - `style:` formatage (black, etc.)
  - `ci:` workflow CI/CD
- **Ne jamais committer sur `main` directement** (PR + protection)
- **Ne jamais utiliser `git add -A` ou `git add .`** : lister explicitement les fichiers à stager (incident `17/04` : commit accidentel de docs privés sur repo public)
- **Ne jamais committer de fichiers de données** (PDFs, CSV, modèles, `.env`)

## Règles de sécurité

- Ne **jamais** uploader les PDFs du corpus dans le repo (`data/raw/` est gitignoré)
- Ne **jamais** hardcoder des credentials (clés API, tokens, mots de passe)
- Utiliser des variables d'environnement via `.env` (listé dans `.gitignore`)
- Le `.env.example` sert de modèle (sans valeurs réelles)
- Le calculator de l'agent utilise une **sandbox AST whitelist** — ne jamais le remplacer par `eval()`
- Les uploads utilisateur sont **session-scoped** — ne jamais les merger dans le corpus officiel

## Retex projet précédent

- **Render** ne rebuild pas l'image complète (cache de layers incrémental) → bugs masqués, solution non reproductible
- Sur ce projet : `docker build --no-cache` **systématique** en CI/CD
- Déploiement principal sur **HF Spaces** (gratuit) au lieu de Render
- AWS App Runner et ECS Fargate ont été évalués mais écartés (pas free tier, ~30-50 $/mois pour une démo qu'HF couvre gratuitement)
- L'image AWS ECR reste en place comme **artefact** (preuve de maîtrise de la chaîne ECR), sans service de runtime attaché

## Pour aller plus loin

- [`README.md`](README.md) — vue d'ensemble utilisateur, installation, usage
- [`docs/architecture.md`](docs/architecture.md) — architecture détaillée
- [`docs/decisions.md`](docs/decisions.md) — 22 ADR (le *pourquoi* derrière les choix)
- [`docs/innovations_techniques.tsv`](docs/innovations_techniques.tsv) — innovations notables
- [`docs/architecture_pour_slides.md`](docs/architecture_pour_slides.md) — version condensée pour la soutenance
