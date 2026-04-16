---

# SAEARCH — Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques

> **Diplôme Universitaire Sorbonne Data Analytics**

> Année Universitaire 2025-2026

> Promotion SDA7

> Module **Generative AI**

---

## Vue d'ensemble

**SAEARCH** est un système d'assistants agentiques multi-compétences qui combine RAG scientifique, agents à outils, machine learning prédictif et aide à la décision humaine.
Il porte le nom **DooMax**. Il est accessible via l'UI Chainlit, un client MCP externe, ou un cron hebdomadaire qui envoie un rapport climatique chaque lundi.

Il est capable de :

- Interroger un **corpus de 10 rapports scientifiques** (GIEC AR6, Copernicus, EM-DAT, NOAA, JRC, WMO, EU Floods Directive) via un RAG hybride BM25 + Dense
- Consulter la **météo temps réel, l'historique des météos, ou les prévisions à 7 jours** (OpenMeteo)
- Prédire le risque catastrophe par pays et par type (sécheresse, inondation, tempête, canicule, feux) via un modèle Machine Learning entraîné sur EM-DAT 1900-2020
- Produire un **score de risque agrégé** 0-1 croisant 4 sources (météo / ML / corpus / historique) pondérées par horizon temporel (court_terme / standard / long_terme)
- Fournir une **aide à la décision humaine** via 4 profils : événementiel, assurance, autorité publique, tourisme — avec boucle **Human-in-the-Loop** (approuver / enrichir / rejeter + log MLflow)
- Envoyer des **emails d'alerte** (immédiats, bulk, ou programmés via APScheduler persistant)
- S'**intégrer dans tout client MCP** (Claude Desktop, Cursor, Claude Code) via un serveur FastMCP exposant 11 outils
- Implémentation STT PCM 16 de l'acquisition des inputs par la voix
- Upload multimodal : PDF/DOCX session-scoped (indexation FAISS in-memory), images via Claude vision, audio via Faster-Whisper

---

## Liens

| Ressource | URL |
|---|---|
| **Repo GitHub** (public) | https://github.com/diegomerchanm/catastrophes-climatiques-rag |
| **Image Docker Hub** (publique, pull direct) | https://hub.docker.com/r/xbizot/rag-catastrophes |
| **Déploiement cloud** (HF Spaces, live) | https://xbizot-saearch.hf.space |
| **Déploiement AWS App Runner** | Configuré, en attente d'activation du compte (délai AWS jusqu'à 24h) |

```bash
# Pull direct de l'image Docker
docker pull xbizot/rag-catastrophes:latest
```

Compte démo : `demo@saearch.ai` / `demo`

---

## Stack technique

| Composant | Technologie |
|---|---|
| **Langage** | Python 3.11 |
| **LLM** | Anthropic Claude — fallback quadruple : Opus 4.6 → Sonnet 4.5 → Haiku 4.5 → Ollama local |
| **Framework agent** | LangChain + LangGraph (boucle ReAct) |
| **RAG** | FAISS + BM25 hybride (EnsembleRetriever) |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (multilingue 50+ langues, local) |
| **Reranking** | CrossEncoder MS-MARCO, placement stratégique (Lost in the Middle) |
| **Bridge lexical** | FR→EN (GIEC↔IPCC, OMM↔WMO, 20 termes), diversité forcée max 3 chunks/PDF |
| **Retrieval** | MMR k=12, lambda_mult=0.7, avec citations page + source |
| **UI** | Chainlit (streaming, STT PCM16, upload PDF/DOCX/image, auth, traduction native 25 langues) |
| **API météo** | OpenMeteo (gratuit, sans clé) |
| **Recherche web** | Tavily (priorité) + DuckDuckGo (fallback) |
| **ML prédictif** | Gradient Boosting Quantile Regression (median) + classification multi-classe (scikit-learn + XGBoost) |
| **Protocole outils** | MCP (Model Context Protocol) via FastMCP — 11 outils exposés en français |
| **Tunnel HTTPS** | Cloudflare Tunnel (pour exposer le serveur MCP local à Claude Desktop) |
| **Monitoring** | MLflow (tokens, coûts, latences, prompts, HITL feedback, Model Registry) |
| **Scheduler** | APScheduler (SQLite persistant) + GitHub Actions cron |
| **Mémoire** | InMemoryChatMessageHistory, fenêtre glissante 20 tours |
| **Prompts** | Versionnés v1.0 + v2.0, tag MLflow `prompt_version` |
| **Conteneurisation** | Docker (python:3.11-slim + ffmpeg, `--no-cache` systématique) |
| **CI/CD** | GitHub Actions (2 jobs : ci_pipeline + cd_pipeline) → black + pylint + pytest (43 tests) → Docker Hub → HF Spaces |
| **Versioning** | Git / GitHub (workflow PR + protection main) |

---

## 13 outils de l'agent

| Catégorie | Outil | Source / API |
|---|---|---|
| **Météo** (×3) | `get_weather`, `get_historical_weather`, `get_forecast` | OpenMeteo (gratuit) |
| **RAG** (×2) | `search_corpus`, `list_corpus` | FAISS + BM25 hybride (10 PDFs GIEC) |
| **Web** | `web_search` | Tavily (priorité) + DuckDuckGo (fallback) |
| **Calcul** | `calculator` | Sandbox AST whitelist |
| **Email** (×3) | `send_email`, `send_bulk_email`, `schedule_email` | SMTP Gmail + APScheduler SQLite |
| **ML** (×2) | `predict_risk`, `predict_risk_by_type` | EM-DAT, horizon 2030 |
| **Scoring** | `calculer_score_risque` | 4 sources pondérées (météo/ML/corpus/historique) |

---

## Architecture

```
Question utilisateur
    ↓
Router (P3) : RAG / Agent / Chat
    ↓
┌────────────┬──────────────────────────────┬──────────┐
│   RAG      │  Agent ReAct (LangGraph)     │   Chat   │
│  hybride   │  13 outils chaînables        │  direct  │
│ BM25+Dense │                              │          │
└────────────┴──────────────────────────────┴──────────┘
                    ↓
        Outils disponibles :
        • search_corpus / list_corpus                (RAG)
        • get_weather / historical / forecast        (OpenMeteo)
        • web_search                                  (Tavily/DuckDuckGo)
        • calculator                                  (eval sandboxé)
        • predict_risk / predict_risk_by_type         (ML NB10)
        • calculer_score_risque                       (scoring 4 sources)
        • send_email / send_bulk_email                (SMTP Gmail)
        • schedule_email                              (APScheduler)
                    ↓
          Donut SVG live + réponse
          + sources citées [Source: X, Page: Y]
          + tokens / coût / durée
          + Human-in-the-Loop (mode décisionnel)
                    ↓
             MLflow tracking
```

### Serveur MCP

**11 des 13 outils** sont exposés via `mcp_server.py` (FastMCP). Les 2 outils non exposés (`send_bulk_email`, `schedule_email`) restent internes à Chainlit par principe de moindre privilège (anti-spam, anti-persistence malveillante).

Le serveur peut être consommé par :
- Claude Desktop (via Cloudflare Tunnel → URL HTTPS `/mcp`)
- Claude Code (stdio local)
- Cursor, Continue, tout client MCP standard

### Fallback LLM quadruple

```
Opus 4.6 (principal)
    ↓ échec / rate-limit 429
Sonnet 4.5 (fallback Anthropic)
    ↓ échec
Haiku 4.5 (fallback économique)
    ↓ échec / panne API
Ollama Mistral (fallback local, hors ligne, gratuit)
```

Résilience contre rate-limit 429, panne API, quota épuisé.

---

## ML prédictif

- **Modèle** : Gradient Boosting Quantile Regression (perte médiane), sélection MAE_test (robustesse outliers, ex : canicule France 2003)
- **Clipping** : 1.5× max historique décadal (plafond prédictions)
- **Grille comparative** : 8 régresseurs + 8 classifieurs dans MLflow (16 nested runs par entraînement)
- **Dataset** : EM-DAT décadal (14 625 lignes, 225 pays × 5 types climatiques)
- **Model Registry** : `NB10_regression v1+`, `NB10_classification v1+`
- **Artefacts** : 7 joblib exportés, consommés par l'agent via `predict_risk` / `predict_risk_by_type`

---

## LLMOps / Monitoring

- **Observabilité par requête** : 1 run MLflow par question utilisateur (prompt_version, tokens in/out, coût USD estimé, durée_s, outils appelés, longueur réponse, tags git_commit + env)
- **Token tracking temps réel** : affichage live dans Chainlit — TokenCounter custom avec pricing Anthropic (Opus/Sonnet/Haiku)
- **Prompt versioning** : dict PROMPTS v1.0 et v2.0, tag MLflow `params.prompt_version` pour filtrer et comparer
- **Cost monitoring** : estimation USD par requête, agrégation MLflow (profiling des questions coûteuses)
- **Latency tracking** : metric `duree_s` par run, détection régressions
- **Fallback LLM quadruple** : Opus → Sonnet → Haiku → Ollama local (résilience cloud)
- **Tracing outils** : détection `tool_calls` via events LangGraph, log MLflow nb_outils / outils appelés
- **Dashboard MLflow** : UI locale SQLite, comparaison runs par tag

---

## Corpus

**10 rapports scientifiques PDF**, placés dans `data/raw/` (non versionnés).

| Document | Source |
|---|---|
| GIEC AR6 — Synthesis Report Summary for Policymakers | IPCC |
| GIEC AR6 — Working Group II (impacts, adaptation) | IPCC |
| GIEC AR6 — Working Group III (atténuation) | IPCC |
| European State of Climate 2023 | Copernicus |
| Global Assessment Report on Disaster Risk Reduction 2022 | UNDRR |
| Natural Disasters Report 2023 | EM-DAT |
| State of Global Water Resources 2024 | WMO |
| Atlantic Hurricane Season 2023 | NOAA |
| Forest Fires in Europe 2024 | JRC |
| Floods Directive Report (CELEX) | EU |

Télécharger l'archive : [Google Drive](https://drive.google.com/drive/folders/1g7JAdrbp1CoioPm7MR4cjQnT_A0hJjYo).

---

## Installation locale

### Prérequis

- **Python 3.11**
- Clés API : `ANTHROPIC_API_KEY` (obligatoire), `TAVILY_API_KEY` (optionnel), `EMAIL_APP_PASSWORD` (optionnel)
- Optionnel : Docker Desktop, Claude Desktop, `cloudflared` pour la démo MCP

### Étapes

```bash
git clone https://github.com/diegomerchanm/catastrophes-climatiques-rag.git
cd catastrophes-climatiques-rag
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Variables d'environnement** (`.env` à la racine) :
```bash
cp .env.example .env
# Renseigner au minimum :
# ANTHROPIC_API_KEY=...                                  (obligatoire)
# CHAINLIT_AUTH_SECRET=...                               (obligatoire, secrets.token_urlsafe(32))
# TAVILY_API_KEY=...                                     (optionnel, recherche web)
# EMAIL_ADDRESS / EMAIL_APP_PASSWORD / EMAIL_RECIPIENT   (optionnels, alertes email)
# MLFLOW_TRACKING_URI=                                   (vide = SQLite local fallback)
```

**Générer le vector store FAISS + modèles ML :**
```bash
.\venv\Scripts\python.exe -m src.rag.embeddings   # génère FAISS 1889 chunks
.\venv\Scripts\python.exe scripts/train_nb10.py   # génère joblibs ML
```

**Lancer l'app :**
```bash
.\venv\Scripts\python.exe -m chainlit run app.py -w --port 8080
```

Ouvrir http://localhost:8080, se connecter avec `demo@saearch.ai / demo`.

---

## Utilisation avancée

### Serveur MCP pour Claude Desktop

```bash
python mcp_server.py
# ou via le proxy pour Streamable HTTP (Claude Desktop 2025+)
mcp-proxy --sse-port 8765 -- python mcp_server.py
```

Exposer en HTTPS public :
```bash
cloudflared tunnel --url http://localhost:63119
# Retourne : https://xxx-xxx.trycloudflare.com
```

Dans Claude Desktop → Paramètres → Connecteurs → Ajouter un connecteur personnalisé → URL : `https://xxx-xxx.trycloudflare.com/mcp`

Les 11 outils SAEARCH apparaissent et sont appelables par Claude.

### Rapport hebdomadaire automatique

Cron GitHub Actions tous les lundis 8h UTC : synthèse des alertes climatiques de la semaine + envoi email (`scheduled_report.py` + `.github/workflows/weekly-report.yml`).

### Dashboard MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8080
```

Expériences suivies : `rag-catastrophes-climatiques`, `nb6-comparatifs-rag`, `NB10_train_script`.

---

## Structure du projet

```
catastrophes-climatiques-rag/
├── app.py                          # UI Chainlit (auth, routing, HITL, decision mode)
├── mcp_server.py                   # Serveur MCP FastMCP (11 outils exposés)
├── scheduled_report.py             # Rapport hebdomadaire (cron lundi 8h UTC)
├── Dockerfile                      # Build reproductible --no-cache (python:3.11-slim + ffmpeg)
├── README.md
├── CLAUDE.md                       # Conventions de code + retex loan-default
├── .env.example
├── requirements.txt
│
├── data/
│   ├── raw/                        # PDFs corpus (non versionnés)
│   └── decadal-*.csv               # Données OWID/EM-DAT décadal (14 625 lignes)
│
├── faiss_store/                    # Vector store FAISS persisté (versionné, 5 MB)
│
├── src/
│   ├── config.py                   # AGENT_CONFIGS (spécialisation LLM), TokenCounter, pricing
│   ├── rag/
│   │   ├── loader.py               # Chargement PDFs, chunking 1500/150
│   │   ├── embeddings.py           # Création FAISS (paraphrase-multilingual)
│   │   ├── retriever.py            # MMR k=12, lambda_mult=0.7
│   │   └── hybrid_retriever.py     # EnsembleRetriever BM25+Dense + reranking CrossEncoder
│   ├── agents/
│   │   ├── tools.py                # 13 outils @tool LangGraph
│   │   └── agent.py                # Agent ReAct LangGraph, mémoire, MLflow, fallback
│   ├── memory/
│   │   └── memory.py               # InMemoryChatMessageHistory, fenêtre 20 tours
│   ├── router/
│   │   └── router.py               # Routing conditionnel RAG / Agent / Chat
│   ├── prompts/
│   │   └── agent_prompts.py        # v1.0 + v2.0 versionnés, A/B testing
│   └── ui/
│       ├── donut_chart.py          # SVG donut inline (10 catégories)
│       └── data_layer.py           # SQLite threads Chainlit
│
├── notebooks/                      # 13 notebooks (RAG, agents, MLflow, MCP, ML, CI/CD)
├── scripts/
│   ├── train_nb10.py               # Pipeline d'entraînement reproductible (CI/CD)
│   └── teaser_vendredi.py          # Teaser soutenance (cron one-shot)
├── tests/                          # pytest (13 outils, calculateur, géocodage, prompts)
├── outputs/                        # Artefacts ML : 7 joblib + plots + prédictions 2030
├── .github/workflows/
│   ├── github-docker-cicd.yaml     # CI : black + pylint + pytest → Docker Hub (:latest + :date)
│   ├── azure.yml                   # CD Azure Container Apps (plan B, workflow prêt)
│   ├── weekly-report.yml           # Cron lundi 8h UTC
│   └── teaser-vendredi.yml         # One-shot 17/04 8h UTC
├── docs/                           # Audit technique, architecture Mermaid, matrice TSV
├── mlflow.db                       # Backend SQLite MLflow
└── scheduler_jobs.db               # Persistance APScheduler
```

---

## Points forts du projet

### Fondations équipe (héritages respectés et étendus)

- **Pipeline RAG hybride robuste** sur 10 PDFs GIEC (travail P1) — BM25 + FAISS + CrossEncoder reranking, citations page + source, chunking optimisé 1500/150
- **Architecture agent LangGraph ReAct multi-outils** (travail P2) — 3 outils initiaux étendus à 13, pattern Reason → Act → Observe → Answer conservé
- **Interface Chainlit streaming + routing conditionnel 3 voies** (travail P3) — UI traduite nativement 25 langues, router RAG / Agent / Chat avec keywords + fallback LLM classifier

### Innovations architecturales (développées sur les fondations équipe)

- **RAG cross-lingual** : embedding multilingue 50+ langues + bridge lexical FR→EN (20 termes), diversité forcée max 3 chunks/PDF
- **Orchestration agentique 4 outils pour scoring multi-sources** : `calculer_score_risque` combine météo + ML + corpus GIEC + historique
- **ML prédictif robuste aux outliers** : Quantile Regression + clipping, fix biais canicule France 2003
- **Scheduler d'emails différés persistant** : `schedule_email` (13e outil), APScheduler + SQLite
- **Fallback LLM quadruple** pour résilience cloud : Opus → Sonnet → Haiku → Ollama
- **MLflow Model Registry + observabilité LLMOps complète** : 1 run par requête Chainlit, unification backend tracking ML + LLM
- **Upload PDF/DOCX/image session-scoped** : anti-pollution corpus, isolation par session
- **Exposition MCP universelle** via FastMCP : 11 outils réutilisés sans duplication de code, démo live Claude Desktop

---

## Équipe

| Phase | Responsabilité |
|---|---|
| **P1** | Corpus & RAG — `loader.py`, `embeddings.py`, `retriever.py`, hybride BM25+Dense |
| **P2** | Agents & Tools — `tools.py`, `agent.py`, 13 outils, ReAct LangGraph |
| **P3** | Router & UI — `router.py`, `app.py` streaming, badges, traduction 25 langues |
| **P4** | Agentic RAG avancé, Mémoire, MLOps/LLMOps, Notebooks NB4-NB10, MCP, HITL, UI décisionnelle, CI/CD, déploiement |

*Les noms des membres de l'équipe ne sont pas publiés ici — ils figurent dans le dossier de rendu académique.*

---

## Notebooks

| Notebook | Contenu | Phase |
|---|---|---|
| 01-03 | Exploration corpus, tests RAG, tests agents | P1/P2/P3 |
| NB4 | Mémoire conversationnelle + multilingue FR/EN/ES | P4 |
| NB5 | Analyse de risque prédictive (passé/présent/futur + multi-villes) | P4 |
| NB6 | Comparatifs MLflow (pondérations, températures, A/B prompts) | P4 |
| NB7 | Serveur MCP + tests outils + validation Claude Desktop live | P4 |
| NB8 | LLMOps / Monitoring (tokens, coûts, spécialisation, fallback) | P4 |
| NB9 | CI/CD, Docker, déploiement HF Spaces + Azure + Cloudflare | P4 |
| NB10 | ML prédictif multi-type (Quantile Regression + classification) | P4 |

Chaque notebook suit le format : Configuration → Analyse → Résultats → Conclusions (quality gate + hypothèse + limites + axes d'amélioration).

---

## Licences et ressources

- **Code** : MIT (sauf mention contraire)
- **Corpus PDF** : propriétés des organisations émettrices (GIEC, Copernicus, EM-DAT, NOAA, JRC, WMO, EU). Usage académique dans le cadre du DU.
- **Claude (Anthropic)** : API payante, clé à fournir dans `.env`
- **Tavily** : gratuit jusqu'à 1 000 requêtes/mois
- **OpenMeteo** : gratuit sans clé

---

## Contact

Projet réalisé dans le cadre du Diplôme Universitaire **Sorbonne Data Analytics** (SDA7), module Generative AI, 2025-2026.

Repository : https://github.com/diegomerchanm/catastrophes-climatiques-rag
