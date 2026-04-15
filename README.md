# SAEARCH — Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques

> Projet d'équipe — **Diplôme Universitaire Sorbonne Data Analytics**, promotion SDA7 (2025-2026).
> Module **Generative AI**, soutenance le 17 avril 2026.

---

## Vue d'ensemble

**SAEARCH** est un assistant agentique multi-compétences qui combine RAG scientifique, agents à outils, machine learning prédictif et aide à la décision humaine. Il est capable de :

- Interroger un **corpus de 10 rapports scientifiques** (GIEC AR6, Copernicus, EM-DAT, NOAA, JRC, WMO, EU Floods Directive) via un RAG hybride BM25 + Dense
- Consulter la **météo temps réel, historique et prévisions 7 jours** (OpenMeteo, gratuit)
- Prédire le risque catastrophe par pays et par type (sécheresse, inondation, tempête, canicule, feux) via un modèle ML entraîné sur EM-DAT 1900-2020 (**`NB10`**)
- Produire un **score de risque agrégé** 0-1 croisant 4 sources (météo / ML / corpus / historique) pondérées par horizon temporel (court_terme / standard / long_terme)
- Fournir une **aide à la décision humaine** via 4 profils : événementiel, assurance, autorité publique, tourisme — avec boucle **Human-in-the-Loop** (approuver / enrichir / rejeter + log MLflow)
- Envoyer des **emails d'alerte** (immédiats, bulk, ou programmés via APScheduler persistant)
- S'**intégrer dans tout client MCP** (Claude Desktop, Cursor, Claude Code) via un serveur FastMCP exposant 11 outils

L'assistant porte le nom interne **DooMax**. Il est accessible via l'UI Chainlit, un client MCP externe, ou un cron hebdomadaire qui envoie un rapport climatique chaque lundi.

---

## Démo en ligne

- **Hugging Face Spaces** : https://xbizot-saearch.hf.space
- Compte démo : `demo@saearch.ai` / `demo`

---

## Stack technique

| Composant | Technologie |
|---|---|
| **LLM** | Anthropic Claude (Haiku 4.5, Sonnet 4.5, Opus 4.6) avec spécialisation par agent |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (multilingue FR/EN/ES, local) |
| **Vector store** | FAISS |
| **Recherche** | Hybride BM25 + Dense (EnsembleRetriever LangChain) |
| **Framework** | LangChain + LangGraph (agent ReAct) |
| **UI** | Chainlit (streaming, STT, upload PDF/DOCX, auth) |
| **API météo** | OpenMeteo (gratuit, sans clé) |
| **Recherche web** | Tavily (priorité) + DuckDuckGo (fallback) |
| **ML prédictif** | Gradient Boosting Quantile Regression + classification multi-classe (scikit-learn + XGBoost) |
| **Protocole outils** | MCP (Model Context Protocol) via FastMCP |
| **Tunnel HTTPS** | Cloudflare Tunnel (pour exposer le serveur MCP local à Claude Desktop) |
| **Monitoring** | MLflow (tokens, coûts, latences, prompts, HITL feedback) |
| **Scheduler** | APScheduler (SQLite persistant) + GitHub Actions cron |
| **Conteneurisation** | Docker (`--no-cache` systématique) |
| **CI/CD** | GitHub Actions → Docker Hub → HF Spaces (plan A) + Azure Container Apps (plan B, workflow prêt) |

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
- Claude Desktop (via Cloudflare Tunnel → URL HTTPS)
- Claude Code (stdio local)
- Cursor, Continue, tout client MCP standard

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

## Installation

### Prérequis

- **Python 3.11** (figé par le Dockerfile et GitHub Actions)
- Clés API : `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `EMAIL_APP_PASSWORD` (Gmail)
- Optionnel : Docker Desktop, Claude Desktop, `cloudflared` pour la démo MCP

### Étapes

**1. Cloner**
```bash
git clone https://github.com/diegomerchanm/catastrophes-climatiques-rag.git
cd catastrophes-climatiques-rag
```

**2. Environnement virtuel**
```bash
python -m venv venv
venv\Scripts\activate    # Windows PowerShell
source venv/bin/activate  # Mac/Linux
```

**3. Dépendances**
```bash
pip install -r requirements.txt
```

**4. Variables d'environnement**
```bash
cp .env.example .env
# Renseigner : ANTHROPIC_API_KEY, TAVILY_API_KEY, EMAIL_ADDRESS,
# EMAIL_APP_PASSWORD, EMAIL_RECIPIENT, CHAINLIT_AUTH_SECRET,
# USER_ACCOUNTS_JSON, TEAM_DIRECTORY_JSON, MLFLOW_TRACKING_URI
```

**5. Télécharger les PDFs dans `data/raw/`** (voir section Corpus)

**6. Générer le vector store FAISS**
```bash
python -m src.rag.embeddings
```

**7. Lancer l'app**
```bash
chainlit run app.py
```

Ouvrir http://localhost:8000, se connecter avec `demo@saearch.ai / demo`.

---

## Utilisation avancée

### Serveur MCP pour Claude Desktop

**1. Lancer le serveur local**
```bash
python mcp_server.py
# ou via le proxy pour Streamable HTTP (Claude Desktop 2025+)
mcp-proxy --sse-port 8765 -- python mcp_server.py
```

**2. Exposer en HTTPS public (pour Claude Desktop)**
```bash
cloudflared tunnel --url http://localhost:63119
# Retourne : https://xxx-xxx.trycloudflare.com
```

**3. Dans Claude Desktop** → Paramètres → Connecteurs → Ajouter un connecteur personnalisé
- URL : `https://xxx-xxx.trycloudflare.com/mcp`

Les 11 outils SAEARCH apparaissent et sont appelables par Claude.

### Rapport hebdomadaire automatique

Le workflow `.github/workflows/weekly-report.yml` exécute `scheduled_report.py` tous les lundis à 8h UTC (10h Paris) et envoie un rapport climatique multisources par email.

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
├── Dockerfile                      # Build reproductible --no-cache
├── README.md
├── CLAUDE.md                       # Conventions de code + retex loan-default
├── .env.example
├── requirements.txt
│
├── data/
│   ├── raw/                        # PDFs corpus (non versionnés)
│   └── decadal-*.csv               # Données OWID pour NB10 ML
│
├── src/
│   ├── config.py                   # AGENT_CONFIGS (spécialisation LLM), TokenCounter
│   ├── rag/
│   │   ├── loader.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── hybrid_retriever.py     # EnsembleRetriever BM25+Dense
│   ├── agents/
│   │   ├── tools.py                # 13 outils @tool
│   │   └── agent.py                # Agent ReAct LangGraph, mémoire, MLflow
│   ├── memory/
│   │   └── memory.py               # InMemoryChatMessageHistory
│   ├── router/
│   │   └── router.py               # Routing RAG / Agent / Chat
│   ├── prompts/
│   │   └── agent_prompts.py        # v1.0 + v2.0 versionnés
│   └── ui/
│       ├── donut_chart.py          # SVG donut inline
│       └── data_layer.py           # SQLite threads Chainlit
│
├── notebooks/
│   ├── 01_exploration_corpus.ipynb
│   ├── 02_test_rag.ipynb
│   ├── 03_test_agents.ipynb
│   ├── NB4_Memoire_Multilingue.ipynb
│   ├── NB5_Analyse_Risque_Predictive.ipynb
│   ├── NB6_Comparatifs_MLflow.ipynb
│   ├── NB7_MCP_Email.ipynb
│   ├── NB8_LLMOps_Monitoring.ipynb
│   ├── NB9_CICD_Docker_Deploiement.ipynb
│   └── NB10_ML_Predictif_Catastrophes.ipynb
│
├── scripts/
│   └── teaser_vendredi.py          # Teaser soutenance (cron one-shot)
│
├── tests/
│   ├── test_config.py
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_structure.py
│
├── .github/workflows/
│   ├── github-docker-cicd.yaml     # CI : black + pylint + pytest → Docker Hub
│   ├── azure.yml                   # CD Azure Container Apps (plan B)
│   ├── weekly-report.yml           # Cron lundi 8h UTC
│   └── teaser-vendredi.yml         # One-shot 17/04 8h UTC
│
├── outputs/                        # Artefacts NB (CSV, PNG, joblib ML, SVG donut)
├── faiss_store/                    # Vector store persisté (non versionné)
├── mlflow.db                       # Backend SQLite MLflow
└── scheduler_jobs.db               # Persistance APScheduler
```

---

## Équipe

| Phase | Responsabilité |
|---|---|
| **P1** | Corpus & RAG — `loader.py`, `embeddings.py`, `retriever.py`, hybride BM25+Dense |
| **P2** | Agents & Tools — `tools.py`, `agent.py`, 13 outils, ReAct LangGraph |
| **P3** | Router & UI — `router.py`, `app.py` streaming, badges |
| **P4** | Agentic RAG avancé, Mémoire, MLOps/LLMOps, Notebooks NB4-NB10, MCP, HITL, UI décisionnelle, CI/CD, déploiement |

*Les noms des membres de l'équipe ne sont pas publiés ici — ils figurent dans le dossier de rendu académique.*

---

## Notebooks

| Notebook | Contenu | Auteur |
|---|---|---|
| 01-03 | Exploration corpus, tests RAG, tests agents | Équipe P1/P2/P3 |
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
