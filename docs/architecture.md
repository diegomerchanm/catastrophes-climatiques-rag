# Architecture du projet — Assistant RAG Catastrophes Climatiques

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        UTILISATEUR                               │
│                  (texte / voix / PDF)                             │
└──────────────┬──────────────┬──────────────┬────────────────────┘
               │              │              │
         ┌─────▼─────┐ ┌─────▼─────┐ ┌──────▼──────┐
         │  Chainlit  │ │    MCP    │ │  Job Cron   │
         │  (app.py)  │ │ (serveur) │ │ (lundi 8h)  │
         └─────┬──────┘ └─────┬─────┘ └──────┬──────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │     MÉMOIRE       │
                    │  (memory.py)      │
                    │  fenêtre 20 msg   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  AGENT ORCHESTR.  │
                    │  (agent.py)       │
                    │  Sonnet — ReAct   │
                    │  Fallback: Haiku  │
                    │  Fallback: Ollama │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │          BOUCLE ReAct         │
              │  Reason → Act → Observe →     │
              │  Repeat → ... → Answer        │
              └───────┬───────────────┬───────┘
                      │               │
        ┌─────────────▼───────────────▼─────────────┐
        │              7 OUTILS (tools.py)           │
        ├────────────────────────────────────────────┤
        │                                            │
        │  ┌──────────────────┐  ┌────────────────┐  │
        │  │  search_corpus   │  │  get_weather   │  │
        │  │  RAG hybride     │  │  météo actuelle│  │
        │  │  BM25 + Dense    │  │  (OpenMeteo)   │  │
        │  │  + reranking     │  │                │  │
        │  └────────┬─────────┘  └────────────────┘  │
        │           │                                 │
        │  ┌────────▼─────────────────────────────┐  │
        │  │  Pipeline RAG (code P1)           │  │
        │  │  loader → embeddings → retriever     │  │
        │  │  + hybrid_retriever (BM25+Dense)     │  │
        │  │  + reranking (cross-encoder)         │  │
        │  │  + placement stratégique             │  │
        │  │  → FAISS vector store                │  │
        │  │  → [Source: fichier, Page: X]        │  │
        │  └──────────────────────────────────────┘  │
        │                                            │
        │  ┌──────────────────┐  ┌────────────────┐  │
        │  │ get_historical   │  │  get_forecast  │  │
        │  │ _weather         │  │  prévisions    │  │
        │  │ météo passée     │  │  7 jours       │  │
        │  │ (OpenMeteo       │  │  (OpenMeteo)   │  │
        │  │  Archive)        │  │                │  │
        │  └──────────────────┘  └────────────────┘  │
        │                                            │
        │  ┌──────────────────┐  ┌────────────────┐  │
        │  │  web_search      │  │  calculator    │  │
        │  │  Tavily (prio)   │  │  math sécurisé │  │
        │  │  DDG (fallback)  │  │  garde-fou     │  │
        │  └──────────────────┘  │  regex         │  │
        │                        └────────────────┘  │
        │  ┌──────────────────┐                      │
        │  │  send_email      │                      │
        │  │  alertes SMTP    │                      │
        │  │  Gmail           │                      │
        │  └──────────────────┘                      │
        └────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    MONITORING     │
                    │  TokenCounter     │
                    │  MLflow tracking  │
                    │  Logging          │
                    │  Estimation coût  │
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
        │   config.py │ │ prompts/ │ │   tests/    │
        │ Haiku/Son/  │ │ v1.0     │ │ 43 tests    │
        │ Opus        │ │ v2.0     │ │ pytest      │
        │ pricing     │ │ A/B test │ │             │
        └─────────────┘ └──────────┘ └─────────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
        │ Dockerfile  │ │ CI/CD    │ │   Azure     │
        │ --no-cache  │ │ black    │ │ Container   │
        │ Python 3.11 │ │ pylint   │ │ Apps        │
        │ Chainlit    │ │ pytest   │ │ (deploy)    │
        └─────────────┘ │ Docker   │ └─────────────┘
                        └──────────┘
```

## Interfaces d'accès

| Interface | Fichier | Usage |
|---|---|---|
| **Chainlit** | `app.py` | Interface web conversationnelle, streaming, upload PDF, STT |
| **MCP** | `mcp_server.py` | Outils accessibles dans Claude Desktop |
| **Job Cron** | `scheduled_report.py` | Rapport hebdomadaire automatique (lundi 8h) |

## 7 Outils de l'agent

| Outil | Source | Fonction |
|---|---|---|
| `search_corpus` | FAISS + BM25 + reranking | Recherche hybride dans le corpus GIEC/Copernicus/EM-DAT |
| `get_weather` | OpenMeteo | Météo actuelle d'une ville |
| `get_historical_weather` | OpenMeteo Archive | Météo d'une date passée |
| `get_forecast` | OpenMeteo Forecast | Prévisions 7 jours |
| `web_search` | Tavily + DuckDuckGo | Recherche web (Tavily prio, DDG fallback) |
| `calculator` | eval sécurisé | Calculs mathématiques |
| `send_email` | SMTP Gmail | Envoi d'alertes climatiques |

## Spécialisation des LLM

| Agent | Modèle | Justification |
|---|---|---|
| Orchestrateur | Claude Sonnet 4.5 | Raisonnement et choix des outils |
| RAG | Claude Sonnet 4.5 | Synthèse de documents longs |
| Météo | Claude Haiku 4.5 | Tâche factuelle, économique |
| Web | Claude Haiku 4.5 | Synthèse courte |
| Analyste | Claude Opus 4.6 | Analyse de risque complexe |
| Chat | Claude Haiku 4.5 | Conversation simple |

## Chaîne de fallback

```
Sonnet (principal)
    ↓ échec
Haiku (fallback Anthropic)
    ↓ échec
Mistral via Ollama (fallback local, hors ligne)
```

## Pipeline RAG

```
PDFs (data/raw/)
    ↓ loader.py
Chunks (1500 car, overlap 150)
    ↓ embeddings.py (all-MiniLM-L6-v2)
FAISS vector store (faiss_store/)
    ↓ retriever.py (MMR k=4, fetch_k=10)
    ↓ hybrid_retriever.py (BM25 50% + Dense 50%)
    ↓ reranking (cross-encoder ms-marco-MiniLM-L-6-v2)
    ↓ placement stratégique (Lost in the Middle)
Chunks pertinents avec citations [Source: fichier, Page: X]
```

## Pipeline CI/CD

```
git push
    ↓
GitHub Actions CI (github-docker-cicd.yaml)
    ├── black --check
    ├── pylint --disable=R,C
    └── pytest -vv
    ↓ si CI passe
GitHub Actions CD
    ├── Docker build --no-cache
    ├── Docker push (Docker Hub)
    └── Deploy Azure Container Apps (azure.yml)
```

## Pipeline Cron (weekly-report.yml)

```
Lundi 8h UTC
    ↓
Recherche web : catastrophes récentes
    ↓
Prévisions météo : 5 villes surveillées
    ↓
Croisement seuils GIEC (search_corpus)
    ↓
Rapport + alertes si seuils dépassés
    ↓
Envoi email (send_email)
```

## Corpus scientifique (10 PDFs)

| Document | Thème | Période |
|---|---|---|
| GIEC AR6 — Synthesis Report SPM | Synthèse changement climatique | 2023 |
| GIEC AR6 — WG II SPM | Impacts et adaptation | 2022 |
| GIEC AR6 — WG III SPM | Atténuation | 2022 |
| Copernicus — European State of Climate | Climat européen | 2023 |
| UNDRR — Global Assessment Report | Réduction des risques | 2022 |
| EM-DAT — Natural Disasters Report | Catastrophes naturelles | 2023 |
| WMO — State of Global Water Resources | Ressources en eau | 2024 |
| NOAA — Atlantic Hurricane Season | Ouragans atlantiques | 2023 |
| JRC — Forest Fires in Europe | Feux de forêt | 2024 |
| EU CELEX — Floods Directive | Directive inondations | 2021 |

## Monitoring LLMOps

| Composant | Fonction |
|---|---|
| `TokenCounter` (config.py) | Comptage tokens par agent, estimation coût |
| `MLflow` (agent.py) | Tracking de chaque prédiction : question, outils, tokens, coût, durée |
| `logging` (tous les modules) | Traçabilité des appels d'outils et décisions |
| `PROMPT_VERSION` (agent_prompts.py) | Versioning des prompts pour A/B testing |

## Équipe

| Phase | Rôle | Branche | Responsabilité |
|---|---|---|---|
| P1 | Chief Data Super Engineer | `feature/p1-rag-corpus` | RAG pipeline (loader, embeddings, retriever) |
| P2 | Backend Developer Engineer | `amorphya/agenttools` | Agents & Tools (tools.py, agent.py) |
| P3 | CTO | `feature/p3-router-ui` | Router & UI Chainlit (router.py, app.py) |
| P4 | Chef de Projet | `feature/p4-memory-agent` | Mémoire, Agentic RAG, MLOps/LLMOps, Notebooks |
