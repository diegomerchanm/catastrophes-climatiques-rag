# Architecture du projet — Système agentique SAEARCH

> **Version 1.1** — Refonte du 17/04/2026, alignée sur le code. Version précédente conservée dans [`archive/architecture_v1.0.md`](archive/architecture_v1.0.md).

> Document de référence : comprendre comment les briques techniques s'articulent, et pourquoi chacune est là.

**SAEARCH** (*Système Agentique d'Évaluation et d'Anticipation des Risques Climatiques et Hydrologiques*) est un système agentique multi-compétences qui combine RAG scientifique, agents à outils, machine learning prédictif et aide à la décision humaine. Il porte le nom **DooMax** côté UI.

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                          UTILISATEUR                             │
│        texte / voix (STT) / PDF / DOCX / image / audio           │
│                  FR / EN / ES / DE / 25 langues UI               │
└──────────────┬──────────────┬──────────────┬────────────────────┘
               │              │              │
         ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼──────┐
         │  Chainlit  │  │    MCP    │  │  Job Cron   │
         │  (app.py)  │  │ (serveur) │  │ (lundi 8h)  │
         └─────┬──────┘  └─────┬─────┘  └──────┬──────┘
               │               │               │
               └───────────────┼───────────────┘
                               │
                     ┌─────────▼─────────┐
                     │     MÉMOIRE       │
                     │   (memory.py)     │
                     │  fenêtre 20 tours │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │   ROUTER (3 voies) │
                     │  keywords > LLM    │
                     │  classifier        │
                     └─┬────────┬────────┬┘
                       │        │        │
                  ┌────▼──┐ ┌──▼──┐ ┌───▼────┐
                  │ CHAT  │ │ RAG │ │ AGENT  │
                  │ direct│ │     │ │ ReAct  │
                  └────┬──┘ └──┬──┘ └───┬────┘
                       │       │        │
                       │       │   ┌────▼──────────────────────┐
                       │       │   │  13 OUTILS @tool          │
                       │       │   ├───────────────────────────┤
                       │       │   │ Météo ×3   (OpenMeteo)    │
                       │       │   │ web_search (Tavily/DDG)   │
                       │       │   │ calculator (sandbox AST)  │
                       │       │   │ search/list_corpus (RAG)  │
                       │       │   │ Email ×3 (SMTP+APS+SQLite)│
                       │       │   │ ML ×2    (joblib NB10)    │
                       │       │   │ score_risque (4 sources)  │
                       │       │   └────┬──────────────────────┘
                       │       │        │
                       │   ┌───▼────────▼──────────────────┐
                       │   │  RAG HYBRIDE                   │
                       │   │  BM25 + Dense (FAISS)          │
                       │   │  + EnsembleRetriever 50/50     │
                       │   │  + CrossEncoder reranking      │
                       │   │  + Bridge lexical FR-EN        │
                       │   │  + Diversité forcée 3/PDF      │
                       │   │  → [Source: fichier, Page: N]  │
                       │   └────────────────────────────────┘
                       │
                  ┌────▼─────────────────────────────────────┐
                  │  LLM ANTHROPIC CLAUDE — fallback x3      │
                  │  Sonnet 4 (principal)                    │
                  │  → Haiku 4.5                             │
                  │  → Ollama Mistral local (hors ligne)     │
                  └────┬─────────────────────────────────────┘
                       │
                  ┌────▼─────────────────────────────────────┐
                  │  RÉPONSE STREAMÉE                        │
                  │  + sources citées                        │
                  │  + donut SVG outils appelés              │
                  │  + tokens / coût / durée                 │
                  │  + Human-in-the-Loop si mode décisionnel │
                  └────┬─────────────────────────────────────┘
                       │
                  ┌────▼─────────────────────────────────────┐
                  │  MLflow tracking (1 run / requête)       │
                  │  prompt_version, tokens, coût USD,       │
                  │  outils appelés, latence, git_commit     │
                  └──────────────────────────────────────────┘
```

---

## Construction par phases

Le projet est construit en quatre phases, chacune posant des fondations sur lesquelles la suivante s'appuie. Cette progression collective est la clé de voûte de l'architecture : **les innovations P4 ne remplacent pas les briques P1/P2/P3, elles les étendent.**

| Phase | Périmètre | Apports principaux |
|---|---|---|
| **P1** | Corpus & RAG | Pipeline d'ingestion PDF, chunking, embeddings FAISS, retriever MMR, métadonnées source/page |
| **P2** | Agents & Tools | Agent LangGraph ReAct, premiers outils (météo, web, calcul) |
| **P3** | Router & UI | Interface Chainlit streaming, routing conditionnel 3 voies, traduction UI |
| **P4** | Agentic RAG avancé | Mémoire, MLOps, observabilité, MCP, ML prédictif, multimodal, CI/CD, déploiement |

Quelques exemples concrets de cet héritage :

- L'**embedding multilingue** (P4) repose sur le pipeline FAISS et le chunking validés en P1
- Les **10 outils ajoutés en P4** réutilisent le pattern ReAct et le décorateur `@tool` mis en place en P2
- L'**upload session-scoped** (P4) s'appuie sur les hooks Chainlit posés en P3
- Le **fallback LLM à 3 niveaux** (P4) s'insère dans l'agent existant sans casser son interface

---

## Interfaces d'accès

| Interface | Fichier | Usage |
|---|---|---|
| **Chainlit** | `app.py` | UI web conversationnelle, streaming, auth, multi-conversations, donut, STT, upload |
| **MCP** | `mcp_server.py` | 11 outils exposés via FastMCP pour Claude Desktop, Cursor, Continue, Cline |
| **Cron weekly** | `scheduled_report.py` + `.github/workflows/weekly-report.yml` | Rapport hebdomadaire automatique (lundi 8h UTC) |
| **Cron one-shot** | `scripts/teaser_vendredi.py` + `.github/workflows/teaser-vendredi.yml` | Teaser ponctuel avant soutenance |

---

## Routing conditionnel

Le router décide en amont quelle voie activer. Trois niveaux successifs, du plus rapide au plus coûteux :

1. **Keywords prioritaires** (`AGENT_PRIORITY`) : termes déclenchant immédiatement l'agent — `météo`, `recherche`, `email`, `risque`, `calcul`, `score`, `predict`, etc.
2. **Keywords RAG** : termes liés au corpus — `corpus`, `giec`, `ipcc`, `copernicus`, `em-dat`, `inondation`, `sécheresse`...
3. **Keywords Agent** complémentaires : `pdf`, `inventaire`, `fichier`...
4. **Fallback LLM classifier** : si aucun keyword n'a tranché, un mini-prompt Claude Haiku catégorise (chat / rag / agent).

Cette stratégie hiérarchique garantit une latence faible : les questions évidentes ne paient jamais le coût d'un appel LLM de classification.

---

## Pipeline RAG hybride

```
PDFs (data/raw/, 10 documents)
    ↓ loader.py — PyPDFLoader + RecursiveCharacterTextSplitter
Chunks (1500 caractères, overlap 150) → 1 889 chunks
    ↓ embeddings.py — paraphrase-multilingual-MiniLM-L12-v2 (50+ langues)
FAISS vector store (faiss_store/, persisté, versionné 5 MB)
    ↓ retriever.py — MMR k=12, fetch_k=40, lambda_mult=0.7
    ↓ hybrid_retriever.py — EnsembleRetriever (BM25 50% + Dense 50%)
    ↓ Reranking CrossEncoder MS-MARCO + placement stratégique
    ↓ Bridge lexical FR↔EN (20 termes : GIEC↔IPCC, OMM↔WMO...)
    ↓ Diversité forcée (max 3 chunks par PDF)
Chunks pertinents avec citations [Source: fichier, Page: N]
```

Ce pipeline s'attaque à plusieurs limitations classiques du RAG :

- **Cross-lingue** : une question en français peut retrouver des passages anglais via l'embedding multilingue + le bridge lexical
- **Lost in the Middle** : le placement stratégique des chunks compense la dégradation d'attention au milieu des contextes longs
- **Dominance documentaire** : la diversité forcée empêche un gros document (par exemple `Forest_Fires_2024` avec ~700 chunks) d'écraser les autres dans le top-12

---

## Agent ReAct et 13 outils

L'agent suit le pattern **Reason → Act → Observe → Answer**, implémenté via LangGraph. Les outils sont regroupés par famille :

| Famille | Outils | Source / Backend |
|---|---|---|
| **Météo** (×3) | `get_weather`, `get_historical_weather`, `get_forecast` | OpenMeteo (gratuit, sans clé) |
| **RAG** (×2) | `search_corpus`, `list_corpus` | FAISS + BM25 hybride |
| **Web** | `web_search` | Tavily (priorité) + DuckDuckGo (fallback) |
| **Calcul** | `calculator` | Sandbox AST whitelist (anti-injection) |
| **Email** (×3) | `send_email`, `send_bulk_email`, `schedule_email` | SMTP Gmail + APScheduler SQLite persistant |
| **ML** (×2) | `predict_risk`, `predict_risk_by_type` | Modèles joblib NB10 (EM-DAT décadal) |
| **Scoring** | `calculer_score_risque` | Combinaison pondérée 4 sources |

L'outil de scoring est emblématique : il **chaîne quatre autres outils** (météo + ML + RAG + historique) en une seule passe et synthétise un score 0-1, avec pondérations variables selon l'horizon temporel demandé (`court_terme` / `standard` / `long_terme`).

---

## Spécialisation des LLM

Les modèles Anthropic Claude sont sélectionnés selon le rôle, pour équilibrer qualité et coût (voir `src/config.py`, dictionnaire `AGENT_CONFIGS`) :

| Rôle | Modèle | Justification |
|---|---|---|
| Orchestrateur agent | Sonnet 4 | Raisonnement et choix d'outils, bon équilibre qualité / coût |
| Analyste (mode décisionnel) | Opus 4 | Analyse de risque complexe, croisement multi-sources |
| RAG (synthèse) | Sonnet 4 | Synthèse de documents longs |
| Météo / Web | Haiku 4.5 | Tâches factuelles courtes, économique |
| Chat direct | Haiku 4.5 | Conversation simple, latence minimale |
| Classifier router | Haiku 4.5 | Catégorisation 1-mot, ultra-rapide |

---

## Fallback LLM (3 niveaux)

```
Sonnet 4  (principal — orchestrateur)
   │  échec / 429 / quota
   ▼
Haiku 4.5  (fallback Anthropic économique)
   │  échec / panne API totale
   ▼
Ollama Mistral local  (hors ligne, dernier recours)
```

Cette redondance protège contre les rate-limits 429, les pannes API et les quotas épuisés. La dernière ligne de défense est entièrement locale, ce qui garantit qu'une démo ne s'effondre pas si Anthropic a un incident pendant la soutenance.

---

## Multimodal (entrée et sortie)

Sur les fondations Chainlit posées en P3, l'application accepte quatre types d'entrée et offre une sortie vocale :

| Type | Traitement |
|---|---|
| **PDF / DOCX** (entrée) | Indexation FAISS in-memory **session-scoped** (anti-pollution du corpus officiel) + texte injecté au LLM |
| **Image** (JPG/PNG/GIF/WEBP, entrée) | Transmise en base64 au LLM Claude vision (multimodal natif) |
| **Audio** (WAV/MP3/OGG, entrée) | Transcription locale via Faster-Whisper (modèle `small`, CPU, int8, **autodétection de langue**) |
| **STT temps réel** (entrée) | Capture micro Chainlit, conversion PCM16 → WAV, transcription Faster-Whisper avec autodétection de langue |
| **TTS bouton flottant 🔊** (sortie) | Lecture vocale de la dernière réponse via la **Web Speech API** du navigateur (`public/tts.js`). Bouton fixe en bas à gauche, vert, sans appel serveur ni coût |

L'isolation par session garantit que les uploads d'un utilisateur ne polluent jamais le corpus de référence ni les sessions des autres utilisateurs.

## Géolocalisation utilisateur

Deux mécanismes complémentaires, gérés côté navigateur (`public/geoloc.js`) puis stockés dans `sessionStorage` :

- **Géoloc native** : `navigator.geolocation.getCurrentPosition()` → coordonnées GPS précises (sous réserve d'autorisation utilisateur)
- **Géoloc par IP** (fallback) : appel à `ip-api.com` au login, donne une position approximative

La position est réinjectée dans le prompt système de l'agent pour contextualiser les réponses météo / risque par défaut.

---

## Mémoire conversationnelle

- `InMemoryChatMessageHistory` côté LangChain
- Fenêtre glissante de 20 tours par session (limite tokens, garde le contexte récent)
- Persistance des threads dans SQLite (data layer Chainlit) pour retrouver les conversations passées dans la sidebar
- Le prénom de l'utilisateur (login ou prénom déclaré en cours de discussion) est mémorisé et réinjecté dans le prompt système

---

## Aide à la décision humaine (HITL)

Un mode décisionnel propose 4 profils utilisateurs (événementiel, assurance, autorité publique, tourisme). Pour chaque scénario :

1. Le système calcule un score 0-1 via `calculer_score_risque`
2. Il propose une recommandation **GO / NO-GO** ou **Alerte / Pas d'alerte**
3. Une boucle **Human-in-the-Loop** demande validation : *Approuver / Enrichir / Rejeter*
4. La décision finale est loggée dans MLflow comme feedback humain (base d'amélioration future des prompts)

Chaque sortie décisionnelle se termine par un **guide de lecture** explicite, par exemple :
> *Guide de lecture : DECISION = verdict opérationnel | HORIZON = fenêtre temporelle qui détermine la pondération des sources | SCORE = indice de risque 0-1 | RISQUE = traduction qualitative du score | CONFIANCE = qualité des sources*

Cette légende rend la sortie auto-explicative : un utilisateur découvrant SAEARCH peut comprendre les termes techniques sans documentation externe. L'agent utilise par défaut le **vouvoiement** pour s'adresser à l'utilisateur, en cohérence avec le ton professionnel de l'outil.

---

## Exposition MCP

Le serveur `mcp_server.py` (FastMCP) expose **11 des 13 outils** au protocole Model Context Protocol. Tout client compatible peut les consommer :

- **Claude Desktop** (via tunnel Cloudflare → URL HTTPS publique `/mcp`)
- **Claude Code** (stdio local)
- **Cursor**, **Continue.dev**, **Cline**, etc.

Les 2 outils non exposés (`send_bulk_email`, `schedule_email`) restent internes par **principe de moindre privilège** : on évite qu'un client MCP externe puisse spammer des destinataires ou planifier des envois persistants à l'insu de l'opérateur humain.

Point fort architectural : **aucune duplication de code**. Les mêmes 11 fonctions `@tool` sont réutilisées entre l'agent Chainlit et le serveur MCP.

---

## ML prédictif (NB10)

Pipeline complet entraîné sur les données EM-DAT décadales (14 625 lignes, 225 pays × 5 types de catastrophes) :

- **Régression** : Gradient Boosting Quantile Regression (perte médiane), sélectionnée par MAE_test pour sa robustesse aux outliers (canicule France 2003)
- **Classification multi-classe** : 5 niveaux de risque (faible / modéré / élevé / critique / extrême)
- **Clipping** : prédictions plafonnées à 1.5× le maximum historique décadal observé
- **Grille comparative** : 8 régresseurs + 8 classifieurs comparés à chaque entraînement (16 nested runs MLflow par entraînement)
- **Model Registry MLflow** : `NB10_regression v1+`, `NB10_classification v1+`
- **Artefacts** : 7 joblibs exportés dans `outputs/`, consommés par `predict_risk` et `predict_risk_by_type`

---

## Observabilité LLMOps

Chaque requête utilisateur produit un run MLflow comprenant :

| Métrique | Description |
|---|---|
| `prompt_version` | v1.0 ou v2.0 (A/B testing) |
| `tokens_in` / `tokens_out` | Compteurs Anthropic exacts |
| `cout_usd` | Estimation pricing Anthropic (Opus / Sonnet / Haiku) |
| `duree_s` | Latence end-to-end |
| `nb_outils` / `outils_appeles` | Tracing via events LangGraph |
| `git_commit` / `env` | Traçabilité du déploiement |

Le dashboard MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) permet de filtrer les requêtes coûteuses, comparer prompts v1/v2, ou détecter une régression de latence.

---

## Pipeline CI/CD

```
git push
    ↓
GitHub Actions — ci_pipeline (github-docker-cicd.yaml)
    ├── black --check
    ├── pylint --disable=R,C (exit-zero)
    └── pytest -vv (43 tests)
    ↓ si CI OK
GitHub Actions — cd_pipeline
    ├── docker build --no-cache (reproductibilité)
    ├── docker push (Docker Hub : :latest + tag date)
    └── Trigger HF Spaces rebuild
```

L'option `--no-cache` est intentionnelle : leçon retenue d'un projet antérieur où Render réutilisait les layers de cache et masquait des régressions silencieuses. Ici chaque image est reconstruite à partir de zéro, garantissant la reproductibilité.

**Cibles de déploiement** :

| Cible | État | Usage |
|---|---|---|
| **HF Spaces** | Live (`xbizot-saearch.hf.space`) | Démo principale, gratuit |
| **Docker Hub** | Image `latest` poussée à chaque commit `main` | Pull direct utilisateur |
| **AWS ECR** | Image disponible (`310971189093.dkr.ecr.eu-west-3.amazonaws.com/saearch:latest`) | Artefact pédagogique, plan B |
| **Azure Container Apps** | Workflow `azure.yml` en `workflow_dispatch` | Plan B documenté |

---

## Cron weekly-report

```
Lundi 8h UTC
    ↓
Recherche web : catastrophes récentes (web_search Tavily)
    ↓
Prévisions météo : villes surveillées (get_forecast)
    ↓
Croisement seuils GIEC (search_corpus)
    ↓
Génération rapport + alertes si seuils dépassés
    ↓
Envoi email (send_email aux destinataires configurés)
```

---

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

---

## Pour aller plus loin

- [`decisions.md`](decisions.md) — Architecture Decision Records (le *pourquoi* derrière les choix techniques)
- [`architecture_pour_slides.md`](architecture_pour_slides.md) — Version condensée pour la soutenance (diagramme + chiffres clés + innovations)
- [`glossaire.md`](glossaire.md) — Définitions courtes des termes techniques (RAG, MMR, BM25, HITL, MCP...)
- [`demo_script.md`](demo_script.md) — Script de démonstration pas-à-pas pour la soutenance
- [`faq.md`](faq.md) — Questions probables du jury et réponses préparées
- [`../README.md`](../README.md) — Vue d'ensemble du projet, installation, usage
- [`../CLAUDE.md`](../CLAUDE.md) — Conventions de code, retex projet
