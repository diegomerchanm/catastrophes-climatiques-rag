# CLAUDE.md — catastrophes-climatiques-rag

## Projet

Assistant intelligent Agentic RAG multi-compétences pour l'analyse des catastrophes climatiques.
Combine RAG sur corpus scientifique + agents météo/web/calcul + analyse de risque prédictive.

## Dataset / Corpus

- **Dossier** : `data/raw/` (ne jamais uploader dans le repo)
- **10 rapports PDF** : GIEC AR6, Copernicus 2023, EM-DAT 2023, NOAA, JRC, WMO, EU Floods Directive
- **Vector store** : `faiss_store/` (généré localement, ignoré par git)

## Stack technique

- **Langage** : Python 3.11
- **LLM** : Anthropic Claude (Haiku 4.5 / Sonnet 4.5 / Opus 4.6 selon l'agent)
- **Embeddings** : HuggingFace all-MiniLM-L6-v2 (local, gratuit)
- **Vector store** : FAISS
- **Framework** : LangChain + LangGraph
- **Recherche** : Hybride BM25 + Dense (EnsembleRetriever)
- **UI** : Chainlit
- **API météo** : OpenMeteo (gratuit, sans clé)
- **Recherche web** : DuckDuckGo
- **Conteneurisation** : Docker
- **CI/CD** : GitHub Actions (black + pylint + pytest + Docker Hub)
- **Déploiement** : Azure Container Apps (retex loan-default : rebuild from scratch, pas Render)

## Structure du projet

```
catastrophes-climatiques-rag/
├── data/
│   └── raw/                    # PDFs du corpus (non versionnés)
├── src/
│   ├── config.py               # Configuration centralisée et spécialisation LLM
│   ├── rag/
│   │   ├── loader.py           # Chargement et découpage des PDFs
│   │   ├── embeddings.py       # Création et persistance du vector store FAISS
│   │   └── retriever.py        # Récupération MMR avec citations
│   ├── agents/
│   │   ├── tools.py            # 6 outils : météo (3), web, calcul, RAG
│   │   └── agent.py            # Agent agentic RAG multi-compétences (ReAct)
│   ├── memory/
│   │   └── memory.py           # Mémoire conversationnelle (InMemoryChatMessageHistory)
│   └── router/
│       └── router.py           # Routing conditionnel (mode linéaire)
├── tests/                      # Tests pytest
├── notebooks/
│   ├── 01_test_rag.ipynb
│   ├── 02_test_agents.ipynb
│   ├── 03_demo_analyse_risque.ipynb
│   └── 04_comparatifs.ipynb
├── .github/workflows/ci.yml   # Pipeline CI/CD
├── app.py                      # Point d'entrée Chainlit
├── Dockerfile
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

## Équipe et branches

| Branche | Membre | Responsabilité |
|---|---|---|
| `main` (P1) | Diego Merchán | Corpus & RAG (loader, embeddings, retriever) |
| `amorphya/agenttools` (P2) | Camille Koenig | Agents & Tools (tools.py, agent.py) |
| `feature/p3-router-ui` (P3) | Jayson Phan Nguyen | Router & UI Chainlit (router.py, app.py) |
| `feature/p4-memory-agent` (P4) | Xia Bizot | Mémoire, Agentic RAG, MLOps, Notebooks |

## Phases du projet

| Phase | Nom | Description |
|---|---|---|
| 0 | Setup | Initialisation du repo, structure, .env |
| 1 | RAG | Pipeline ingestion, embeddings FAISS, retriever MMR |
| 2 | Agents & Tools | 6 outils + agent LangGraph ReAct |
| 3 | Router & UI | Routing conditionnel + interface Chainlit streaming |
| 4 | Agentic RAG | Multi-agents, mémoire, analyse de risque, MLOps |
| 5 | Notebooks | Tests, métriques, démo analyse de risque |
| 6 | CI/CD | Docker, GitHub Actions, déploiement Azure |
| 7 | Soutenance | Présentation finale |

## Conventions de code

### Langue
- **Variables, fonctions, commentaires, docstrings** : en français
- **Noms de modules et fichiers** : en français ou anglais technique (snake_case)

### Nommage
- **Fonctions et variables** : `snake_case` (ex: `creer_retriever`, `chat_history`)
- **Constantes** : `UPPER_CASE` (ex: `CHUNK_SIZE`, `MODEL_SONNET`)
- **Classes** : `PascalCase` (ex: `TokenCounter`, `AgentState`)
- **Fichiers** : `snake_case.py` (ex: `hybrid_retriever.py`)

### Docstrings
- **Format Google** pour toutes les fonctions publiques
- Inclure : description, Args, Returns
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
- Pas d'emojis dans le code source ni les retours d'outils
- Emojis autorisés uniquement dans l'interface Chainlit (app.py)

### Formattage et qualité
- **Formatteur** : black (longueur max 88 caractères)
- **Linter** : pylint (--disable=R,C)
- **Tests** : pytest
- Pas d'imports inutilisés
- Pas de code mort (variables/fonctions jamais appelées)

### Git
- **Commits** : Conventional Commits
  - `feat:` nouvelle fonctionnalité
  - `fix:` correction de bug
  - `chore:` tâche technique (dépendances, config)
  - `docs:` documentation
- **Ne jamais committer sur main directement**
- **Ne jamais committer de fichiers de données** (PDFs, CSV, modèles)

## Règles de sécurité

- Ne **jamais** uploader les PDFs du corpus dans le repo
- Ne **jamais** hardcoder des credentials (clés API, tokens)
- Utiliser des variables d'environnement via `.env` (listé dans `.gitignore`)
- Le `.env.example` sert de modèle

## Retex projet loan-default-mlops

- Render ne rebuild pas l'image complète (incrémental) → solution non reproductible
- Sur ce projet : Docker build `--no-cache` systématique en CI/CD
- Déploiement Azure Container Apps au lieu de Render
