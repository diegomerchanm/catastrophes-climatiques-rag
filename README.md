# Assistant de Prédiction des Catastrophes Climatiques

## Description

Un assistant intelligent combinant RAG et agents pour répondre aux questions sur les catastrophes climatiques. Le système peut :
- Répondre aux questions basées sur un corpus de rapports scientifiques (GIEC, NOAA, Copernicus, UNDRR)
- Appeler des outils externes (météo en temps réel, recherche web, calculatrice)
- Maintenir une conversation contextuelle avec l'utilisateur
- Router intelligemment entre RAG, agents, et réponse directe

## Architecture

Le projet est organisé autour de 4 composants principaux :

- **RAG Pipeline** (`src/rag/`) : charge les PDFs du corpus, les découpe en chunks, génère les embeddings et les indexe dans FAISS. Lors d'une question, les documents les plus pertinents sont récupérés via MMR et injectés dans le prompt.
- **Agents & Tools** (`src/agents/`) : un agent LangGraph peut appeler 3 outils — météo en temps réel (OpenMeteo), recherche web (DuckDuckGo) et calculatrice — pour répondre aux questions nécessitant des données dynamiques.
- **Router** (`src/router/`) : analyse la question entrante et décide du chemin de traitement : RAG (question sur le corpus), agent (météo, recherche web), ou réponse directe du LLM.
- **UI Chainlit** (`app.py`) : interface de chat conversationnelle avec gestion de la mémoire contextuelle entre les échanges.

## Stack technique

| Composant | Technologie |
|---|---|
| LLM | Groq + Llama3-8b (gratuit) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, gratuit) |
| Vector store | FAISS |
| Framework | LangChain + LangGraph |
| UI | Chainlit |
| API météo | OpenMeteo (gratuit, sans clé) |
| Recherche web | DuckDuckGo |

## Corpus

10 rapports scientifiques sur les catastrophes climatiques :

- GIEC AR6 — Synthesis Report Summary for Policymakers
- GIEC AR6 — Working Group II Summary for Policymakers (impacts, adaptation)
- GIEC AR6 — Working Group III Summary for Policymakers (atténuation)
- Copernicus — European State of Climate 2023
- UNDRR — Global Assessment Report on Disaster Risk Reduction 2022
- EM-DAT — Natural Disasters Report 2023
- WMO — State of Global Water Resources 2024
- NOAA — Atlantic Hurricane Season 2023
- JRC — Forest Fires in Europe 2024
- EU — Floods Directive Report (CELEX)

## Installation

### Prérequis

- Python 3.10+
- Compte Groq gratuit : https://console.groq.com

### Étapes

**1. Cloner le repo**
```bash
git clone https://github.com/[username]/catastrophes-climatiques-rag.git
cd catastrophes-climatiques-rag
```

**2. Créer et activer l'environnement virtuel**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Configurer les variables d'environnement**
```bash
cp .env.example .env
# Ouvrir .env et renseigner GROQ_API_KEY
```

**5. Télécharger les PDFs du corpus dans `data/raw/`**

Voir la section Corpus ci-dessus pour la liste complète.

**6. Générer le vector store FAISS**
```bash
python -m src.rag.embeddings
```

**7. Lancer l'application**
```bash
chainlit run app.py
```

## Structure du projet

```
catastrophes-climatiques-rag/
├── data/
│   ├── raw/               # PDFs du corpus scientifique (non versionnés)
│   └── processed/         # Données intermédiaires éventuelles
├── src/
│   ├── rag/
│   │   ├── loader.py      # Chargement et découpage des PDFs
│   │   ├── embeddings.py  # Création et persistance du vector store FAISS
│   │   └── retriever.py   # Récupération MMR avec citations
│   ├── agents/
│   │   ├── tools.py       # Outils : météo, recherche web, calculatrice
│   │   └── agent.py       # Agent LangGraph orchestrant les tools
│   ├── memory/
│   │   └── memory.py      # Mémoire conversationnelle
│   └── router/
│       └── router.py      # Routing conditionnel RAG / Agent / LLM direct
├── notebooks/
│   ├── 01_exploration_corpus.ipynb
│   ├── 02_test_rag.ipynb
│   └── 03_test_agents.ipynb
├── docs/
│   ├── architecture.md    # Schéma et description de l'architecture
│   └── decisions.md       # Justification des choix techniques
├── faiss_store/           # Vector store persisté (non versionné)
├── app.py                 # Point d'entrée Chainlit
├── .env.example           # Template des variables d'environnement
├── requirements.txt       # Dépendances Python
└── README.md
```

## Équipe

| Membre | Responsabilité |
|---|---|
| P1 | Corpus & RAG — `loader.py`, `embeddings.py`, `retriever.py` |
| P2 | Agents & Tools — `tools.py`, `agent.py` |
| P3 | UI & Intégration — `app.py`, `router.py`, `memory.py` |
| P4 | Documentation & Tests — `docs/`, notebooks |
