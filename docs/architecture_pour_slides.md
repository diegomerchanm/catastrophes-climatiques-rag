# Architecture SAEARCH — Vue compactes pour slides

## Diagramme simplifie 1 slide (a inserer)

```
                          UTILISATEUR (FR/EN/ES/DE)
                                   |
                                   v
                  +----------------------------------+
                  |  CHAINLIT UI                     |
                  |  - Auth / Donut / STT / Upload   |
                  |  - Memoire conversationnelle     |
                  +----------------------------------+
                                   |
                                   v
                  +----------------------------------+
                  |  ROUTER LangGraph (3 voies)      |
                  |  Keywords + LLM classifier       |
                  +-------+----------+---------------+
                          |          |          |
                       chat        rag       agent
                          |          |          |
                          v          v          v
                       LLM     +---------+   +-----------------------+
                               |  RAG    |   |  AGENT ReAct          |
                               | hybride |   |  13 outils orchestres |
                               | BM25+   |   |  - Meteo x3 / web     |
                               | Dense + |   |  - calculator / RAG   |
                               | rerank  |   |  - email x3 / predict |
                               +----+----+   |  - score / inventory  |
                                    |        +-----------+-----------+
                                    |                    |
                                    v                    v
                  +-----------------------------------------------+
                  |  SOURCES                                       |
                  |  FAISS 1889 chunks (10 PDFs GIEC)              |
                  |  + OpenMeteo + Tavily + Gmail + ML joblib NB10 |
                  +-----------------------------------------------+
                                    |
                                    v
                  +-----------------------------------------------+
                  |  LLM Anthropic Claude (fallback triple)        |
                  |  Opus 4.6 -> Sonnet 4.5 -> Haiku 4.5 -> Ollama |
                  +-----------------------------------------------+
                                    |
                                    v
                  +-----------------------------------------------+
                  |  MLflow (SQLite) - tracking + Model Registry   |
                  |  CI/CD GitHub Actions - Docker Hub + Azure     |
                  |  MCP Server - 11 outils Claude Desktop         |
                  +-----------------------------------------------+
```

## Architecture en 3 phrases

1. **Frontend** : Chainlit (Python) avec auth, mémoire conversationnelle, upload de documents session-scoped, STT audio et donut chart visuel des outils appelés.

2. **Cerveau** : Router LangGraph qui dispatche en 3 voies (chat / rag / agent). L'agent ReAct orchestre 13 outils (météo, web, calcul, RAG, email, ML prédictif, scoring) avec un LLM Claude (fallback triple).

3. **Backend** : RAG hybride BM25+Dense sur 10 PDFs GIEC (multilingue), modèles ML quantile sur EM-DAT, MLflow pour le tracking, CI/CD GitHub Actions vers Docker Hub + Azure.

## Chiffres cles

- **13 outils** disponibles (vs 9 prevu initialement)
- **1889 chunks** indexes dans FAISS sur **10 PDFs GIEC/IPCC**
- **3 LLM fallback** : Opus -> Sonnet -> Haiku -> Ollama local
- **16 modeles ML** compares dans MLflow (Quantile Regression vainqueur)
- **2 versions de prompts** A/B testable (v1.0 active, v2.0 ready)
- **3 workflows CI/CD** : Docker Hub auto + Azure manual + cron weekly
- **11 outils MCP** exposes pour Claude Desktop

## Innovations techniques

| Innovation | Impact |
|---|---|
| Embedding multilingue | Questions FR/DE/ES sur corpus EN |
| Bridge lexical FR-EN | GIEC <-> IPCC, OMM <-> WMO (20 termes) |
| Diversite forcee 3/PDF | Eviter qu'un gros doc ecrase les autres |
| Quantile Regression | Robustesse aux outliers (canicule 2003) |
| Clipping predictions | Plafond 1.5x max historique decadal |
| Upload session-scoped | Empeche pollution corpus officiel |
| schedule_email persistant | APScheduler + SQLite, jobs survivent au restart |
| Fallback triple LLM | Resilience cloud (429, panne API) |
| Tags MLflow experiment_type | Tracabilite A/B avant/apres pivot quantile |
