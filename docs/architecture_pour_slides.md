# Architecture SAEARCH — vue condensée pour slides

> **Version 1.1** — Refonte du 17/04/2026, alignée sur le code. Version précédente conservée dans [`archive/architecture_pour_slides_v1.0.md`](archive/architecture_pour_slides_v1.0.md).

> Extraits destinés à la soutenance : diagramme simplifié, résumé 3 phrases, chiffres clés, innovations.

---

## Diagramme simplifié (1 slide)

```
                          UTILISATEUR (FR / EN / ES / DE)
                        texte / voix (STT) / PDF / DOCX / image
                                        |
                                        v
                  +-------------------------------------------+
                  |  CHAINLIT UI                              |
                  |  Auth / Donut / Streaming / Multimodal    |
                  |  STT autodetect + TTS bouton flottant     |
                  |  Geoloc native + IP fallback              |
                  |  Memoire conversationnelle 20 tours        |
                  +-------------------------------------------+
                                        |
                                        v
                  +-------------------------------------------+
                  |  ROUTER LangGraph (3 voies)               |
                  |  Keywords prio > RAG > Agent > LLM classif|
                  +---------+----------+----------+-----------+
                            |          |          |
                          chat       rag       agent
                            |          |          |
                            v          v          v
                          LLM     +---------+  +-----------------------+
                                  |  RAG    |  |  AGENT ReAct LangGraph|
                                  | hybride |  |  13 outils orchestres |
                                  | BM25 +  |  |  Meteo x3 / web       |
                                  | Dense + |  |  calculator / RAG     |
                                  | rerank  |  |  email x3 / ML x2     |
                                  +----+----+  |  scoring / inventaire |
                                       |       +-----------+-----------+
                                       |                   |
                                       v                   v
                  +-----------------------------------------------+
                  |  SOURCES                                      |
                  |  FAISS 1889 chunks (10 PDFs GIEC)             |
                  |  + OpenMeteo + Tavily + Gmail + ML joblib NB10|
                  +-----------------------------------------------+
                                        |
                                        v
                  +-----------------------------------------------+
                  |  LLM Anthropic Claude (fallback 3 niveaux)     |
                  |  Sonnet 4 -> Haiku 4.5 -> Ollama Mistral local |
                  +-----------------------------------------------+
                                        |
                                        v
                  +-----------------------------------------------+
                  |  HITL (mode decisionnel, 4 profils)            |
                  |  Approuver / Enrichir / Rejeter + log MLflow  |
                  +-----------------------------------------------+
                                        |
                                        v
                  +-----------------------------------------------+
                  |  MLflow (SQLite) — tracking + Model Registry   |
                  |  CI/CD GitHub Actions — Docker Hub + HF Spaces |
                  |  MCP Server FastMCP — 11 outils Claude Desktop |
                  +-----------------------------------------------+
```

---

## Architecture en 3 phrases

1. **Frontend** — Chainlit (Python) avec authentification, mémoire conversationnelle, upload session-scoped de documents, multimodal complet (entrée image via Claude vision, audio via Faster-Whisper avec autodétection de langue, sortie vocale via TTS bouton flottant), géolocalisation native + IP, et donut SVG qui visualise les outils appelés.

2. **Cerveau** — Router LangGraph qui dispatche en trois voies (chat / rag / agent). L'agent ReAct orchestre **13 outils** (météo, web, calcul, RAG hybride, email, ML prédictif, scoring multi-sources) avec Claude Sonnet 4 en principal et un fallback à 3 niveaux vers Haiku puis Ollama local.

3. **Backend** — RAG hybride BM25 + Dense sur **10 rapports scientifiques** (GIEC, Copernicus, EM-DAT, NOAA, JRC, WMO) avec embedding multilingue, modèles ML Quantile Regression sur EM-DAT décadal, MLflow pour le tracking, CI/CD GitHub Actions vers Docker Hub + HF Spaces, et exposition MCP pour intégration dans Claude Desktop.

---

## Chiffres clés

- **13 outils** disponibles via l'agent (11 exposés également en MCP)
- **1 889 chunks** indexés dans FAISS sur **10 PDFs GIEC / IPCC / Copernicus / EM-DAT / NOAA / JRC / WMO**
- **3 niveaux de fallback LLM** : Sonnet 4 → Haiku 4.5 → Ollama Mistral local
- **16 modèles ML** comparés dans MLflow (Quantile Regression retenue pour robustesse outliers)
- **2 versions de prompts** A/B testables (v1.0 active, v2.0 prête)
- **25 langues** traduites nativement dans l'UI Chainlit
- **43 tests pytest** exécutés à chaque commit en CI
- **4 workflows GitHub Actions** : CI/CD Docker + Azure (plan B) + cron weekly + cron teaser

---

## Innovations techniques majeures

| Innovation | Impact |
|---|---|
| RAG cross-lingue (embedding multilingue + bridge FR-EN) | Questions FR/DE/ES sur un corpus à dominante anglaise |
| Diversité forcée 3 chunks/PDF | Évite qu'un gros document écrase les autres |
| Reranking + placement stratégique | Atténue le Lost in the Middle sur contextes longs |
| Scoring multi-sources (4 outils pondérés) | Combine météo + ML + GIEC + historique en une passe |
| Quantile Regression + clipping | Robustesse aux outliers (canicule France 2003) |
| Fallback LLM 3 niveaux | Résilience 429 / panne API / quota |
| Upload session-scoped | Empêche la pollution du corpus officiel |
| schedule_email + cancel persistants | Jobs APScheduler + SQLite survivent au restart |
| Exposition MCP sans duplication de code | Mêmes @tool partagés entre Chainlit et FastMCP |
| HITL 4 profils décisionnels | Sas humain Approuver / Enrichir / Rejeter + log MLflow |
| Guide de lecture en sortie décisionnelle | Légende auto-explicative DECISION / HORIZON / SCORE / RISQUE |
| STT PCM16 + Faster-Whisper local + autodétection langue | Vocal multilingue sans fuite de données ni coût API |
| TTS bouton flottant Web Speech API | Lecture vocale réponses, sans coût ni serveur |
| Multimodal Claude vision (base64) | Analyse d'images sans OCR séparé |
| Géolocalisation native + IP fallback | Position GPS précise si autorisée, IP sinon |
| Token tracking live + estimation USD | Affichage temps réel dans l'UI Chainlit |
| Build Docker --no-cache + faiss_store versionné | Reproductibilité stricte + cold start instantané |
