# Tests manuels Chainlit - catastrophes-climatiques-rag

Plan de test pour valider les 12 outils, la memoire, le routing, le donut et le multilingue avant la soutenance.

**Environnement** : `chainlit run app.py` sur http://localhost:8000
**Compte test** : `demo@saearch.ai` / `demo` (compte public de demonstration — ne jamais committer de credentials personnels ici)

---

## Grille de resultats

| # | Question | Outil attendu | Outil appele | Donut OK | Sources OK | Temps (s) | Notes |
|---|---|---|---|---|---|---|---|

### Chat / Memoire

| 1 | `Bonjour` | chat simple, pas d'outil | | | n/a | | |
| 2 | `Comment tu t'appelles ?` | identite agent | | | n/a | | |
| 3 | `Je suis P4 / Alice` | memoire : retient le nom | | | n/a | | |
| 4 | `Comment je m'appelle ?` | memoire : recupere le nom | | | n/a | | |

### Meteo (3 outils)

| 5 | `Quel temps fait-il a Paris ?` | `get_weather` | | | n/a | | |
| 6 | `Donne moi la meteo ici` | `get_weather` (geoloc ou fallback) | | | n/a | | |
| 7 | `Meteo historique a Marseille le 15 octobre 2023` | `get_historical_weather` | | | n/a | | |
| 8 | `Previsions pour Lyon` | `get_forecast` | | | n/a | | |
| 9 | `Donne moi la temperature a New York hier a 2 pm` | `get_historical_weather` | | | n/a | | |

### Calcul

| 10 | `Combien font 3+7*2 ?` (attendu : 17) | `calculator` | | | n/a | | |
| 11 | `sqrt(144) + log(1000)/log(10)` (attendu : 15) | `calculator` | | | n/a | | |

### Web

| 12 | `Recherche les catastrophes climatiques recentes` | `web_search` | | | oui | | |
| 13 | `Donne moi les news du jour` | `web_search` | | | oui | | |

### RAG corpus

| 14 | `Que dit le GIEC sur les inondations ?` | `search_corpus` | | | oui (page + source) | | |
| 15 | `Quels sont les seuils critiques de precipitations ?` | `search_corpus` | | | oui | | |
| 16 | `Liste moi les docs du corpus` | `inventaire_corpus` | | | n/a | | |

### Email

| 17 | `Envoie un email a test@test.com avec le sujet Test et le contenu ceci est un test` | `send_email` | | | n/a | | |
| 18 | `Envoie moi un mail pour me rappeler l'anniversaire de ma belle-mere` | `send_email` (memoire email utilisateur) | | | n/a | | |
| 19 | `Envoyer un email a kamila pour avoir une excellente note` | `send_email` (edge case : ton) | | | n/a | | |

### ML predictif (2 nouveaux outils)

| 20 | `Quel est le risque d'inondation au Bangladesh en 2030 ?` | `predict_risque_par_type` (Bangladesh + flood) | | | n/a | | |
| 21 | `Predis l'impact des catastrophes climatiques en France en 2030` | `predict_risque` (tous types) | | | n/a | | |

### Scoring multi-sources

| 22 | `Calcule le score de risque d'inondation a Haiti en 2030 en combinant meteo, ML et corpus` | `calculer_score_risque` (scenario multi-outils) | | | oui | | |

### Multi-outils (orchestration)

| 23 | `Quel temps fait-il a Paris et combien font 2.5+7.3` | `get_weather` + `calculator` | | | n/a | | |
| 24 | `Ajoute 3 quartiles a la meteo d'ici` | `get_weather` + `calculator` sequentiel | | | n/a | | |

### Multilingue

| 25 | `Cual es el clima en Bogota?` | `get_weather` + reponse en espagnol | | | n/a | | |
| 26 | `Was sagt der IPCC uber Uberschwemmungen?` | `search_corpus` (corpus EN, Claude traduit) | | | oui | | |

---

## Checks transverses a chaque reponse (si anomalie, signaler)

| A verifier | Comment |
|---|---|
| **Donut colore correct** | RAG (violet), Meteo (bleu), Web (orange), Calcul (vert), Email (rouge), ML Predict (teal), Scoring (jaune), Agent (gris), Chat (neutre) |
| **Sources citees** | Pour RAG : titre PDF + page |
| **Streaming fluide** | Texte apparait token par token, pas bloc d'un coup |
| **Tokens / cout logges** | Panneau MLflow mis a jour (run cree dans `mlflow.db`) |
| **Memoire conversation** | Questions 3 -> 4 -> 18 (belle-mere) utilisent le contexte precedent |
| **Pas de plantage** | Meme sur tests edge : 19 (ton inapproprie), 17 (email fake) |
| **Fallback LLM** | Deconnecter internet puis tester : bascule Claude -> Ollama |

---

## Tests bonus (nice to have)

| 27 | Upload **PDF** climat + question dessus | ingestion FAISS + `search_corpus` | | | oui | | |
| 28 | Upload **DOC** sujet random + question dessus | ingestion FAISS et/ou reponse WEB | | | | | |
| 29 | Upload **JPEG** sujet random + question dessus | ingestion FAISS et/ou reponse WEB | | | | | |
| 30 | STT audio : enregistrer `quel temps a Paris` au micro | transcription + `get_weather` | | | n/a | | |
| 31 | `Envoie un email a Alice et l'equipe pour Bravo Demo` | `send_bulk_email` | | | n/a | | |

---

## Ordre recommande de la demo soutenance

1. **Auth** (login ecran Chainlit avec compte jury)
2. **Chat + memoire** : tests 1, 3, 4 (impressionne rapidement)
3. **RAG** : test 14 (montre sources + page)
4. **Meteo** : test 5 (rapide, visuel)
5. **ML predictif** : test 20 (Bangladesh inondation) -> le plus fort techniquement
6. **Scoring multi-sources** : test 22 (Haiti) -> montre l'orchestration
7. **Multilingue** : test 25 (espagnol) ou 26 (allemand)
8. **Upload PDF + question** : test 27 (fonctionnalite differenciante)
9. **Si temps** : test 30 (STT audio)

**Duree cible** : 8 min demo + 10 min questions = 18 min total.
