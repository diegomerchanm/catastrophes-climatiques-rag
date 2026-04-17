# FAQ technique — questions probables du jury et réponses préparées

> 18 questions susceptibles d'être posées en soutenance, avec des éléments de réponse synthétiques. À adapter à ton oral.

---

## Choix techniques

### 1. Pourquoi Claude (Anthropic) plutôt que GPT (OpenAI), Gemini ou Mistral ?

Trois raisons principales :

1. **Tool-use natif fiable** : Claude a un mode tool-use bien intégré et stable, essentiel pour un agent qui doit chaîner 13 outils (notamment `calculer_score_risque` qui en chaîne 4 d'affilée).
2. **Famille de modèles cohérente** : Opus / Sonnet / Haiku partagent les mêmes prompts, ce qui rend le fallback transparent. On peut basculer Sonnet → Haiku sans réécrire les prompts.
3. **Alignement avec l'écosystème du cours** et choix d'équipe.

GPT-4 aurait techniquement marché. Gemini est moins mature côté tool-use. Mistral via Groq a été testé en début de projet mais le tool-use était moins fiable pour les chaînages complexes.

---

### 2. Pourquoi 3 modèles différents (Opus, Sonnet, Haiku) ?

Spécialisation par rôle pour optimiser le rapport qualité / coût :

- **Sonnet 4** comme orchestrateur principal : équilibre raisonnement / coût pour la grande majorité des requêtes
- **Opus 4** uniquement pour l'analyste en mode décisionnel : on paie le tarif premium ($15/M tokens) seulement quand on a besoin du raisonnement maximal pour croiser 4 sources
- **Haiku 4.5** pour les tâches courtes (météo, web, classifier router) : 60× moins cher que Opus, suffisant pour synthétiser un JSON OpenMeteo

Cela évite de payer le prix d'Opus pour répondre à "bonjour".

---

### 3. Comment vous évitez les hallucinations du LLM ?

Trois mécanismes :

1. **RAG avec citations obligatoires** : pour toute question scientifique, l'agent appelle `search_corpus` et la réponse cite `[Source: nom_fichier, Page: N]`. L'utilisateur peut vérifier.
2. **Prompt système strict** : "cite toujours tes sources avec [Source: ...]", "ne fabrique JAMAIS d'adresse email", "réponds dans la même langue que la question".
3. **Outils factuels pour les données chiffrées** : météo via OpenMeteo, calculs via `calculator` sandboxé, prédictions ML via les joblibs entraînés. Le LLM ne génère pas des chiffres de tête.

---

### 4. Et si l'API Anthropic tombe pendant la soutenance ?

Chaîne de fallback à 3 niveaux automatique :

1. **Sonnet 4** (principal)
2. → **Haiku 4.5** (Anthropic, économique)
3. → **Ollama Mistral local** (hors ligne)

Le dernier niveau tourne en local sur la machine, sans Internet. Donc même si Anthropic et le réseau tombent, on peut continuer à démontrer le système.

---

### 5. Combien ça coûte par requête en moyenne ?

Selon le type :

- Chat simple (Haiku) : ~$0.001 par requête
- RAG avec synthèse (Sonnet) : ~$0.01 - $0.03 par requête
- Score multi-sources avec Opus + 4 outils : ~$0.30 - $0.50 par requête

Le `TokenCounter` affiche le coût en temps réel dans Chainlit, et chaque requête est loggée dans MLflow pour audit. C'est l'observabilité LLMOps du projet.

---

## RAG et qualité

### 6. Pourquoi un retrieval hybride BM25 + Dense plutôt que juste embeddings ?

- **BM25** capture les termes exacts (acronymes GIEC, IPCC, EM-DAT) que les embeddings sémantiques tendent à diluer
- **FAISS Dense** capture les paraphrases et reformulations sémantiques

Sans BM25, une question contenant "OMM" pourrait passer à côté de tous les chunks qui parlent de "World Meteorological Organization". L'EnsembleRetriever pondère 50/50 et le CrossEncoder réordonne ensuite par pertinence fine.

---

### 7. Comment vous gérez les questions en plusieurs langues ?

Deux mécanismes complémentaires :

1. **Embedding multilingue** (`paraphrase-multilingual-MiniLM-L12-v2`) : 50+ langues partagent le même espace vectoriel, donc une question en français peut retrouver des passages en anglais.
2. **Bridge lexical FR↔EN** : un dictionnaire de 20 termes (GIEC↔IPCC, OMM↔WMO, inondation↔flood...) reformule transparente pour booster le BM25 cross-lingue.

Le prompt système impose ensuite que la réponse soit dans la même langue que la question.

---

### 8. Comment évaluez-vous la qualité du RAG ?

Notebook `NB6_Comparatifs_MLflow` documente les expérimentations :

- Comparaison de pondérations BM25 / Dense (30/70, 50/50, 70/30)
- Comparaison de tailles de chunks
- Comparaison de températures LLM
- A/B testing prompt v1.0 vs v2.0

Tous les résultats sont tracés dans MLflow avec le tag `experiment_type`. Sur des questions de référence, on constate qu'une question type GIEC retrouve les bons passages dans 80%+ des cas.

---

### 9. Pourquoi 1 889 chunks et pas plus / moins ?

C'est le résultat du chunking automatique des 10 PDFs avec `chunk_size=1500` et `overlap=150`. Ces paramètres ont été choisis pour :

- Chunks assez longs pour conserver le contexte d'un paragraphe
- Assez courts pour rester précis dans le retrieval
- Overlap pour ne pas couper une phrase importante en deux

Les 1 889 chunks sont indexés en FAISS et **versionnés dans le repo** (5 MB), pour que le déploiement Docker / HF Spaces démarre instantanément sans devoir re-générer.

---

## ML prédictif

### 10. Le modèle ML est entraîné sur des données EM-DAT jusqu'en 2020. Comment ça reste pertinent en 2026 ?

C'est une bonne limite à reconnaître. Trois éléments de réponse :

1. **Données décadales** : on prédit par décennie, pas par année. La décennie 2020-2030 est ce qu'on extrapole, c'est l'objectif explicite.
2. **Quantile Regression** : le modèle prédit la médiane robuste, pas une tendance fragile à un événement extrême récent.
3. **Réentraînement automatique** : `scripts/train_nb10.py` est lancé à chaque commit en CI/CD, donc dès qu'on intègre des données plus récentes, les joblibs sont régénérés.

Pour aller plus loin : intégrer un connecteur EM-DAT live (pas fait dans le périmètre du projet, axe d'amélioration).

---

### 11. Pourquoi Quantile Regression et pas une régression classique (MSE) ?

Robustesse aux outliers. Le dataset EM-DAT contient des événements extrêmes (canicule France 2003 = 70 000 morts) qui faussent une régression MSE classique : le modèle apprendrait à prédire ces extrêmes systématiquement, donnant des chiffres irréalistes.

La perte médiane (quantile 0.5) est intrinsèquement robuste : un seul outlier ne déplace pas la médiane.

Le clipping à 1.5× le max historique décadal ajoute un garde-fou arithmétique en complément.

---

### 12. 16 modèles comparés dans MLflow — pourquoi pas plus ?

8 régresseurs et 8 classifieurs couvrent les grandes familles : linéaires, arbres, gradient boosting (sklearn et XGBoost), kernel-based. Au-delà, le coût de calcul augmente sans gain significatif sur ce dataset.

La comparaison est visible dans le Model Registry MLflow et chaque modèle entraîné est versionné (`NB10_regression v1`, `v2`, ...).

---

## Sécurité et coût

### 13. Qu'est-ce qui empêche un utilisateur de spammer / abuser le système ?

Plusieurs garde-fous :

- **Authentification obligatoire** (Chainlit auth) : seuls les comptes pré-configurés peuvent accéder
- **Calculator sandboxé** (whitelist AST) : impossible d'exécuter du code arbitraire via `__import__('os').system(...)`
- **Email** : `send_bulk_email` et `schedule_email` ne sont **pas exposés** via MCP (principe de moindre privilège, anti-spam)
- **Upload session-scoped** : un utilisateur ne peut pas polluer le corpus officiel ni les sessions des autres
- **Token tracking** : chaque requête loggée dans MLflow, on peut détecter un usage anormal

---

### 14. Quel est le throughput / la latence ?

- Chat simple (Haiku) : ~1-2 secondes
- RAG avec retrieval + synthèse (Sonnet) : ~3-5 secondes
- Multi-outils (4 outils chaînés + scoring) : ~15-20 secondes

Latence dominée par les appels LLM. Le retrieval FAISS est <100 ms. Streaming token par token côté UI pour réduire le ressenti d'attente.

---

### 15. Vie privée des utilisateurs ?

- **STT en local** (Faster-Whisper) : la voix ne part jamais sur un serveur tiers
- **TTS en local** (Web Speech API du navigateur) : idem
- **Uploads PDF/DOCX** : indexés en mémoire session-scoped, jamais persistés en disque côté serveur
- **Embeddings en local** (HuggingFace) : aucun appel API pour l'indexation
- **Données envoyées à Anthropic** : seulement le prompt + contexte RAG nécessaire à la requête, pas l'intégralité de l'historique utilisateur

---

## Architecture / DevOps

### 16. Pourquoi `docker build --no-cache` systématique ?

Retex d'un projet précédent (`loan-default-mlops`) où Render réutilisait silencieusement les layers de cache et masquait des régressions critiques (par exemple : version pinnée d'une dépendance modifiée mais non réinstallée).

`--no-cache` ajoute ~3-5 minutes au build mais garantit que l'image est strictement reproductible à partir des sources actuelles. C'est documenté dans `ADR-18`.

---

### 17. Pourquoi Hugging Face Spaces et pas AWS / Azure / GCP en prod ?

HF Spaces est :
- **Gratuit** (CPU + 16 GB RAM)
- **Suffisant** pour la démo (latence acceptable, pas de pic de trafic prévu)
- **Intégré nativement à Docker**
- **Versionné via git** comme un repo classique

AWS App Runner et ECS Fargate ont été évalués mais coûteraient ~30-50 $/mois pour une démo qu'HF couvre gratuitement. L'image AWS ECR reste en place comme **artefact** (preuve de maîtrise de la chaîne ECR), sans service de runtime payant attaché. Azure Container Apps a un workflow prêt en plan B (`azure.yml` en `workflow_dispatch`).

---

### 18. Code propriétaire ou open source ? Comment quelqu'un peut le réutiliser ?

Le code projet est **MIT** (sauf mention contraire). Quelqu'un peut :

- `git clone` le repo public GitHub
- `docker pull xbizot/rag-catastrophes:latest` pour récupérer l'image directement
- Suivre le README pour installer en local : `python -m venv venv` + `pip install -r requirements.txt` + lancement Chainlit

Le corpus PDF reste propriété des organisations émettrices (GIEC, Copernicus, etc.) — usage académique uniquement, à télécharger séparément depuis les sources officielles ou le Google Drive lié dans le README.
