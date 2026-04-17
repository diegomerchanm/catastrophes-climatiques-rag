# Décisions techniques (ADR)

> **Version 1.1** — Refonte du 17/04/2026, alignée sur le code (Sonnet 4 / Opus 4 / Haiku 4.5, fallback 3 niveaux, 13 outils). Version précédente conservée dans [`archive/decisions_v1.0.md`](archive/decisions_v1.0.md).

> Architecture Decision Records — pour chaque choix structurant : ce qui a été retenu, pourquoi, et quelles alternatives ont été écartées.

---

## ADR-01 — LLM : Anthropic Claude (multi-modèles)

**Décision retenue.** L'orchestration s'appuie sur l'API Anthropic Claude, en utilisant trois modèles selon la nature de la tâche (voir `src/config.py`, `AGENT_CONFIGS`) :

- **Sonnet 4** (`claude-sonnet-4-20250514`) — orchestrateur agent et synthèse RAG
- **Opus 4** (`claude-opus-4-20250514`) — analyste en mode décisionnel (croisement multi-sources)
- **Haiku 4.5** (`claude-haiku-4-5-20251001`) — météo, web, chat direct, classifier de routing

**Raisons.** Claude est performant pour le tool-use natif (pattern ReAct), supporte une fenêtre de contexte étendue nécessaire au RAG sur documents scientifiques, et offre une famille de modèles cohérente (mêmes prompts compatibles entre Opus / Sonnet / Haiku, transparent pour le fallback). L'API est intégrée à LangChain via `langchain-anthropic`.

**Alternatives écartées.** OpenAI (GPT-4) écarté pour rester dans l'écosystème Anthropic du cours. Groq + Llama3-8b avait été envisagé pour le free tier, mais le tool-use est moins fiable et la qualité de raisonnement insuffisante pour les chaînages d'outils complexes (notamment `calculer_score_risque` qui appelle 4 outils en cascade).

---

## ADR-02 — Fallback LLM à 3 niveaux

**Décision retenue.** Bascule automatique en cas d'échec : Sonnet 4 (principal) → Haiku 4.5 → Ollama Mistral local (`src/agents/agent.py:64-78`).

**Raisons.** Une démo de soutenance ne doit pas s'effondrer si l'API Anthropic a un incident, un rate-limit 429 ou un quota épuisé. Les deux premiers niveaux restent dans la famille Claude (transitions transparentes côté prompts). Le troisième niveau, Ollama en local, garantit une continuité même en cas de panne complète d'Internet ou de l'API.

**Coût marginal.** Le fallback ne s'active qu'en cas d'échec, donc le coût normal reste celui de Sonnet 4. La présence d'Ollama ajoute une dépendance optionnelle (Ollama doit tourner en local pour être utilisable), mais sans cela le système se contente du fallback Haiku.

---

## ADR-03 — Embeddings : `paraphrase-multilingual-MiniLM-L12-v2`

**Décision retenue.** Modèle HuggingFace exécuté en local, sans appel API.

**Raisons.** Le modèle supporte 50+ langues, ce qui permet à une question posée en français de retrouver des passages anglais dans le corpus (les rapports GIEC originaux sont en anglais). Il produit des vecteurs de 384 dimensions, offrant un bon compromis entre compacité et qualité. L'exécution locale élimine tout coût d'embedding et garantit la reproductibilité.

**Alternative écartée.** `all-MiniLM-L6-v2` (utilisé en début de projet) est plus léger mais monolingue, ce qui limite le RAG à des questions formulées dans la langue du corpus. Le passage au modèle multilingue a permis le RAG cross-lingue sans modifier la chaîne d'ingestion.

---

## ADR-04 — Vector store : FAISS (versionné après adaptation)

**Décision retenue.** Index FAISS persisté dans `faiss_store/`, désormais versionné dans le repo (5 MB).

**Raisons.** FAISS fonctionne entièrement en mémoire ou sur disque, sans serveur à déployer. **Initialement le store était ignoré** (`.gitignore`), ce qui imposait de régénérer 1 889 chunks à chaque cold start de l'image Docker ou du déploiement HF Spaces — environ 3 minutes de retard à chaque démarrage. Après plusieurs déploiements ratés ou trop lents, la décision pragmatique a été de **versionner les 5 MB** (commit `37cb00f`). Le commentaire en tête de `.gitignore` documente ce choix pour les futurs contributeurs.

**Alternatives écartées.** Chroma, Pinecone, Weaviate — toutes nécessitent un serveur dédié ou une dépendance cloud, ce qui complique le déploiement sans bénéfice tangible à cette échelle de corpus.

---

## ADR-05 — Retriever hybride BM25 + Dense + reranking

**Décision retenue.** `EnsembleRetriever` combinant BM25 (lexical) et FAISS Dense (sémantique) avec pondération 50/50, suivi d'un reranking par CrossEncoder MS-MARCO et d'un placement stratégique des chunks (`src/rag/hybrid_retriever.py`).

**Raisons.** BM25 capture les termes exacts (acronymes GIEC, IPCC, EM-DAT, noms de phénomènes) que les embeddings sémantiques tendent à diluer. FAISS Dense capture les paraphrases et reformulations. Le reranking CrossEncoder réordonne ensuite les top-N par pertinence fine, en lisant les paires (question, chunk) en contexte plein. Le placement stratégique (meilleur en début, deuxième meilleur en fin, moins pertinents au milieu) compense le phénomène **"Lost in the Middle"** observé sur les contextes longs.

**Paramètres MMR retenus.** `k=12`, `fetch_k=40`, `lambda_mult=0.7` (`src/config.py:80-83`) — équilibre pertinence ↔ diversité, calibré pour des questions multi-aspects (par exemple "inondations en Europe : impact, adaptation, prévisions").

---

## ADR-06 — Bridge lexical FR↔EN et diversité forcée

**Décision retenue.** Avant retrieval, un dictionnaire de 20 termes traduit transparentement les mots-clés français en équivalents anglais (GIEC↔IPCC, OMM↔WMO, inondation↔flood, etc.). Après retrieval, un post-filtre limite à 3 chunks maximum par PDF dans le top-12.

**Raisons.** Sans bridge, les questions françaises rappelleraient mal les passages anglais malgré l'embedding multilingue. Sans diversité forcée, un document volumineux comme `Forest_Fires_2024` (~700 chunks) écraserait systématiquement les autres dans le top-12, biaisant la réponse vers un seul thème.

---

## ADR-07 — Framework agent : LangChain + LangGraph

**Décision retenue.** LangChain pour les briques (tools, retrievers, memory) et LangGraph pour le routing à état et la boucle ReAct.

**Raisons.** LangChain est l'écosystème de référence du cours, ce qui facilite la prise en main et la collaboration. LangGraph permet d'implémenter un routing conditionnel propre entre les trois voies (chat / RAG / agent) sous forme de graphe de nœuds, avec gestion d'état explicite. Cette modularité a permis d'étendre l'agent de 3 outils initiaux à 13 sans refondre la boucle ReAct existante.

---

## ADR-08 — Routing à 3 niveaux + LLM classifier de secours

**Décision retenue.** Le router (`src/router/router.py`) applique d'abord des keywords prioritaires (force agent), puis des keywords RAG (force corpus), puis des keywords agent complémentaires. Si aucun ne déclenche, un mini-prompt Claude Haiku catégorise la question en chat / rag / agent.

**Raisons.** Les questions évidentes (la grande majorité du trafic typique) sont routées en quelques millisecondes sans appel LLM. Les cas ambigus tombent dans le filet du classifier LLM, avec un coût négligeable (Haiku ≈ 0,001 $ par classification).

---

## ADR-09 — Outils agent : sandbox AST pour le `calculator`

**Décision retenue.** L'outil `calculator` utilise une whitelist AST (modules `math` autorisés, opérateurs limités) plutôt que `eval()` brut.

**Raisons.** Un agent avec accès à `eval()` est une faille d'injection majeure : un utilisateur malveillant pourrait demander "calcule `__import__('os').system('rm -rf /')`" et l'agent l'exécuterait. La whitelist AST bloque tout ce qui n'est pas une opération arithmétique légitime.

---

## ADR-10 — `calculer_score_risque` : pondération 4 sources

**Décision retenue.** Le score 0-1 combine quatre sources avec des pondérations dépendantes de l'horizon temporel : météo actuelle / prévisions ML / corpus GIEC / historique EM-DAT.

**Raisons.** Aucune source seule ne couvre tous les angles d'un risque climatique : la météo est précise à court terme, le corpus GIEC pose les seuils théoriques, l'historique EM-DAT contextualise les ordres de grandeur, et le ML projette à l'horizon 2030. Pondérer ces sources reflète leur pertinence respective selon qu'on demande un score à 24h, à 7 jours ou à long terme. C'est l'outil qui chaîne le plus d'autres outils en une seule passe — il met à l'épreuve la robustesse de la boucle ReAct LangGraph.

---

## ADR-11 — API météo : OpenMeteo

**Décision retenue.** OpenMeteo pour les trois outils météo (actuel, historique, forecast).

**Raisons.** API open source, gratuite, sans inscription ni clé API, données fiables à partir de coordonnées GPS. L'absence de contrainte d'authentification la rend idéale pour un outil appelé dynamiquement par l'agent (pas de gestion de quotas, pas de rotation de clés).

---

## ADR-12 — Recherche web : Tavily prioritaire + DuckDuckGo fallback

**Décision retenue.** L'outil `web_search` interroge Tavily en premier ; si Tavily échoue ou retourne un résultat vide, bascule sur DuckDuckGo.

**Raisons.** Tavily est conçu pour les agents LLM (résultats déjà résumés et pertinents, mais payant après le free tier de 1 000 requêtes/mois). DuckDuckGo est gratuit et sans clé, mais moins adapté aux agents (résultats bruts à parser). Cette combinaison maximise la qualité tout en garantissant une continuité gratuite si la clé Tavily expire.

---

## ADR-13 — ML prédictif : Quantile Regression + clipping

**Décision retenue.** Gradient Boosting Quantile Regression (perte médiane) sélectionnée par MAE_test, prédictions plafonnées à 1.5× le maximum historique décadal observé.

**Raisons.** Une régression MSE classique est très sensible aux outliers comme la canicule France 2003 (70 000 morts), qui fausse l'apprentissage et fait prédire des catastrophes irréalistes pour les décennies futures. La perte médiane (quantile 0.5) est intrinsèquement robuste. Le clipping ajoute un garde-fou arithmétique pour empêcher toute prédiction extrême liée à un artéfact d'extrapolation.

**Comparatif tracé.** À chaque entraînement, 8 régresseurs et 8 classifieurs sont comparés dans MLflow (16 nested runs). Le choix de Quantile Regression résulte de cette comparaison empirique, traçable dans le Model Registry.

---

## ADR-14 — Upload session-scoped (anti-pollution corpus)

**Décision retenue.** Les PDFs / DOCX uploadés par l'utilisateur sont indexés dans un FAISS **in-memory séparé** par session, et jamais mergés avec le corpus officiel.

**Raisons.** Si les uploads enrichissaient le corpus principal, un utilisateur pourrait polluer les réponses des autres utilisateurs avec des documents non vérifiés. L'isolation par session garantit la fiabilité du corpus scientifique de référence, tout en offrant la flexibilité d'analyser des documents personnels temporairement.

---

## ADR-15 — `schedule_email` : APScheduler + SQLite persistant

**Décision retenue.** L'outil `schedule_email` utilise APScheduler avec backend SQLite (`scheduler_jobs.db`), accompagné d'un outil complémentaire `cancel_scheduled_emails` pour annuler les envois planifiés.

**Raisons.** Un planning d'envoi doit survivre au restart de l'application. APScheduler avec SQLite persiste les jobs entre redémarrages, contrairement à un planning en mémoire qui serait perdu à chaque redéploiement Docker. La paire schedule + cancel donne à l'utilisateur le contrôle complet sur ses envois différés.

---

## ADR-16 — Exposition MCP via FastMCP

**Décision retenue.** Le serveur `mcp_server.py` expose 11 des 13 outils via FastMCP (Model Context Protocol).

**Raisons.** Le MCP est le protocole standard d'Anthropic pour exposer des outils à n'importe quel client LLM (Claude Desktop, Cursor, Continue, Cline). Cette exposition démultiplie la valeur des outils sans duplication de code : les mêmes fonctions `@tool` LangGraph sont réutilisées telles quelles côté MCP.

**Outils non exposés.** `send_bulk_email` et `schedule_email` (avec son complément `cancel_scheduled_emails`) restent internes par principe de moindre privilège. Un client MCP externe pourrait sinon spammer ou planifier des envois persistants à l'insu de l'opérateur humain.

---

## ADR-17 — Observabilité : MLflow SQLite + Model Registry

**Décision retenue.** Backend MLflow SQLite local (`mlflow.db`), un run par requête utilisateur, Model Registry pour les modèles ML.

**Raisons.** SQLite évite la dépendance à un serveur MLflow distant, tout en conservant la richesse de l'API (filtrer / comparer / exporter les runs). Un run par requête permet de profiler les questions coûteuses, comparer A/B les versions de prompt (`prompt_version=v1.0` vs `v2.0`), et détecter les régressions de latence. Le Model Registry versionne les modèles ML (`NB10_regression v1`, `v2`...).

---

## ADR-18 — Build Docker : `--no-cache` systématique

**Décision retenue.** Tous les builds Docker en CI utilisent `--no-cache`.

**Raisons.** Retex d'un projet précédent (loan-default-mlops) où Render réutilisait silencieusement les layers de cache et masquait des régressions critiques (par exemple : dépendance pinnée modifiée mais non réinstallée). `--no-cache` rallonge le build de quelques minutes mais garantit que chaque image est strictement reproductible à partir des sources actuelles.

---

## ADR-19 — Déploiement principal : Hugging Face Spaces

**Décision retenue.** L'application tourne live sur HF Spaces (`xbizot-saearch.hf.space`), avec Docker Hub comme registry intermédiaire et AWS ECR comme registre alternatif. Azure Container Apps est documenté en plan B (workflow `azure.yml` en `workflow_dispatch`).

**Raisons.** HF Spaces est gratuit, suffisamment performant pour la démo (CPU + 16 GB RAM), et s'intègre nativement à un Dockerfile. AWS App Runner et ECS Fargate ont été évalués mais écartés : ils ne sont pas dans le free tier et coûteraient ~30-50 $/mois pour une démo qu'HF Spaces couvre gratuitement. L'image AWS ECR (`310971189093.dkr.ecr.eu-west-3.amazonaws.com/saearch:latest`) reste en place comme **artefact** (preuve de maîtrise de la chaîne ECR), sans service de runtime payant attaché.

---

## ADR-20 — STT local : Faster-Whisper avec autodétection de langue

**Décision retenue.** Transcription vocale en local via Faster-Whisper (modèle `small`, CPU, int8), avec **autodétection de la langue parlée**.

**Raisons.** Aucun coût récurrent (vs API Whisper d'OpenAI), aucune fuite de données vocales vers un tiers, latence acceptable sur CPU pour des messages courts (< 30 secondes). Le modèle `small` offre un bon compromis qualité / vitesse pour le multilingue. L'autodétection évite de demander à l'utilisateur de présélectionner sa langue avant de parler — un Espagnol peut poser sa question sans configuration préalable.

---

## ADR-21 — Boucle Human-in-the-Loop pour le mode décisionnel

**Décision retenue.** En mode décisionnel (4 profils utilisateurs : événementiel, assurance, autorité publique, tourisme), chaque recommandation GO / NO-GO passe par une validation humaine *Approuver / Enrichir / Rejeter* avant d'être finalisée.

**Raisons.** Aucun système ML, aussi sophistiqué soit-il, ne devrait prendre seul une décision aux conséquences réelles (annuler un événement, déclencher une évacuation). La boucle HITL impose un sas humain et logge la décision finale dans MLflow, créant une trace pour auditer la pertinence des recommandations sur la durée.

---

## ADR-22 — Tests : pytest + black + pylint en CI

**Décision retenue.** Pipeline CI strict : `black --check` (formatage), `pylint --disable=R,C` (qualité, exit-zero pour les warnings), `pytest -vv` (43 tests) — tout doit passer pour qu'un commit soit accepté sur `main`.

**Raisons.** Le formatage automatique élimine les débats stylistiques (et les diffs parasites). Pylint en mode permissif (refactoring R et conventions C désactivés) capte les vraies erreurs sans bloquer sur la cosmétique. Les 43 tests pytest couvrent les outils, le calculateur, le géocodage, les prompts, et les régressions identifiées au fil du projet.

---

## ADR-23 — TTS via Web Speech API du navigateur

**Décision retenue.** La lecture vocale des réponses (`public/tts.js` + intégré dans `public/geoloc.js`) utilise la **Web Speech API native du navigateur**, accessible via un bouton flottant 🔊 en bas à gauche de l'UI Chainlit.

**Raisons.** Aucun coût (pas d'API tierce), aucun appel serveur (latence nulle), multilingue selon les voix installées sur le système de l'utilisateur (priorité Google FR, fallback FR-FR, puis toute voix française disponible). Le bouton offre un toggle 🔊 / ⏹️ permettant d'interrompre la lecture en cours. Approche cohérente avec le STT local : on garde tout le pipeline vocal côté client / local, sans dépendance cloud.

**Alternatives écartées.** API ElevenLabs / OpenAI TTS — qualité supérieure mais coût récurrent et fuite de données (la réponse complète est envoyée à un tiers à chaque lecture). Bibliothèques Python serveur (gTTS, pyttsx3) — nécessitent de générer un fichier audio et de le streamer, latence et complexité injustifiées pour une démo.

---

## ADR-24 — Géolocalisation : navigateur natif + IP en fallback

**Décision retenue.** Deux mécanismes complémentaires gérés côté navigateur (`public/geoloc.js`) :
1. **Native** via `navigator.geolocation.getCurrentPosition()` — coordonnées GPS précises si l'utilisateur autorise
2. **IP** via `ip-api.com` au login — position approximative si la géoloc native est refusée ou indisponible

Les deux résultats sont stockés dans `sessionStorage` et réinjectés dans le prompt système de l'agent.

**Raisons.** La géolocalisation native donne une précision au mètre près (utile pour des questions du type "quel temps fait-il près de moi"), mais nécessite une autorisation utilisateur explicite. L'IP-lookup donne au moins une ville par défaut sans interaction. Le double dispositif évite l'écueil "rien ne marche si refus géoloc" tout en exploitant la précision native quand c'est possible.

---

## ADR-25 — Guide de lecture en sortie décisionnelle

**Décision retenue.** Chaque réponse en mode décisionnel (4 profils : événementiel, assurance, autorité publique, tourisme) se termine par une légende structurée des termes utilisés (`Guide de lecture : DECISION = ... | HORIZON = ... | SCORE = ... | ...`).

**Raisons.** Les sorties décisionnelles emploient un vocabulaire dense (DECISION, HORIZON, SCORE, RISQUE, CONFIANCE, URGENCE, CERTITUDE...) qui peut dérouter un utilisateur découvrant le système. La légende intégrée rend la sortie auto-explicative — pas besoin de relire la documentation pour comprendre comment lire le verdict. C'est aussi un atout en soutenance : le jury voit directement le sens de chaque champ sans qu'il faille lui présenter un glossaire séparé.

L'agent emploie par défaut le **vouvoiement** dans toutes ses réponses, en cohérence avec le ton professionnel attendu d'un outil destiné à des décideurs (collectivités, assureurs, autorités).
