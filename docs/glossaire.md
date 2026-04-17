# Glossaire technique — SAEARCH

> Définitions courtes des termes techniques utilisés dans le projet, pour un lecteur non spécialiste ou pour répondre rapidement à une question pendant la soutenance.

---

## RAG et recherche documentaire

| Terme | Définition courte |
|---|---|
| **RAG** (*Retrieval-Augmented Generation*) | Technique qui consiste à chercher des passages pertinents dans une base documentaire avant de demander à un LLM de répondre, pour qu'il s'appuie sur des sources réelles plutôt que sur sa mémoire d'entraînement. |
| **Retriever** | Composant qui sélectionne les passages les plus pertinents d'une base documentaire pour une question donnée. |
| **Vector store** | Base de données qui stocke des embeddings (vecteurs numériques) et permet de retrouver les plus proches d'une requête. |
| **FAISS** (*Facebook AI Similarity Search*) | Bibliothèque open source qui implémente une recherche par similarité sur des vecteurs, en mémoire ou sur disque. |
| **Embedding** | Vecteur numérique (typiquement 384 ou 768 dimensions) qui représente le sens d'un texte. Deux textes proches sémantiquement ont des embeddings proches. |
| **Chunk** | Petit morceau de document (ici 1500 caractères max). Un PDF est découpé en chunks avant indexation. |
| **MMR** (*Maximal Marginal Relevance*) | Stratégie de retrieval qui équilibre pertinence (similarité à la question) et diversité (dissimilarité entre les résultats). Évite de retourner 5 fois le même contenu. |
| **BM25** | Algorithme classique de recherche par mots-clés (lexical), basé sur la fréquence des termes. Complémentaire des embeddings sémantiques. |
| **Hybrid retrieval** | Combinaison de BM25 (lexical) et embeddings denses (sémantique) pour bénéficier des deux. |
| **EnsembleRetriever** | Composant LangChain qui orchestre plusieurs retrievers et pondère leurs résultats. |
| **Reranking** | Réordonnancement des résultats du retriever via un modèle plus précis (souvent un CrossEncoder), pour améliorer la pertinence du top-K. |
| **CrossEncoder** | Modèle qui prend en entrée la paire (question, document) et retourne un score de pertinence. Plus précis qu'un calcul de similarité d'embeddings, mais plus lent. |
| **LITM** (*Lost in the Middle*) | Phénomène observé chez les LLMs : ils prêtent moins attention aux informations situées au milieu d'un contexte long. Solution : placer les chunks les plus pertinents en début et en fin de contexte. |
| **Bridge lexical FR↔EN** | Petit dictionnaire qui traduit avant retrieval certains mots français vers leurs équivalents anglais (GIEC↔IPCC, OMM↔WMO, inondation↔flood...) pour améliorer la recherche cross-lingue. |
| **Diversité forcée** | Post-filtre qui limite le nombre de chunks venant d'un même document dans le top-K, pour éviter qu'un gros document monopolise la réponse. |

---

## Agents et orchestration

| Terme | Définition courte |
|---|---|
| **Agent** | Système qui utilise un LLM pour décider d'actions à entreprendre (appeler des outils, raisonner) plutôt que de simplement répondre par du texte. |
| **ReAct** (*Reason → Act → Observe → Answer*) | Pattern d'agent où le LLM alterne phases de raisonnement, appels d'outils, observation des résultats, jusqu'à pouvoir répondre. |
| **Tool / Outil** | Fonction Python que l'agent peut appeler (météo, calcul, recherche, email...). Le LLM décide quels outils appeler et avec quels arguments. |
| **Tool calling** | Capacité native d'un LLM à formuler des appels structurés à des outils, avec arguments typés. |
| **LangChain** | Framework Python pour construire des applications LLM (chaînes, retrievers, mémoire, outils). |
| **LangGraph** | Extension de LangChain pour les workflows à état complexe (graphes de nœuds, routing conditionnel, boucles). |
| **Router** | Composant qui décide de la voie de traitement d'une requête (chat / RAG / agent) selon des règles ou un LLM. |
| **Sandbox AST** | Mécanisme de sécurité qui n'autorise que des opérations Python listées explicitement (whitelist), via analyse syntaxique. Empêche l'injection de code malveillant. |
| **HITL** (*Human-in-the-Loop*) | Boucle où une décision automatisée est validée (ou rejetée) par un humain avant d'être finalisée. |
| **Guide de lecture** | Légende structurée placée en fin de réponse décisionnelle, qui explique les termes utilisés (DECISION, HORIZON, SCORE, RISQUE, CONFIANCE). |

---

## LLMs et Anthropic

| Terme | Définition courte |
|---|---|
| **LLM** (*Large Language Model*) | Modèle de langage entraîné sur d'énormes quantités de texte, capable de générer et comprendre du langage naturel. |
| **Anthropic Claude** | Famille de LLMs développée par Anthropic. Trois tailles : Opus (le plus puissant), Sonnet (équilibré), Haiku (rapide et économique). |
| **Sonnet 4** | Modèle Anthropic utilisé comme orchestrateur principal du projet (`claude-sonnet-4-20250514`). |
| **Opus 4** | Modèle Anthropic le plus puissant, utilisé spécifiquement pour l'analyste en mode décisionnel (`claude-opus-4-20250514`). |
| **Haiku 4.5** | Modèle Anthropic rapide et économique, utilisé pour les tâches courtes : météo, web, chat, classifier (`claude-haiku-4-5-20251001`). |
| **Token** | Unité de comptage du LLM (~4 caractères en français). La facturation et les limites de contexte se mesurent en tokens. |
| **Fenêtre de contexte** | Quantité maximale de tokens que le LLM peut traiter en une seule requête. |
| **Fallback LLM** | Bascule automatique vers un autre modèle en cas d'échec du modèle principal. Ici 3 niveaux : Sonnet 4 → Haiku 4.5 → Ollama Mistral local. |
| **Ollama** | Outil qui permet de faire tourner des LLMs open source en local (Mistral, Llama, etc.). Utilisé ici comme dernier filet de sécurité hors ligne. |
| **Prompt** | Texte envoyé au LLM pour orienter sa réponse (instructions de rôle, format, contraintes). |
| **Versioning de prompts** | Stockage de plusieurs versions du même prompt (v1.0, v2.0...) pour A/B testing et traçabilité. |
| **MCP** (*Model Context Protocol*) | Protocole standard d'Anthropic pour exposer des outils à n'importe quel client LLM (Claude Desktop, Cursor, Continue, etc.). |
| **FastMCP** | Bibliothèque Python qui simplifie la création d'un serveur MCP. |

---

## ML prédictif

| Terme | Définition courte |
|---|---|
| **EM-DAT** | Base de données mondiale des catastrophes naturelles depuis 1900 (université de Louvain). Source du dataset d'entraînement ML du projet. |
| **Quantile Regression** | Variante de la régression qui prédit la médiane (ou un autre quantile) au lieu de la moyenne. Plus robuste aux valeurs extrêmes (outliers). |
| **Gradient Boosting** | Famille d'algorithmes ML qui combinent itérativement des arbres de décision, en se concentrant à chaque étape sur les erreurs précédentes. |
| **Outlier** | Valeur extrême qui s'écarte fortement du reste des données (ex : la canicule France 2003 dans EM-DAT). |
| **Clipping** | Plafonnage arithmétique des prédictions (ici 1.5× le max historique) pour éviter qu'une extrapolation aberrante donne des chiffres irréalistes. |
| **MAE** (*Mean Absolute Error*) | Métrique de régression : moyenne des écarts absolus entre prédiction et réalité. |
| **Nested runs** | Structure MLflow où un run parent contient plusieurs runs enfants (ici : 1 entraînement = 1 parent + 16 enfants pour la grille comparative). |

---

## MLOps / LLMOps / Observabilité

| Terme | Définition courte |
|---|---|
| **MLOps** | Pratiques d'ingénierie pour industrialiser le cycle de vie des modèles ML (versionning, monitoring, déploiement, tests). |
| **LLMOps** | Adaptation des MLOps aux LLMs : tracking de tokens, coûts USD, latences, prompts versionnés, fallback. |
| **MLflow** | Outil open source de tracking d'expériences ML : logs de paramètres, métriques, artefacts, et registry de modèles versionnés. |
| **Model Registry** | Composant MLflow qui versionne les modèles entraînés et permet de les promouvoir (Staging → Production). |
| **Token tracking** | Comptage en temps réel des tokens consommés par chaque requête, avec estimation du coût USD. |
| **A/B testing de prompts** | Comparaison contrôlée de deux versions d'un prompt sur les mêmes questions, pour mesurer laquelle est meilleure. |

---

## UI et multimodal

| Terme | Définition courte |
|---|---|
| **Chainlit** | Framework Python pour créer rapidement des interfaces conversationnelles avec streaming, auth, multimodal. |
| **Streaming** | Affichage progressif de la réponse au fur et à mesure qu'elle est générée, plutôt que d'attendre la fin. |
| **Multimodal** | Capacité à traiter plusieurs types de données (texte, image, audio) dans la même conversation. |
| **Claude Vision** | Capacité native des modèles Claude récents à analyser des images transmises en base64. |
| **STT** (*Speech-to-Text*) | Conversion de la voix en texte. Ici : Faster-Whisper en local, autodétection de langue. |
| **Faster-Whisper** | Implémentation optimisée du modèle Whisper d'OpenAI, exécutable en local sur CPU. |
| **TTS** (*Text-to-Speech*) | Conversion de texte en voix. Ici : Web Speech API du navigateur (gratuit, pas de serveur). |
| **Web Speech API** | API standard des navigateurs modernes pour la synthèse vocale et la reconnaissance vocale, gratuite et instantanée. |
| **PCM16** | Format audio non compressé 16 bits, utilisé pour la capture micro côté navigateur. |
| **Session-scoped** | Mécanisme qui isole les uploads par session utilisateur, évitant la pollution du corpus partagé. |

---

## Infra et DevOps

| Terme | Définition courte |
|---|---|
| **Docker** | Outil de conteneurisation : empaqueter une application avec toutes ses dépendances dans une image reproductible. |
| **Dockerfile** | Recette de construction d'une image Docker. |
| **`--no-cache`** | Option Docker qui force la reconstruction complète de l'image sans réutiliser les couches précédentes. Garantit la reproductibilité. |
| **CI/CD** (*Continuous Integration / Continuous Deployment*) | Automatisation des tests et du déploiement à chaque commit. |
| **GitHub Actions** | Service d'automatisation CI/CD intégré à GitHub. |
| **Hugging Face Spaces** | Hébergement gratuit d'applications IA, supporte Docker. Notre déploiement principal. |
| **Docker Hub** | Registry public d'images Docker. Notre image y est poussée à chaque commit. |
| **AWS ECR** (*Elastic Container Registry*) | Registry d'images Docker AWS. Notre image y est aussi présente comme artefact. |
| **APScheduler** | Bibliothèque Python pour planifier l'exécution de tâches (jobs cron). Persiste les jobs dans SQLite. |
| **Cron** | Expression de planification temporelle (ex : "tous les lundis à 8h"). |

---

## Organisations et sources

| Acronyme | Signification |
|---|---|
| **GIEC** | Groupe d'experts intergouvernemental sur l'évolution du climat (en anglais : IPCC). |
| **IPCC** | *Intergovernmental Panel on Climate Change* — équivalent anglais de GIEC. |
| **AR6** | *Sixth Assessment Report* — sixième rapport d'évaluation du GIEC (2021-2023). |
| **WG II / WG III** | *Working Group II / III* du GIEC : impacts/adaptation, atténuation. |
| **Copernicus** | Programme européen d'observation de la Terre (climat, atmosphère, mer, surface...). |
| **NOAA** | *National Oceanic and Atmospheric Administration* — agence US météo et océanographique. |
| **WMO** | *World Meteorological Organization* — Organisation météorologique mondiale (OMM en français). |
| **JRC** | *Joint Research Centre* — Centre commun de recherche de la Commission européenne. |
| **UNDRR** | *United Nations Office for Disaster Risk Reduction* — Bureau ONU pour la réduction des risques de catastrophe. |
| **CELEX** | Identifiant des documents juridiques européens (ex : la Floods Directive). |
