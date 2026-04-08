# Décisions Techniques

## Choix du LLM : Groq + Llama3-8b

Groq offre une API gratuite sans carte de crédit requise, avec des temps d'inférence très rapides grâce à son hardware LPU. Le modèle Llama3-8b est suffisamment performant pour la compréhension de documents scientifiques et la génération de réponses structurées. Il est nativement compatible avec LangChain via `langchain-groq`.

L'alternative OpenAI (GPT-3.5 / GPT-4) a été écartée car elle nécessite une carte de crédit et n'est pas accessible à tous les membres de l'équipe, ce qui compromettrait la reproductibilité du projet.

## Choix des embeddings : all-MiniLM-L6-v2

Ce modèle HuggingFace s'exécute entièrement en local, sans appel API et donc sans coût. Il produit des vecteurs de 384 dimensions, offrant un bon compromis entre compacité et qualité de représentation sémantique. Il est performant pour la recherche sémantique sur des textes techniques et supporte plusieurs langues, ce qui est pertinent pour un corpus mixte français/anglais.

Aucune clé API n'est nécessaire, ce qui simplifie la configuration et garantit la reproductibilité.

## Choix du vector store : FAISS

FAISS (Facebook AI Similarity Search) est une bibliothèque légère qui fonctionne entièrement en mémoire ou sur disque, sans serveur à déployer. Le vector store est persisté localement dans le dossier `faiss_store/`, ce qui évite de recalculer les embeddings à chaque lancement.

C'est la solution utilisée dans les notebooks du cours, ce qui facilite l'alignement avec les ressources pédagogiques et la prise en main par toute l'équipe.

## Choix du retriever : MMR (Maximum Marginal Relevance)

La recherche par similarité cosinus simple tend à retourner des documents très similaires entre eux, ce qui appauvrit le contexte fourni au LLM. Le MMR équilibre pertinence (similarité à la question) et diversité (dissimilarité entre les résultats), ce qui améliore la qualité des réponses sur des questions couvrant plusieurs aspects d'un phénomène climatique.

Paramètres retenus :
- `k=4` : nombre de documents finalement retournés au LLM
- `fetch_k=10` : pool de candidats évalués avant sélection MMR

## Choix de l'API météo : OpenMeteo

OpenMeteo est une API météo open source, gratuite, sans inscription et sans clé API. Elle fournit des données fiables à partir de coordonnées GPS (latitude/longitude) pour des données actuelles et historiques. Son absence de contraintes d'authentification la rend idéale pour un outil appelé dynamiquement par un agent LangGraph.

## Choix du framework : LangChain + LangGraph

LangChain est le framework standard du cours, ce qui facilite la prise en main et l'alignement avec les ressources pédagogiques. LangGraph, son extension pour les workflows à état, permet d'implémenter un routing conditionnel propre entre les trois chemins de traitement (RAG, agent, réponse directe) sous forme de graphe de nœuds.

Cette combinaison offre une architecture modulaire où chaque composant (retriever, tools, mémoire) peut être développé et testé indépendamment.

## Usage de l'IA

Une partie du code de ce projet a été généré avec l'assistance de Claude (Anthropic). Chaque fichier généré a été relu, compris et validé par les membres de l'équipe concernés. Les imports dépréciés (`langchain.text_splitter`, `langchain_community.embeddings`) ont été identifiés et corrigés manuellement lors de la relecture. Les décisions d'architecture (choix du stack, découpage en modules, stratégie de routing) ont été prises par l'équipe avant la génération du code.
