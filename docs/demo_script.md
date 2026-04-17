# Script de démo soutenance — SAEARCH / DooMax

> Scénario pas-à-pas pour la soutenance, avec les questions exactes à taper, les outils attendus en réponse, et les points à commenter à voix haute. Durée totale visée : ~15-20 min.

---

## Préparation (avant de commencer)

- [ ] HF Spaces ouvert dans un onglet : `https://xbizot-saearch.hf.space`
- [ ] Connecté avec un compte (`demo@saearch.ai` / `demo` ou ton compte perso)
- [ ] `docs/architecture_pour_slides.md` ouvert dans un autre onglet ou en PDF
- [ ] `SAEARCH_architecture.html` ouvert en parallèle (double-clic local) si tu veux montrer le diagramme
- [ ] Dashboard MLflow lancé en parallèle si possible : `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
- [ ] Volume audio activé pour le STT/TTS
- [ ] Une question vocale préparée mentalement

---

## Plan de démo (10 séquences)

### 1. Présentation rapide de l'UI (1 min)
**À montrer** : page d'accueil après login.

**À commenter** :
- Sidebar à gauche : multi-conversations, persistées en SQLite
- Donut central qui visualise les outils appelés (vide pour l'instant)
- Footer noir : tokens, coût USD, version du prompt
- Géoloc auto-détectée (ex : "Houplines, Hauts-de-France")

---

### 2. Chat simple en français (1 min)
**À taper** :
```
Bonjour, qui es-tu ?
```

**Réponse attendue** : DooMax se présente (jamais "Claude"), explique son rôle, propose ses capacités. Aucun outil appelé (donut reste vide).

**À commenter** :
- Le prompt système l'oblige à dire "DooMax", jamais "Claude"
- Vouvoiement par défaut
- Le routeur a tranché "chat" en quelques millisecondes (pas d'appel LLM de classification, juste keywords)

---

### 3. RAG sur le corpus GIEC (2 min)
**À taper** :
```
Que dit le GIEC sur l'intensification des inondations en Europe méditerranéenne ?
```

**Réponse attendue** :
- L'agent appelle `search_corpus`
- Réponse synthétique avec citations `[Source: IPCC_AR6_WGII_SummaryForPolicymakers.pdf, Page: X]`
- Donut affiche le segment "RAG"

**À commenter** :
- Pipeline hybride BM25 + Dense + reranking CrossEncoder
- Bridge lexical FR↔EN : la question en français a retrouvé des passages dans des PDFs en anglais
- Diversité forcée : maximum 3 chunks par PDF, pour éviter qu'un seul document monopolise la réponse

---

### 4. Question multilingue (1 min, optionnel)
**À taper** (en anglais) :
```
What does the IPCC say about heat waves in Europe?
```

**Ou en espagnol** :
```
¿Qué dice el IPCC sobre las olas de calor en Europa?
```

**Réponse attendue** : Réponse intégralement dans la langue de la question, citations toujours présentes.

**À commenter** :
- Embedding multilingue `paraphrase-multilingual-MiniLM-L12-v2` (50+ langues)
- Le prompt impose la cohérence linguistique stricte

---

### 5. Agent météo (1-2 min)
**À taper** :
```
Quel est le temps prévu à Marseille la semaine prochaine ?
```

**Réponse attendue** :
- Appel `get_forecast`
- Résumé jour par jour des températures, précipitations, vent
- Donut affiche le segment "Météo"

**À commenter** :
- API OpenMeteo, gratuite, sans clé
- Géocodage automatique (ville → coordonnées)
- Pattern ReAct : Reason ("c'est une question météo prévisionnelle") → Act (`get_forecast`) → Observe (JSON OpenMeteo) → Answer (synthèse en langage naturel)

---

### 6. Multi-outils chaînés (2 min)
**À taper** :
```
Quel temps fait-il à Paris aujourd'hui, et combien font 23.5 + 18.7 ?
```

**Réponse attendue** :
- Appel `get_weather` (Paris) puis `calculator` (23.5 + 18.7)
- Donut montre 2 segments différents allumés

**À commenter** :
- L'agent enchaîne plusieurs outils en une seule requête
- Le calculator passe par une sandbox AST (anti-injection)

---

### 7. ML prédictif (1 min)
**À taper** :
```
Quel est le risque catastrophe prédit au Bangladesh pour 2030 ?
```

**Réponse attendue** :
- Appel `predict_risk` (ou `predict_risk_by_type` selon le détail)
- Niveau de risque + impact humain moyen prédit
- Citation du modèle (Quantile Regression EM-DAT)

**À commenter** :
- Modèle entraîné sur EM-DAT décadal (1900-2020, 14 625 lignes)
- Quantile Regression robuste aux outliers (sinon canicule France 2003 fausserait tout)
- Clipping 1.5× max historique : empêche les prédictions extrêmes par extrapolation
- Comparatif MLflow : 16 modèles testés, ce GBQR est le meilleur sur MAE_test

---

### 8. Scoring multi-sources (2 min, point fort)
**À taper** :
```
Calcule le score de risque inondation à Bangkok à horizon court terme
```

**Réponse attendue** :
- L'agent enchaîne 4 outils : `get_forecast` + `predict_risk` + `search_corpus` + (historique)
- Puis appelle `calculer_score_risque`
- Score 0-1 + interprétation qualitative + sources

**À commenter** :
- C'est l'outil emblématique : il croise 4 sources en une seule passe
- Pondération dépendante de l'horizon : court_terme privilégie la météo, long_terme privilégie ML + corpus
- Le donut affiche 5 segments allumés simultanément

---

### 9. Mode décisionnel + HITL (2 min)
**À cliquer** : un des 4 boutons d'action décisionnelle (par exemple "Événementiel"), puis suivre le dialogue guidé.

**Réponse attendue** :
- Dialogue qui pose 2-3 questions (lieu, date, type d'événement)
- Réponse finale avec recommandation GO / NO-GO
- **Guide de lecture** intégré : `DECISION = ... | HORIZON = ... | SCORE = ... | RISQUE = ... | CONFIANCE = ...`
- Boutons d'action : Approuver / Enrichir / Rejeter

**À commenter** :
- Boucle Human-in-the-Loop : aucun système ne décide seul d'annuler un événement
- Toute décision finale est loggée dans MLflow comme feedback humain
- Le guide de lecture rend la sortie auto-explicative — pas besoin de glossaire séparé pour comprendre

---

### 10. Multimodal (2 min)
**Au choix selon le temps** :

#### 10a. Upload PDF
- Glisser-déposer un PDF de ton choix dans le chat
- Taper : `Résume-moi ce document en 5 points`
- Commenter : indexation FAISS in-memory **session-scoped** → pas de pollution du corpus officiel

#### 10b. Upload image
- Glisser-déposer une image (carte météo, schéma, etc.)
- Taper : `Que vois-tu sur cette image ?`
- Commenter : Claude vision natif, pas d'OCR séparé

#### 10c. STT (vocal)
- Cliquer sur le micro, dire : *"Quel temps fait-il à Lyon ?"* (ou en anglais/espagnol pour montrer l'autodétection)
- Commenter : Faster-Whisper en local, sans API tierce, autodétection de langue

#### 10d. TTS (lecture vocale)
- Sur n'importe quelle réponse, cliquer le bouton 🔊 en bas à gauche
- Commenter : Web Speech API du navigateur, gratuit, instantané

---

## Bonus si temps restant

### MCP via Claude Desktop
**Si tu as configuré le tunnel Cloudflare avant la soutenance** :
- Ouvre Claude Desktop sur ta machine
- Montre les 11 outils SAEARCH disponibles
- Demande à Claude Desktop : *"Cherche dans le corpus SAEARCH ce que disent les rapports sur les feux de forêt en Europe"*
- Commenter : pas de duplication de code, mêmes fonctions `@tool` partagées entre Chainlit et MCP

### Dashboard MLflow
**Si tu as lancé MLflow UI** :
- Onglet `rag-catastrophes-climatiques` → un run par requête
- Filtrer par `prompt_version` v1.0 vs v2.0 pour montrer l'A/B testing
- Onglet Model Registry : versions `NB10_regression v1+`, `NB10_classification v1+`

---

## Conclusion (1 min)

Points à clore :
- Architecture complète disponible dans `docs/architecture.md`, slides dans `docs/architecture_pour_slides.md`
- Diagramme interactif consultable séparément (`SAEARCH_architecture.html`)
- 25 décisions techniques justifiées dans `docs/decisions.md`
- Code et image disponibles : GitHub, Docker Hub, AWS ECR, HF Spaces

**Phrase de clôture suggérée** :
> *"Le projet combine 4 capacités — RAG scientifique, agents à outils, ML prédictif, aide à la décision humaine — sur les fondations posées par toute l'équipe en P1, P2 et P3. Toutes les décisions techniques sont justifiées et reproductibles via les workflows CI/CD."*

---

## Erreurs probables et plans B

| Si... | Plan B |
|---|---|
| L'API Anthropic timeout / 429 | Le fallback Haiku s'active automatiquement → continuer la démo, juste mentionner que la résilience marche |
| HF Spaces lent / timeout | Lancer une démo en local en backup (port 8090) |
| Une question RAG ne trouve rien | Reformuler avec un terme plus spécifique (`GIEC` au lieu de `IPCC`) |
| Le STT ne détecte pas le micro | Vérifier autorisation navigateur, sinon skip et passer au suivant |
| Le diagramme HTML ne s'ouvre pas | Ouvrir directement le fichier en double-clic depuis l'explorateur |
