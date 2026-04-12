"""
Retriever hybride BM25 + Dense (EnsembleRetriever) avec reranking.
Combine la recherche sémantique (FAISS/MMR) avec la recherche par mots-clés (BM25/TF-IDF).
Résolution du Lost in the Middle (slide 14) via reranking + placement stratégique.
Référence : slide 11 du cours — "Hybrid: BM25 + Dense + Filters".

Ne modifie pas le retriever.py de Diego, l'enveloppe par-dessus.
"""

import logging

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import BM25_WEIGHT, DENSE_WEIGHT, RETRIEVER_K
from src.rag.retriever import creer_retriever

logger = logging.getLogger(__name__)


def creer_hybrid_retriever(vector_store, chunks: list) -> EnsembleRetriever:
    """
    Crée un retriever hybride combinant :
    - Dense (FAISS/MMR) : comprend le sens sémantique des questions
    - BM25 (TF-IDF) : trouve les correspondances exactes de mots-clés

    Args:
        vector_store: Le vector store FAISS chargé.
        chunks: La liste des documents LangChain (nécessaire pour BM25).

    Returns:
        EnsembleRetriever pondéré BM25 + Dense.
    """
    # Retriever dense (celui de Diego, inchangé)
    dense_retriever = creer_retriever(vector_store)

    # Retriever BM25 (recherche par mots-clés)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=RETRIEVER_K)

    # Combinaison pondérée
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[BM25_WEIGHT, DENSE_WEIGHT],
    )

    logger.info(
        "Retriever hybride créé (BM25=%.1f, Dense=%.1f, k=%d)",
        BM25_WEIGHT,
        DENSE_WEIGHT,
        RETRIEVER_K,
    )

    return hybrid_retriever


# ══════════════════════════════════════════════════════════════════
# Reranking — résolution du Lost in the Middle (slide 14)
# ══════════════════════════════════════════════════════════════════


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """
    Reranke les documents avec un cross-encoder pour améliorer la pertinence.
    Résout le problème du Lost in the Middle en triant par pertinence réelle.

    Args:
        query: La question de l'utilisateur.
        documents: Les documents retournés par le retriever.

    Returns:
        Documents triés par pertinence décroissante.
    """
    if not documents:
        return documents

    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)

        # Trier par score décroissant
        scored_docs = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )
        reranked = [doc for _, doc in scored_docs]

        logger.info(
            "Reranking effectué : scores min=%.3f, max=%.3f",
            min(scores),
            max(scores),
        )

        return reranked
    except Exception as exc:
        logger.warning("Reranking échoué, retour des documents bruts : %s", exc)
        return documents


def placement_strategique(documents: list[Document]) -> list[Document]:
    """
    Place les documents stratégiquement pour éviter le Lost in the Middle.
    Le meilleur document en premier, le deuxième en dernier,
    les moins pertinents au milieu.

    Hypothèse : les documents sont déjà triés par pertinence (après reranking).

    Args:
        documents: Documents triés par pertinence décroissante.

    Returns:
        Documents réordonnés : [1er, 4e, 3e, 2e] (début et fin = meilleurs).
    """
    if len(documents) <= 2:
        return documents

    # Premier = le plus pertinent (début — bien vu par le LLM)
    # Dernier = le deuxième plus pertinent (fin — bien vu par le LLM)
    # Milieu = les moins pertinents (zone d'oubli du LLM)
    reordered = [documents[0]]
    reordered.extend(documents[2:])
    reordered.append(documents[1])

    logger.debug("Placement stratégique appliqué sur %d documents", len(documents))

    return reordered


def recherche_avec_reranking(
    retriever, query: str, rerank: bool = True
) -> list[Document]:
    """
    Pipeline complet : retriever → reranking → placement stratégique.

    Args:
        retriever: Le retriever (hybride ou dense).
        query: La question de l'utilisateur.
        rerank: Activer le reranking (défaut True).

    Returns:
        Documents optimisés pour le LLM.
    """
    documents = retriever.invoke(query)
    logger.info("%d documents récupérés pour : %s", len(documents), query[:80])

    if rerank and documents:
        documents = rerank_documents(query, documents)
        documents = placement_strategique(documents)

    return documents


if __name__ == "__main__":
    from src.rag.embeddings import charger_vector_store
    from src.rag.loader import charger_et_decouper
    from src.rag.retriever import formater_contexte_avec_citations

    vector_store = charger_vector_store()
    if vector_store is None:
        print("Vector store introuvable. Lancez d'abord embeddings.py.")
    else:
        chunks = charger_et_decouper("data/raw")
        hybrid = creer_hybrid_retriever(vector_store, chunks)

        question = "Quelles sont les recommandations du rapport CELEX sur les inondations ?"
        print(f"\nQuestion : {question}\n")

        # Sans reranking
        docs_bruts = hybrid.invoke(question)
        print("--- Sans reranking ---")
        print(formater_contexte_avec_citations(docs_bruts))

        # Avec reranking + placement stratégique
        docs_reranked = recherche_avec_reranking(hybrid, question)
        print("\n--- Avec reranking + placement stratégique ---")
        print(formater_contexte_avec_citations(docs_reranked))
