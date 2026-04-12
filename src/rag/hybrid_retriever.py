"""
Retriever hybride BM25 + Dense (EnsembleRetriever).
Combine la recherche sémantique (FAISS/MMR) avec la recherche par mots-clés (BM25/TF-IDF).
Référence : slide 11 du cours — "Hybrid: BM25 + Dense + Filters".

Ne modifie pas le retriever.py de Diego, l'enveloppe par-dessus.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.config import BM25_WEIGHT, DENSE_WEIGHT, RETRIEVER_K
from src.rag.retriever import creer_retriever


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

    return hybrid_retriever


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

        docs = hybrid.invoke(question)
        contexte = formater_contexte_avec_citations(docs)
        print("--- Contexte hybride BM25 + Dense ---\n")
        print(contexte)
