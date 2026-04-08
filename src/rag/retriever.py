# Logique de récupération des documents avec MMR et citations

import os


def creer_retriever(vector_store):
    """
    Crée un retriever MMR à partir d'un vector store FAISS.
    Retourne k=4 documents en candidate pool de fetch_k=10.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )
    return retriever


def rechercher_documents(retriever, question: str) -> list:
    """
    Recherche les documents pertinents pour une question donnée.
    Retourne la liste brute des documents LangChain.
    """
    documents = retriever.invoke(question)
    print(f"{len(documents)} document(s) trouvé(s) pour la question.")
    return documents


def formater_contexte_avec_citations(documents: list) -> str:
    """
    Formate une liste de documents en un contexte lisible avec citations.
    Chaque bloc indique la source (nom de fichier sans chemin) et le numéro de page.
    Retourne une chaîne de caractères avec les blocs séparés par '---'.
    """
    blocs = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        bloc = f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        blocs.append(bloc)
    return "\n\n---\n\n".join(blocs)


def interroger_rag(retriever, question: str) -> dict:
    """
    Fonction principale du retriever : recherche et formate les documents pertinents.
    Retourne un dictionnaire avec :
      - 'contexte' : le contexte formaté avec citations
      - 'documents' : la liste brute des documents
    """
    documents = rechercher_documents(retriever, question)
    contexte = formater_contexte_avec_citations(documents)
    return {
        "contexte": contexte,
        "documents": documents,
    }


if __name__ == "__main__":
    from src.rag.embeddings import charger_vector_store

    vector_store = charger_vector_store()

    if vector_store is None:
        print("Impossible de charger le vector store. Veuillez d'abord exécuter embeddings.py.")
    else:
        retriever = creer_retriever(vector_store)

        question = "Quelles régions sont les plus vulnérables aux inondations selon le GIEC ?"
        print(f"\nQuestion : {question}\n")

        resultat = interroger_rag(retriever, question)
        print("\n--- Contexte formaté avec citations ---\n")
        print(resultat["contexte"])
