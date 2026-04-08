# Création et persistance du vector store FAISS avec embeddings HuggingFace

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOSSIER_FAISS = "faiss_store"


def creer_vector_store(chunks: list) -> FAISS:
    """
    Crée un vector store FAISS à partir d'une liste de chunks LangChain.
    Utilise le modèle d'embeddings 'all-MiniLM-L6-v2' de HuggingFace.
    Sauvegarde le vector store sur disque dans DOSSIER_FAISS.
    Retourne le vector store.
    """
    print("Initialisation du modèle d'embeddings 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Création du vector store FAISS à partir de {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(DOSSIER_FAISS)
    print(f"Vector store sauvegardé dans '{DOSSIER_FAISS}' ({len(chunks)} chunks indexés).")

    return vector_store


def charger_vector_store() -> FAISS:
    """
    Charge un vector store FAISS existant depuis le disque.
    Retourne le vector store si le dossier existe, None sinon.
    """
    if not os.path.exists(DOSSIER_FAISS):
        print(f"Erreur : le dossier '{DOSSIER_FAISS}' est introuvable. Veuillez d'abord créer le vector store.")
        return None

    print(f"Chargement du vector store depuis '{DOSSIER_FAISS}'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(DOSSIER_FAISS, embeddings, allow_dangerous_deserialization=True)
    print("Vector store chargé avec succès.")
    return vector_store


def obtenir_ou_creer_vector_store(chunks: list = None) -> FAISS:
    """
    Fonction principale : retourne un vector store FAISS prêt à l'emploi.
    - Si un vector store existe déjà sur disque, il est chargé et retourné.
    - Sinon, un nouveau vector store est créé à partir des chunks fournis.
    """
    if os.path.exists(DOSSIER_FAISS):
        print("Vector store existant détecté, chargement en cours...")
        return charger_vector_store()

    if not chunks:
        print("Erreur : aucun chunk fourni et aucun vector store existant sur disque.")
        return None

    print("Aucun vector store existant, création à partir des chunks fournis...")
    return creer_vector_store(chunks)


if __name__ == "__main__":
    from src.rag.loader import charger_et_decouper

    chunks = charger_et_decouper("data/raw")
    vector_store = creer_vector_store(chunks)

    print(f"\nNombre de vecteurs dans le store : {vector_store.index.ntotal}")
