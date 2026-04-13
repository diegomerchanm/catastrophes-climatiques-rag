# Chargement et découpage des documents PDF du corpus climatique

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def charger_documents(dossier: str) -> list:
    """
    Charge tous les fichiers PDF d'un dossier donné.
    Retourne une liste de documents LangChain.
    """
    documents = []
    fichiers_pdf = [f for f in os.listdir(dossier) if f.endswith(".pdf")]

    if not fichiers_pdf:
        print(f"Aucun fichier PDF trouvé dans '{dossier}'.")
        return documents

    for nom_fichier in sorted(fichiers_pdf):
        chemin = os.path.join(dossier, nom_fichier)
        chargeur = PyPDFLoader(chemin)
        pages = chargeur.load()
        documents.extend(pages)
        print(f"  Chargé : {nom_fichier} ({len(pages)} pages)")

    print(
        f"\nTotal : {len(fichiers_pdf)} fichiers chargés, {len(documents)} pages au total."
    )
    return documents


def decouper_documents(documents: list) -> list:
    """
    Découpe une liste de documents LangChain en chunks.
    Utilise RecursiveCharacterTextSplitter avec chunk_size=1500 et chunk_overlap=150.
    Conserve les métadonnées (source, page) de chaque document.
    Retourne la liste des chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)
    print(f"Découpage terminé : {len(chunks)} chunks créés.")
    return chunks


def charger_et_decouper(dossier: str) -> list:
    """
    Fonction principale : charge les PDFs du dossier puis les découpe en chunks.
    Retourne la liste des chunks prêts pour l'indexation.
    """
    print(f"Chargement des documents depuis '{dossier}'...")
    documents = charger_documents(dossier)

    print("\nDécoupage des documents en chunks...")
    chunks = decouper_documents(documents)

    return chunks


if __name__ == "__main__":
    chunks = charger_et_decouper("data/raw")

    if chunks:
        print("\n--- Métadonnées du premier chunk ---")
        print(chunks[0].metadata)

        print("\n--- Métadonnées du dernier chunk ---")
        print(chunks[-1].metadata)
