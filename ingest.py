"""
ingest.py — Módulo de ingesta (Setup) del sistema RAG
=====================================================

Lee los documentos de ./data, los fragmenta (chunking) y los persiste como
embeddings en una colección local de ChromaDB (./vectorstore).

Uso:
    python ingest.py              # crea la base si no existe
    python ingest.py --rebuild    # borra y reconstruye la base desde cero
"""

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_PATH = Path("data")
VECTORSTORE_PATH = "vectorstore"

# Debe coincidir EXACTAMENTE con el modelo usado en rag.py.
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 500     # tokens
CHUNK_OVERLAP = 50   # tokens

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def cargar_documentos() -> list:
    """Carga todos los .txt y .md de la carpeta ./data."""
    documentos = []
    for archivo in sorted(list(DATA_PATH.glob("*.txt")) + list(DATA_PATH.glob("*.md"))):
        loader = TextLoader(str(archivo), encoding="utf-8")
        documentos.extend(loader.load())
    return documentos


def construir_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en TOKENS reales (tiktoken), como pide la consigna.

    La primera ejecución descarga el vocabulario de tiktoken. Si estás sin
    internet o detrás de un proxy que lo bloquea, usá el fallback por
    caracteres: 1 token ≈ 4 caracteres en español.
    """
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    except Exception:
        print("Aviso: no se pudo usar tiktoken, se fragmenta por caracteres.")
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * 4,
            chunk_overlap=CHUNK_OVERLAP * 4,
        )


def ingest_documents(rebuild: bool = False) -> None:
    # Persistencia: no re-indexamos si la base ya existe (ahorra tiempo y costo).
    if Path(VECTORSTORE_PATH).exists():
        if not rebuild:
            print(
                f"La base vectorial '{VECTORSTORE_PATH}' ya existe. "
                "Usá --rebuild para reconstruirla."
            )
            return
        print("Eliminando la base vectorial existente...")
        shutil.rmtree(VECTORSTORE_PATH)

    documentos = cargar_documentos()
    if not documentos:
        print(f"No se encontraron documentos en '{DATA_PATH}'.")
        return
    print(f"Se cargaron {len(documentos)} documentos.")

    chunks = construir_splitter().split_documents(documentos)
    print(f"Se generaron {len(chunks)} fragmentos.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
    )
    print("Base vectorial creada correctamente en ./vectorstore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos a ChromaDB")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Borra la base vectorial existente y la reconstruye.",
    )
    args = parser.parse_args()
    ingest_documents(rebuild=args.rebuild)
