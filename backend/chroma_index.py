import os
import sys
import shutil
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from backend.config import DEFAULT_EMBEDDING_TYPE
from backend.embedding import get_embedding_function
from backend.logger import app_logger

# Pfade
CHROMA_INDEX_PATH = "models/chroma_index"
DATA_DIR = "data"

# Sicherstellen, dass Verzeichnisse vorhanden sind
Path(CHROMA_INDEX_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Embedding laden
embeddings = get_embedding_function(DEFAULT_EMBEDDING_TYPE)

def clear_chroma_index():
    if os.path.exists(CHROMA_INDEX_PATH):
        shutil.rmtree(CHROMA_INDEX_PATH)
        app_logger.info("🧹 Chroma-Index wurde vollständig gelöscht.")

def delete_chunks_by_source(source_name: str):
    """
    Löscht alle Chunks, die unter einer bestimmten 'source' gespeichert wurden.
    """
    db = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embeddings)

    try:
        # Alle Chunks mit Metadaten abrufen
        existing = db.get(include=["metadatas"])
        all_ids = existing["ids"]
        metadaten = existing["metadatas"]

        # IDs herausfiltern, deren 'source' zur gegebenen Quelle passt
        ids_to_delete = [
            chunk_id for chunk_id, metadata in zip(all_ids, metadaten)
            if metadata.get("source") == source_name
        ]

        if not ids_to_delete:
            app_logger.info(f"ℹ️ Keine Chunks mit Quelle '{source_name}' gefunden.")
            return

        db.delete(ids=ids_to_delete)
        app_logger.info(f"🗑️ {len(ids_to_delete)} Chunks mit Quelle '{source_name}' wurden gelöscht.")

    except Exception as e:
        app_logger.error(f"🚨 Fehler beim Löschen von Chunks mit Quelle '{source_name}': {e}")

def load_documents():
    all_docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            docs = loader.load()
            all_docs.extend(docs)
            app_logger.info(f"📄 Geladen: {filename} ({len(docs)} Seiten)")
    return all_docs

def split_documents(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    return splitter.split_documents(docs)

def assign_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embeddings)
    try:
        existing_ids = set(db.get(include=["metadatas"])["ids"])
        app_logger.info(f"📦 {len(existing_ids)} bestehende Chunks in der Datenbank gefunden.")
    except Exception as e:
        app_logger.warning(f"⚠️ Konnte bestehende Chunks nicht abrufen: {e}")
        existing_ids = set()

    new_chunks = [chunk for chunk in chunks if chunk.metadata.get("id") not in existing_ids]

    if not new_chunks:
        app_logger.info("ℹ️ Keine neuen Chunks zum Hinzufügen gefunden.")
        return

    new_ids = [chunk.metadata["id"] for chunk in new_chunks]

    try:
        db.add_documents(new_chunks, ids=new_ids)
        app_logger.info(f"✅ {len(new_chunks)} neue Chunks wurden in Chroma gespeichert.")
    except Exception as e:
        app_logger.error(f"🚨 Fehler beim Hinzufügen von Chunks zu ChromaDB: {e}")

def load_pdfs_and_index():
    docs = load_documents()
    if not docs:
        app_logger.warning("⚠️ Keine PDFs im data/-Ordner gefunden.")
        return

    chunks = split_documents(docs)
    chunks = assign_chunk_ids(chunks)
    add_to_chroma(chunks)

def query_chroma(query, top_k=4, source_filter=None, embeddings=None):
    """
    Führt eine semantische Suche mit ChromaDB durch, optional gefiltert nach Quelle.

    :param query: Die Suchanfrage als Text.
    :param top_k: Anzahl der zurückgegebenen Top-Ergebnisse.
    :param source_filter: Optionaler Filter auf die Dokumentquelle (Dateiname).
    :param embeddings: Optional: eine eigene Embedding-Funktion.
    :return: Liste von Seiteninhalten der besten Treffer.
    """

    if embeddings is None:
        embeddings = get_embedding_function(DEFAULT_EMBEDDING_TYPE)
        app_logger.debug("🔧 Kein Embedding übergeben")

    if not os.path.exists(CHROMA_INDEX_PATH):
        app_logger.warning("⚠️ Kein Chroma-Index gefunden! Bitte zuerst PDFs indexieren.")
        return ["⚠️ Kein Chroma-Index gefunden! Bitte zuerst PDFs indexieren."]

    vectorstore = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embeddings)

    # Filter optional anwenden
    filter_args = {"source": source_filter} if source_filter else None

    # Suche mit optionalem Filter
    results = vectorstore.similarity_search(query, k=top_k, filter=filter_args)

    if not results:
        app_logger.warning(f"⚠️ Keine passenden Ergebnisse für Anfrage: {query}")
        return ["⚠️ Keine relevanten Informationen gefunden."]

    app_logger.info(f"✅ ChromaDB hat {len(results)} passende Ergebnisse für die Anfrage gefunden.")
    return [f"<{r.metadata.get('source')}> {r.page_content}" for r in results]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Chroma-Index vollständig löschen")
    parser.add_argument("--delete-source", type=str, help="Löscht alle Chunks einer bestimmten Quelle (Dateiname)")
    args = parser.parse_args()

    if args.reset:
        clear_chroma_index()
        return

    if args.delete_source:
        delete_chunks_by_source(args.delete_source)
        return

    load_pdfs_and_index()

if __name__ == "__main__":
    main()