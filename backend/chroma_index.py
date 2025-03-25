from pathlib import Path
import os
import shutil
import argparse
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

# Embedding wird beim Start im Main gesetzt
embeddings = None

def clear_chroma_index():
    if os.path.exists(CHROMA_INDEX_PATH):
        shutil.rmtree(CHROMA_INDEX_PATH)
        app_logger.info("🧹 Chroma-Index wurde gelöscht.")

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

def query_chroma(query, top_k=20, embeddings=None):

    if embeddings is None:
        embeddings = get_embedding_function(DEFAULT_EMBEDDING_TYPE)
        app_logger.debug("🔧 Kein Embedding übergeben - Standard wird verwendet.")

    if not os.path.exists(CHROMA_INDEX_PATH):
        app_logger.warning("⚠️ Kein Chroma-Index gefunden! Bitte zuerst PDFs indexieren.")
        return ["⚠️ Keine relevanten Informationen gefunden."]

    vectorstore = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=top_k)

    if not results:
        app_logger.warning(f"⚠️ Keine passenden Ergebnisse für Anfrage: {query}")
        return ["⚠️ Keine relevanten Informationen gefunden."]

    app_logger.info(f"✅ ChromaDB hat {len(results)} passende Ergebnisse für die Anfrage gefunden.")
    return [r.page_content for r in results]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Chroma-Index zurücksetzen")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_TYPE, choices=["huggingface", "nomic"],
                        help="Welcher Embedding-Typ soll verwendet werden?")
    args = parser.parse_args()

    global embeddings
    embeddings = get_embedding_function(args.embedding)
#    print(f"📦 [DEBUG] Verwende Typ: {args.embedding}")

    if args.reset:
        clear_chroma_index()

    load_pdfs_and_index()

if __name__ == "__main__":
    main()
