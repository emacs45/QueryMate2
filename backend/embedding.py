import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import DEFAULT_EMBEDDING_TYPE
from backend.logger import app_logger

#print(f"📦 [DEBUG] DEFAULT_EMBEDDING_TYPE: {DEFAULT_EMBEDDING_TYPE}")

def get_embedding_function(embedding_type: str = DEFAULT_EMBEDDING_TYPE):
    """Gibt eine passende Embedding-Funktion basierend auf dem Typ zurück."""

    if embedding_type == "nomic":
        app_logger.info("🔧 Verwendetes Embedding: nomic-embed-text (Ollama)")
        return OllamaEmbeddings(model="nomic-embed-text")

    elif embedding_type in ["sentence", "huggingface"]:
        app_logger.info("🔧 Verwendetes Embedding: sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    else:
        raise ValueError("❌ Ungültiger embedding_type. Erlaubt: 'sentence' oder 'nomic'")

# Optional über CLI testen
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialisiere eine Embedding-Funktion.")
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["sentence", "nomic"],
        default=DEFAULT_EMBEDDING_TYPE,
        help="Wähle den Embedding-Typ: 'sentence' oder 'nomic'"
    )
    args = parser.parse_args()

    embedding = get_embedding_function(args.embedding)
    print(f"✅ Embedding initialisiert: {embedding}")
