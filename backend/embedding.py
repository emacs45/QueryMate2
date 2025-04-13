from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Union

from backend.config import DEFAULT_EMBEDDING_TYPE
from backend.logger import app_logger


def get_embedding_function(embedding_type: str = DEFAULT_EMBEDDING_TYPE) -> Union[HuggingFaceEmbeddings, OllamaEmbeddings]:
    """
    Returns the appropriate embedding function based on the provided type.
    """
    # ollama 
    if embedding_type == "nomic":
        app_logger.info("🔧 Using embedding: nomic-embed-text (Ollama, Tokens: 8192)")
        return OllamaEmbeddings(model="nomic-embed-text")
    
    elif embedding_type == 'mxbai':
        app_logger.info("🔧 Using embedding: mxbai-embed-large (Ollama, Dimension: >= 1024))")
        return OllamaEmbeddings(model="mxbai-embed-large")
    
    #huggingface
    elif embedding_type == "sentence":
        app_logger.info("🔧 Using embedding: sentence-transformers/all-MiniLM-L6-v2 (Dimension: 384)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif embedding_type == "bge":
        app_logger.info("🔧 Using embedding: BAAI/bge-base-en-v1.5 (Dimension: 768)")
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    else:
        raise ValueError("❌ Invalid embedding_type. Allowed: 'nomic', 'sentence', 'bge', or 'mxbai")


# CLI test mode
if __name__ == "__main__":
    import sys
    import os
    import argparse

    # Add project root to sys.path for direct execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    parser = argparse.ArgumentParser(description="Initialize an embedding function.")
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["sentence", "bge", "nomic", "mxbai"],
        default=DEFAULT_EMBEDDING_TYPE,
        help="Choose embedding type: 'sentence', 'huggingface', 'nomic' or 'mxbai'"
    )
    args = parser.parse_args()

    embedding = get_embedding_function(args.embedding)
    print(f"✅ Embedding initialized: {embedding}")

