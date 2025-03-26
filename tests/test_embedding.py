# tests/test_embedding.py

from backend.embedding import get_embedding_function
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def test_get_embedding_function_nomic():
    emb = get_embedding_function("nomic")
    assert isinstance(emb, OllamaEmbeddings)


def test_get_embedding_function_sentence():
    emb = get_embedding_function("sentence")
    assert isinstance(emb, HuggingFaceEmbeddings)
