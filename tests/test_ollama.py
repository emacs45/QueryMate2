# tests/test_ollama_client.py

from backend.ollama_client import ask_ollama


def test_ask_ollama_mock():
    """Basistest, der überprüft, ob ein String zurückkommt."""
    response = ask_ollama("Was ist KI?", model="mistral:latest", method="library")
    assert isinstance(response, str)
    assert len(response) > 0