import os

# Wähle die Methode für Ollama: "requests" oder "library"
OLLAMA_METHOD = os.getenv("OLLAMA_METHOD", "library")  # Standard: "library"

# Konfiguriere die API-URL (nur für "requests"-Methode)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# Wähle den globalen Embedding-Typ: "huggingface" oder "nomic"
DEFAULT_EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "nomic")

# Nach dem Ändern neu initialisieren mit 
# python3 backend/chroma_index.py --reset

# Wähle das Standard-LLM-Modell
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Debugging-Ausgaben
#print(f"🔍 Verwendete Methode für Ollama: {OLLAMA_METHOD}")
#print(f"🧠 Verwendeter Embedding-Typ: {EMBEDDING_TYPE}")
#print(f"🤖 Standardmodell: {DEFAULT_LLM_MODEL}")
