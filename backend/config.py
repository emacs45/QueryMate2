# Wähle die Methode für Ollama: "requests" oder "library"
OLLAMA_METHOD = "library"  # Standard: "library"

# Konfiguriere die API-URL (nur für "requests"-Methode)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Wähle den globalen Embedding-Typ: "sentence", "nomic", "mxbai" oder "bge"
DEFAULT_EMBEDDING_TYPE = "mxbai"

# Nach dem Ändern neu initialisieren mit 
# python3 backend/chroma_index.py --reset

# Wähle das Standard-LLM-Modell
DEFAULT_LLM_MODEL = "mistral:latest"

# Debugging-Ausgaben
#print(f"🔍 Verwendete Methode für Ollama: {OLLAMA_METHOD}")
#print(f"🧠 Verwendeter Embedding-Typ: {EMBEDDING_TYPE}")
#print(f"🤖 Standardmodell: {DEFAULT_LLM_MODEL}")
