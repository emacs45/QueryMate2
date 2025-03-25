from backend.ollama_client import ask_ollama

query = "Was ist NCP Secure Enterprise Management?"
response = ask_ollama(query)

print("🔍 Frage:", query)
print("🤖 Antwort von Ollama:")
print(response)
