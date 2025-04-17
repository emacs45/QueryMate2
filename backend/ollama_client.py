import argparse
import requests
import ollama
from backend.chroma_index import query_chroma
from backend.config import OLLAMA_URL
from backend.logger import app_logger


def ask_ollama(question: str, model: str, method: str):
    """ Fragt den Chatbot mit Chroma als Wissensbasis und nutzt das ausgewählte LLM-Modell. """

    app_logger.info(f"🔍 Eingehende Anfrage: '{question}' mit Modell: {model} (Methode: {method})")

    # Suche relevante Informationen in ChromaDB
    context_results = query_chroma(question)

    if not context_results or "Keine relevanten Informationen gefunden." in context_results:
        app_logger.warning("⚠️ Keine passenden Informationen in Chroma gefunden.")
        return "⚠️ Ich konnte dazu leider keine passenden Informationen finden."

    # Erstelle das Prompt für Ollama mit dem gefundenen Kontext
    context_text = "\n\n".join(context_results)
    prompt = f"""
    Du bist ein KI-Assistent für Unternehmensdokumente. Deine Aufgabe ist es, präzise Antworten basierend auf diesem Unternehmenswissen zu geben.

    ### Kontext aus Unternehmensdokumenten:
    {context_text}

    ---
    **Anweisung:**  
    - Verwende nur die Informationen aus dem Kontext.  
    - Wenn du keine passende Information findest, antworte mit: „Ich konnte dazu keine Information finden.“  
    - Fasse die Informationen klar, verständlich und sachlich zusammen.  
    - Antworte in der Sprache der Frage.  
    - Füge nur dann Inline-Zitate im Format [source_id] ein, wenn das Tag <source_id> explizit im Kontext vorhanden ist.  
    - Verwende keine XML-Tags in deiner Antwort.  
    - Wenn der Kontext unvollständig oder schwer verständlich ist, weise höflich darauf hin und gib die bestmögliche Antwort.  
    - Wenn du die Antwort kennst, aber sie nicht im Kontext vorkommt, erkläre das ehrlich und antworte aus deinem Wissen.  
    - Zitiere nur relevante Stellen und sei dabei knapp und präzise.

    **Beispiel für ein korrektes Zitat:**  
    „Die Methode verbessert die Systemleistung um 20 % [whitepaper.pdf].“

    **Frage:**  
    {question}
    """


    if method == "requests":
        # Methode mit `requests`
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "⚠️ Keine Antwort erhalten.")
            return result
        except requests.exceptions.RequestException as e:
            app_logger.error(f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}")
            return f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}"

    else:
        # Methode mit `ollama`-Bibliothek
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            result = response.get("message", {}).get("content", "⚠️ Keine Antwort erhalten.")
            return result
        except Exception as e:
            app_logger.error(f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}")
            return f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fragt Ollama mit einem bestimmten Modell und einer Methode zur API-Kommunikation.")
    
    parser.add_argument("--question", type=str, required=True, help="Die Frage an den Chatbot.")
    parser.add_argument("--model", type=str, default="mistral:latest", help="Das LLM-Modell für Ollama.")
    parser.add_argument("--method", type=str, choices=["requests", "library"], default="library",
                        help="Die Methode, um Ollama anzusprechen: 'requests' oder 'library'.")

    args = parser.parse_args()

    # Ollama-Anfrage starten
    response = ask_ollama(args.question, args.model, args.method)
    print("🔍 Ollama-Antwort:", response)