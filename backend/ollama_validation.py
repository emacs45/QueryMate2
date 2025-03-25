import argparse
import requests
from sentence_transformers import SentenceTransformer, util
from backend.ollama_client import ask_ollama
from backend.config import OLLAMA_URL
from backend.logger import app_logger

# Evaluations-Prompt für Ollama
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def query_and_validate(question: str, expected_response: str, model: str, method: str):
    """ Prüft die Antwort des Chatbots mit Ollama. """

    app_logger.info(f"🔍 Validierung gestartet für Frage: '{question}' mit Modell: {model} (Methode: {method})")

    # Generiere die Antwort mit Chroma-gestützter KI
    actual_response = ask_ollama(question, model=model)

    if not actual_response:
        app_logger.warning("⚠️ Keine Antwort von Ollama erhalten.")
        return False

    # Berechne die semantische Ähnlichkeit
    similarity_score = evaluate_similarity(expected_response, actual_response)
    app_logger.info(f"📏 Ähnlichkeitsscore: {similarity_score:.2f}")

    if similarity_score > 0.8:
        app_logger.info("✅ Antwort ist semantisch ähnlich genug! Validierung erfolgreich.")
        return True

    app_logger.info("⚠️ Antwort war nicht ähnlich genug → Starte klassische `true/false` Bewertung mit Ollama.")

    # Falls die Antwort nicht ähnlich genug ist, prüfe mit klassischem `true/false`-Modell
    prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=actual_response)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        evaluation_results_str = response.json().get("response", "").strip().lower()

        app_logger.info(f"📊 Evaluationsantwort: {evaluation_results_str}")

        if "true" in evaluation_results_str:
            app_logger.info("✅ Validierung erfolgreich!")
            return True
        elif "false" in evaluation_results_str:
            app_logger.warning("❌ Validierung fehlgeschlagen!")
            return False
        else:
            app_logger.error(f"🚨 Ungültiges Evaluations-Ergebnis: {evaluation_results_str}")
            raise ValueError(f"Ungültiges Evaluations-Ergebnis: {evaluation_results_str}")

    except requests.exceptions.RequestException as e:
        app_logger.error(f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}")
        return False

def evaluate_similarity(expected, actual):
    """ Berechnet die semantische Ähnlichkeit zwischen zwei Antworten. """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    score = util.pytorch_cos_sim(model.encode(expected), model.encode(actual))
    return score.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validiert die Antwort eines LLM gegen eine erwartete Antwort.")

    parser.add_argument("--question", type=str, required=True, help="Die Frage an den Chatbot.")
    parser.add_argument("--expected", type=str, required=True, help="Die erwartete Antwort für die Validierung.")
    parser.add_argument("--model", type=str, default="mistral:latest", help="Das LLM-Modell für Ollama.")
    parser.add_argument("--method", type=str, choices=["requests", "library"], default="library",
                        help="Die Methode, um Ollama anzusprechen: 'requests' oder 'library'.")

    args = parser.parse_args()

    # Validierung starten
    result = query_and_validate(args.question, args.expected, args.model, args.method)
    print("✅ Validierungsergebnis:", result)
