import asyncio
import streamlit as st
import os
from backend.ollama_client import ask_ollama
from backend.chroma_index import query_chroma
from backend.config import OLLAMA_METHOD, DEFAULT_EMBEDDING_TYPE
from backend.embedding import get_embedding_function
from backend.logger import app_logger  # Logger importiert
from frontend.sidebar import sidebar
from frontend.faq import faq

# Log-Start der UI
app_logger.info("🔵 QueryMate UI gestartet.")

embeddings = get_embedding_function(DEFAULT_EMBEDDING_TYPE)

# Deaktiviere JIT für Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit Setup
st.set_page_config(page_title="QueryMate", page_icon="🤖", layout="wide")
st.title("QueryMate 🤖")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Sidebar laden
sidebar()

# FAQ anzeigen
faq()

# Verzeichnis für Dateien
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Datei-Upload für PDFs, TXT, DOCX
st.header("📥 Datei hochladen")
uploaded_files = st.file_uploader("Lade eine Datei hoch", accept_multiple_files=True, type=["pdf", "txt", "docx"])
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        app_logger.info(f"📄 Datei hochgeladen: {file.name}")
    st.success("✅ Datei(en) erfolgreich hochgeladen!")

# Chatverlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chatverlauf anzeigen
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat-Eingabe
if prompt := st.chat_input("Wie kann ich helfen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    app_logger.info(f"🗨️ User-Frage: {prompt}")

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        # Chroma-Abfrage ausführen
        try:
            chroma_response = query_chroma(prompt, embeddings=embeddings)
            context = "\n\n".join(chroma_response) if chroma_response else "Keine relevanten Informationen gefunden."
            app_logger.info(f"✅ ChromaDB hat {len(chroma_response)} passende Ergebnisse für die Anfrage gefunden.")
        except Exception as e:
            app_logger.error(f"🚨 Fehler bei der Chroma-Abfrage: {str(e)}")
            context = "Keine relevanten Informationen gefunden."

        # Falls keine relevanten Infos in Chroma gefunden wurden
        if not context or "Keine relevanten Informationen gefunden." in context:
            full_response = "⚠️ Ich konnte dazu leider keine passenden Informationen finden."
            app_logger.warning("⚠️ Keine relevanten Informationen gefunden.")
        else:
            # Ollama LLM verwenden
            selected_model = st.session_state.get("selected_model", "mistral:latest")
            try:
                full_response = ask_ollama(prompt, model=selected_model, method=OLLAMA_METHOD)
                app_logger.info(f"✅ Erfolgreiche Antwort von {selected_model} erhalten.")
            except Exception as e:
                app_logger.error(f"🚨 Fehler bei der Ollama-Anfrage: {str(e)}")
                full_response = "⚠️ Fehler bei der Verarbeitung der Anfrage."

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})