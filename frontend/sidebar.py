import os
import streamlit as st
from backend.chroma_index import load_pdfs_and_index
from backend.logger import app_logger

DATA_DIR = "data"

def sidebar():
    """ Erstellt die Sidebar mit Modell-Auswahl und PDF-Management. """
    st.sidebar.header("🤖 LLM-Modelle")
    
    # Modell-Auswahl speichern
    llm_option = st.sidebar.selectbox(
        "Modell auswählen",
        ["llama3.1:8b", "gemma3:12b", "mistral:latest"]
    )
    st.session_state["selected_model"] = llm_option
    app_logger.info(f"🔍 Modell ausgewählt: {llm_option}")

    # PDF-Management
    st.sidebar.header("📂 Hochgeladene Dateien")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".txt", ".docx"))]

    if pdf_files:
        for file in pdf_files:
            col1, col2 = st.sidebar.columns([0.8, 0.2])  # Spalten für Text und Button
            col1.write(f"📄 {file}")  # Dateiname anzeigen
            if col2.button("❌", key=file):  # "X"-Button zum Löschen
                file_path = os.path.join(DATA_DIR, file)
                os.remove(file_path)
                app_logger.info(f"🗑️ Datei gelöscht: {file}")

                # Re-Indexierung als notwendig markieren, aber noch nicht starten
                st.session_state["reindex_needed"] = True
                st.rerun()  # UI aktualisieren

    else:
        st.sidebar.write("🚀 Noch keine Dateien hochgeladen.")

    # Falls eine Datei gelöscht oder hochgeladen wurde → Indexierung ausführen
    if st.session_state.get("reindex_needed", False) or st.sidebar.button("🔄 Index aktualisieren"):
        load_pdfs_and_index()
        st.sidebar.success("✅ ChromaDB-Index aktualisiert!")
        app_logger.info("✅ ChromaDB-Index wurde aktualisiert.")
        st.session_state["reindex_needed"] = False  # Verhindert erneute unnötige Indexierung

    # Chatverlauf löschen
    if st.sidebar.button("🗑️ Chatverlauf löschen"):
        st.session_state.messages = []
        app_logger.info("🗑️ Chatverlauf wurde gelöscht.")
        st.sidebar.success("✅ Chatverlauf gelöscht!")
        st.rerun()  # UI aktualisieren
