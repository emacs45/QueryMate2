import streamlit as st

def faq():
    """ Zeigt eine Liste häufig gestellter Fragen an. """
    
    with st.expander("ℹ️ Häufig gestellte Fragen (FAQ)", expanded=False):
        st.write("""
        **1. Wie lade ich eine Datei hoch?**  
        → Klicke auf "Datei hochladen" und wähle eine PDF-, TXT- oder DOCX-Datei aus.

        **2. Wie kann ich eine Datei löschen?**  
        → Gehe zur Sidebar unter "Datei-Management", wähle die Datei aus und klicke auf "❌".

        **3. Wie kann ich den Chatverlauf löschen?**  
        → Klicke in der Sidebar auf "Chatverlauf löschen".

        **4. Wie kann ich den Index aktualisieren?**  
        → Falls neue Dateien hochgeladen wurden, klicke auf "Index aktualisieren".

        **5. Welche Modelle kann ich auswählen?**  
        → Aktuell stehen `llama3.1:8b`, `gemma3:12b` und `mistral:latest` zur Verfügung.
        """)
