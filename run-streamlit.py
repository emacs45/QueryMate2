# run-streamlit.py
import os
import sys
from streamlit.web import cli as stcli

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "frontend/ui.py"]
    stcli.main()
