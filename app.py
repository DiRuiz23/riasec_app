# Entrypoint para Streamlit — redirige a src/presentation/streamlit_ui.py
import runpy
import sys
import os

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "src", "presentation", "streamlit_ui.py"),
    run_name="__main__",
)
