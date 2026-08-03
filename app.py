# Entrypoint para compatibilidad hacia atrás
import sys
import os

# Asegurar que el directorio raíz está en el path para las importaciones de src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

filepath = os.path.join(os.path.dirname(__file__), "src", "presentation", "streamlit_ui.py")
with open(filepath, "r", encoding="utf-8") as f:
    code = compile(f.read(), filepath, 'exec')
    exec(code, globals(), locals())
