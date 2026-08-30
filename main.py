"""Point d'entrée historique : lance désormais l'interface Clip Creator."""

import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"]))
