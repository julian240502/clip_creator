import os

# Dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le dossier data
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chemin pour les vidéos brutes (raw)
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw")

# Chemin pour les fichiers de transcription
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions")

# Chemin pour les vidéos sous-titrées (process)
PROCESSED_VIDEOS_DIR = os.path.join(DATA_DIR, "process")

# Assure que les dossiers existent
os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
