import os

# Dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le dossier data
DATA_DIR = os.path.join(BASE_DIR, "data")

# Chemin pour les vidéos brutes (raw)
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw")

# Chemin pour les fichiers de transcription
TRANSCRIPTIONS_DIR = os.path.join(DATA_DIR, "transcriptions")

# Cache des vidéos sources téléchargées (indexé par URL + qualité)
SOURCE_CACHE_DIR = os.path.join(DATA_DIR, "sources")

# Assure que les dossiers existent
os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(SOURCE_CACHE_DIR, exist_ok=True)
