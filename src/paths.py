import os

# Dossier racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le dossier data. Redéfinissable via CLIP_CREATOR_DATA_DIR pour le
# sortir d'un dossier synchronisé (OneDrive, Drive…) qui peut verrouiller des
# fichiers fraîchement écrits, sans déplacer le dépôt git lui-même.
DATA_DIR = os.environ.get("CLIP_CREATOR_DATA_DIR") or os.path.join(BASE_DIR, "data")

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
