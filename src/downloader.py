import os
from src.paths import RAW_VIDEOS_DIR
import subprocess

def download_video(youtube_url):
    """
    Télécharge une vidéo depuis YouTube dans le dossier RAW_VIDEOS_DIR.

    Args:
        youtube_url (str): URL de la vidéo YouTube.

    Returns:
        str: Chemin complet de la vidéo téléchargée.
        None: En cas d'échec.
    """
    try:
        # Commande yt-dlp
        command = [
            "yt-dlp",
            youtube_url,
            "-o", os.path.join(RAW_VIDEOS_DIR, "%(title)s.%(ext)s"),
            "--merge-output-format", "mp4"
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Erreur lors du téléchargement : {result.stderr}")
            return None

        print(f"Vidéo téléchargée avec succès : {result.stdout}")
        return RAW_VIDEOS_DIR
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return None
