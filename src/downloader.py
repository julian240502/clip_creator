import os
import subprocess
from src.paths import RAW_VIDEOS_DIR

def download_video(youtube_url):
    """
    Télécharge une vidéo YouTube avec sous-titres automatiques.

    Args:
        youtube_url (str): Lien de la vidéo YouTube.

    Returns:
        str: Chemin complet de la vidéo téléchargée.
        None: En cas d'échec.
    """
    try:
        # Commande yt-dlp
        command = [
            "yt-dlp", youtube_url,
            "-o", os.path.join(RAW_VIDEOS_DIR, "%(title)s.%(ext)s"),
            "--write-auto-subs", "--sub-format", "vtt",
            "--merge-output-format", "mp4"
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Erreur lors du téléchargement : {result.stderr}")
            return None

        print(f"Vidéo téléchargée avec succès : {result.stdout}")
        # Retourne le chemin de la vidéo téléchargée
        video_filename = result.stdout.split("Destination:")[-1].strip()
        return os.path.join(RAW_VIDEOS_DIR, video_filename)
    except Exception as e:
        print(f"Erreur inattendue lors du téléchargement : {e}")
        return None
