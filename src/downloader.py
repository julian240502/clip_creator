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
        # Assurer que le répertoire de destination existe
        os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)

        # Récupérer le chemin attendu
        get_filename_command = [
            "yt-dlp", "--get-filename",
            "-o", os.path.join(RAW_VIDEOS_DIR, "%(title)s.%(ext)s"),
            youtube_url
        ]
        filename_result = subprocess.run(get_filename_command, capture_output=True, text=True)
        if filename_result.returncode != 0:
            print(f"Erreur lors de la récupération du nom de fichier : {filename_result.stderr}")
            return None

        # Chemin attendu pour la vidéo
        expected_path = filename_result.stdout.strip()
        print(f"Chemin attendu pour la vidéo : {expected_path}")

        # Si le fichier existe déjà, ne pas télécharger
        if os.path.exists(expected_path):
            print(f"La vidéo existe déjà : {expected_path}")
            return expected_path

        # Commande yt-dlp pour télécharger
        download_command = [
            "yt-dlp", youtube_url,
            "-o", os.path.join(RAW_VIDEOS_DIR, "%(title)s.%(ext)s"),
            "--write-auto-subs", "--sub-format", "vtt",
            "--merge-output-format", "mp4",
            "--no-overwrites",
            
        ]
        print("Téléchargement en cours :", " ".join(download_command))
        download_result = subprocess.run(download_command, capture_output=True, text=True)
        if download_result.returncode != 0:
            print(f"Erreur lors du téléchargement : {download_result.stderr}")
            return None

        # Vérifier le fichier téléchargé dans le répertoire
        downloaded_files = os.listdir(RAW_VIDEOS_DIR)
        if not downloaded_files:
            print("Aucun fichier téléchargé.")
            return None

        # Identifier le fichier le plus récent (pour garantir la correspondance)
        downloaded_file = max(
            [os.path.join(RAW_VIDEOS_DIR, f) for f in downloaded_files],
            key=os.path.getctime
        )
        print(f"Vidéo téléchargée avec succès : {downloaded_file}")
        return downloaded_file

    except Exception as e:
        print(f"Erreur inattendue lors du téléchargement : {e}")
        return None
