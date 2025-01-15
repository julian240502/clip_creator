import os
import subprocess

def extract_audio(video_path, output_path="data/raw"):
    """
    Extrait l'audio d'une vidéo et enregistre un fichier MP3.

    Args:
        video_path (str): Chemin de la vidéo source.
        output_path (str): Répertoire où enregistrer l'audio extrait.

    Returns:
        str: Chemin complet de l'audio extrait.
        None: En cas d'échec.
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        audio_path = os.path.join(output_path, os.path.basename(video_path).replace(".mp4", ".mp3"))

        # Commande FFmpeg pour extraire l'audio
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path, "-y"
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Erreur lors de l'extraction de l'audio : {result.stderr}")
            return None

        print(f"Audio extrait avec succès : {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return None
