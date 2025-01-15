import os
import subprocess

def download_video(youtube_url, output_path="data/raw"):
    """
    Télécharge une vidéo depuis YouTube en utilisant yt-dlp.

    Args:
        youtube_url (str): Lien de la vidéo YouTube.
        output_path (str): Chemin du répertoire où enregistrer la vidéo.

    Returns:
        str: Chemin complet de la vidéo téléchargée.
        None: En cas d'échec.
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Commande yt-dlp pour télécharger la vidéo
        command = [
            "yt-dlp",
            youtube_url,
            "-o", os.path.join(output_path, "%(title)s.%(ext)s"),
            "--merge-output-format", "mp4"
        ]

        # Exécuter la commande
        result = subprocess.run(command, capture_output=True, text=True)

        # Vérification de l'exécution
        if result.returncode != 0:
            print(f"Erreur lors du téléchargement : {result.stderr}")
            return None

        # Extraire le chemin du fichier depuis la sortie standard
        print(f"Téléchargement réussi : {result.stdout}")
        return os.path.join(output_path, f"{youtube_url.split('=')[-1]}.mp4")  # Nom générique
    except Exception as e:
        print(f"Erreur inattendue lors du téléchargement : {e}")
        return None
