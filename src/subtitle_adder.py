import os
import subprocess

def add_subtitles_ffmpeg(input_video, subtitles_file, output_path=None):
    """
    Ajoute des sous-titres à une vidéo avec FFmpeg, en générant dynamiquement le chemin de sortie si nécessaire.

    Args:
        input_video (str): Chemin relatif de la vidéo source.
        subtitles_file (str): Chemin relatif du fichier de sous-titres.
        output_path (str, optional): Chemin relatif de la vidéo de sortie. Si None, un chemin sera généré automatiquement.

    Returns:
        str: Chemin relatif de la vidéo générée ou None en cas d'échec.
    """
    try:
        # Normaliser les chemins relatifs et utiliser des slashes
        input_video = os.path.relpath(input_video).replace("\\", "/")
        subtitles_file = os.path.relpath(subtitles_file).replace("\\", "/")

        # Générer un chemin de sortie si nécessaire
        if output_path is None:
            base, ext = os.path.splitext(os.path.basename(input_video))
            output_path = os.path.join("data", "process", f"{base}_subtitled{ext}")
        else:
            output_path = os.path.relpath(output_path).replace("\\", "/")

        # Encodage des chemins en UTF-8 pour gérer les caractères spéciaux
        input_video = input_video.encode('utf-8').decode('utf-8')
        subtitles_file = subtitles_file.encode('utf-8').decode('utf-8')
        output_path = output_path.encode('utf-8').decode('utf-8')

        # Générer la commande FFmpeg
        command = [
            "ffmpeg", "-i", input_video,
            "-vf", f"subtitles={subtitles_file}",
            output_path
        ]

        # Afficher la commande pour debugging
        print("Commande FFmpeg :", " ".join(command))

        # Exécuter la commande
        result = subprocess.run(command, capture_output=True, text=True, shell=False)
        if result.returncode != 0:
            print(f"Erreur FFmpeg : {result.stderr}")
            return None

        return output_path

    except Exception as e:
        print(f"Exception lors de l'ajout des sous-titres : {str(e)}")
        return None