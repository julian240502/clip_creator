import os
import openai
from src.paths import TRANSCRIPTIONS_DIR

def transcribe_video(video_path):
    """
    Transcrit une vidéo en texte à l'aide de Whisper et sauvegarde la transcription.

    Args:
        video_path (str): Chemin de la vidéo à transcrire.

    Returns:
        str: Chemin complet du fichier de transcription.
    """
    try:
        # Créer un nom de fichier pour la transcription
        transcription_filename = os.path.basename(video_path).replace(".mp4", ".txt")
        transcription_path = os.path.join(TRANSCRIPTIONS_DIR, transcription_filename)

        # Ouvrir le fichier vidéo pour Whisper
        with open(video_path, "rb") as video_file:
            response = openai.Audio.transcribe(
                model="whisper-1",  # Modèle Whisper
                file=video_file
            )
            transcription_text = response.get("text", "")

        # Enregistrer la transcription dans un fichier texte
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)

        print(f"Transcription sauvegardée : {transcription_path}")
        return transcription_path
    except Exception as e:
        print(f"Erreur lors de la transcription : {e}")
        return None


import whisper

def transcribe_video_local(video_path):
    """
    Transcrit une vidéo en texte à l'aide de Whisper exécuté sur GPU.

    Args:
        video_path (str): Chemin de la vidéo à transcrire.

    Returns:
        str: Texte transcrit.
    """
    try:
        # Charger le modèle Whisper (utilise automatiquement le GPU si disponible)
        model = whisper.load_model("base", device="cuda")

        # Transcrire la vidéo
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print(f"Erreur lors de la transcription locale avec GPU : {e}")
        return None
