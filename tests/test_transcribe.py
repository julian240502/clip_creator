import os
from src.transcription import transcribe_video, transcribe_video_local
from src.paths import TRANSCRIPTIONS_DIR, RAW_VIDEOS_DIR, BASE_DIR

def test_transcribe_video_local():
    video_path = os.path.join(RAW_VIDEOS_DIR, "test.mp4")
  # Chemin d'une vidéo existante pour le test
    transcription_path = transcribe_video_local(video_path)

    assert transcription_path is not None, "La transcription a échoué."
    assert transcription_path.startswith(TRANSCRIPTIONS_DIR), "La transcription n'est pas enregistrée dans le bon dossier."
    assert os.path.exists(transcription_path), "Le fichier de transcription n'existe pas."
    with open(transcription_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0, "Le fichier de transcription est vide."
