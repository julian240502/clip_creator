from src.audio_extractor import extract_audio

def test_extract_audio():
    video_path = "C:\Users\akues\OneDrive\Documents\GitHub\clip-creator\data\raw\test.mp4 b"  # Chemin d'une vidéo existante pour le test
    audio_path = extract_audio(video_path, output_path="data/raw")

    assert audio_path is not None, "L'extraction de l'audio a échoué."
    assert audio_path.endswith(".mp3"), "Le fichier audio n'est pas au format MP3."
