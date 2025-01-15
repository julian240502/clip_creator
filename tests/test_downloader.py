from src.downloader import download_video
from src.paths import RAW_VIDEOS_DIR

def test_download_video():
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Exemple d'URL
    video_path = download_video(youtube_url)

    assert video_path is not None, "Le téléchargement a échoué."
    assert RAW_VIDEOS_DIR in video_path, "La vidéo n'est pas dans le bon dossier."
