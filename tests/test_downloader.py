from src.downloader import download_video

def test_download_video():
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Exemple d'URL publique
    video_path = download_video(youtube_url, output_path="data/raw")

    assert video_path is not None, "Le téléchargement a échoué"
