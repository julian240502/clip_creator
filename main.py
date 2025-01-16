from src.downloader import download_video
from src.subtitle_adder import add_subtitles_ffmpeg

if __name__ == "__main__":
    youtube_url = input("Entrez l'URL de la vidéo YouTube : ")

    # Télécharger la vidéo avec sous-titres
    video_path = download_video(youtube_url)
    if video_path:
        # Chemin du fichier de sous-titres généré par yt-dlp
        subtitles_path = video_path.replace(".mp4", ".vtt")

        # Ajouter les sous-titres à la vidéo
        processed_video_path = add_subtitles_ffmpeg(video_path, subtitles_path)
        if processed_video_path:
            print(f"Vidéo sous-titrée disponible : {processed_video_path}")
        else:
            print("Erreur lors de l'ajout des sous-titres.")
    else:
        print("Le téléchargement a échoué.")
