from src.downloader import download_video
from src.transcription import transcribe_video
from src.video_processing import split_video
from src.subtitle_generator import add_subtitles

def main(youtube_url):
    # Étape 1 : Télécharger la vidéo
    video_path = download_video(youtube_url)
    
    if video_path:
        # Étape 2 : Transcrire l'audio
        transcription = transcribe_video(video_path)
        
        # Étape 3 : Découper la vidéo
        clips = split_video(video_path, transcription)
        
        # Étape 4 : Ajouter des sous-titres
        for clip in clips:
            add_subtitles(clip, transcription)

if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ")
    main(youtube_url)
