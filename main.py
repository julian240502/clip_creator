import os
import time
from src.downloader import download_video
from src.subtitle_adder import add_subtitles_ffmpeg

if __name__ == "__main__":
    youtube_url = input("Entrez l'URL de la vidéo YouTube : ")

    # Télécharger la vidéo
    video_path = download_video(youtube_url)

    if video_path:
        # Si la vidéo existe déjà
        if os.path.exists(video_path):
            print(f"La vidéo '{video_path}' existe déjà.")

            # Proposer d'ajouter les sous-titres
            choice = input("Voulez-vous ajouter les sous-titres ? (o/n) : ").strip().lower()
            if choice == 'n':
                print("Opération annulée.")
                exit()
        else:
            # Attendre que le fichier soit téléchargé
            while not os.path.exists(video_path):
                print("En attente du téléchargement...")
                time.sleep(1)

        # Chemin des sous-titres générés par yt-dlp
        subtitles_path = video_path.replace(".mp4", ".en.vtt")

        # Vérifier si les sous-titres existent
        if not os.path.exists(subtitles_path):
            print(f"Le fichier de sous-titres '{subtitles_path}' est introuvable.")
            exit()

        # Ajouter les sous-titres à la vidéo
        processed_video_path = add_subtitles_ffmpeg(video_path, subtitles_path)
        if processed_video_path:
            print(f"Vidéo sous-titrée disponible : {processed_video_path}")
        else:
            print("Erreur lors de l'ajout des sous-titres.")
    else:
        print("Le téléchargement a échoué.")
