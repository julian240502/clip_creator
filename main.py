import os
from src.downloader import download_video
from src.subtitle_adder import add_subtitles_ffmpeg
from src.video_splitter import split_video
from src.resizer import resize_clip_for_tiktok  # Ajout de la fonction de redimensionnement

def process_video(video_path):
    """
    Gère les étapes d'ajout de sous-titres, découpage et redimensionnement pour une vidéo donnée.
    """
    # Chemin des sous-titres générés par yt-dlp
    subtitles_path = video_path.replace(".mp4", ".en.vtt")

    # Étape 1 : Ajout des sous-titres
    processed_video_path = video_path.replace(".mp4", "_subtitled.mp4")
    if not os.path.exists(processed_video_path):
        if os.path.exists(subtitles_path):
            add_choice = input("Voulez-vous ajouter les sous-titres à la vidéo ? (o/n) : ").strip().lower()
            if add_choice == 'o':
                processed_video_path = add_subtitles_ffmpeg(video_path, subtitles_path)
                if processed_video_path:
                    print(f"Vidéo sous-titrée disponible : {processed_video_path}")
                else:
                    print("Erreur lors de l'ajout des sous-titres.")
            else:
                print("Ajout des sous-titres ignoré.")
        else:
            print("Fichier de sous-titres introuvable, sous-titres ignorés.")
    else:
        print(f"Les sous-titres ont déjà été ajoutés : {processed_video_path}")

    # Étape 2 : Découpage en clips
    output_dir = "data/clips"
    if not os.path.exists(output_dir) or not any(os.scandir(output_dir)):
        clip_choice = input("Voulez-vous découper la vidéo en clips ? (o/n) : ").strip().lower()
        if clip_choice == 'o':
            while True:
                try:
                    clip_length = int(input("Durée des clips en secondes : "))
                    if clip_length <= 0:
                        raise ValueError("La durée doit être un nombre positif.")
                    break
                except ValueError as e:
                    print(f"Erreur : {e}. Veuillez entrer un nombre valide.")

            clips = split_video(processed_video_path, clip_length, output_dir)
            print(f"{len(clips)} clips générés dans : {output_dir}")
            for clip in clips:
                print(f" - {clip}")
        else:
            print("Découpage ignoré.")
    else:
        print(f"Les clips existent déjà dans : {output_dir}")

    # Étape 3 : Redimensionnement des clips
    resized_dir = "data/clips_resized"
    if not os.path.exists(resized_dir) or not any(os.scandir(resized_dir)):
        resize_choice = input("Voulez-vous redimensionner les clips pour TikTok/Shorts ? (o/n) : ").strip().lower()
        if resize_choice == 'o':
            os.makedirs(resized_dir, exist_ok=True)
            clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")]
            for clip in clips:
                output_clip = os.path.join(resized_dir, os.path.basename(clip))
                resize_clip_for_tiktok(clip, output_clip)
                print(f"Clip redimensionné : {output_clip}")
        else:
            print("Redimensionnement ignoré.")
    else:
        print(f"Les clips redimensionnés existent déjà dans : {resized_dir}")

    # Fin du traitement
    print("Toutes les étapes ont été proposées. Fin du traitement.")


if __name__ == "__main__":
    youtube_url = input("Entrez l'URL de la vidéo YouTube : ")

    # Télécharger la vidéo
    video_path = download_video(youtube_url)

    if video_path:
        print(f"La vidéo a été téléchargée ou est déjà présente : {video_path}")
        process_video(video_path)
    else:
        print("Le téléchargement a échoué.")
