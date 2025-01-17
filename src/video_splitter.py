import os
import ffmpeg

def split_video(video_path, clip_length, output_dir):
    """
    Découpe une vidéo en plusieurs clips de durée fixe.

    Args:
        video_path (str): Chemin de la vidéo source.
        clip_length (int): Durée de chaque clip en secondes.
        output_dir (str): Répertoire où les clips seront sauvegardés.

    Returns:
        list: Liste des chemins des clips générés.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Obtenir les informations de la vidéo
    video_info = ffmpeg.probe(video_path)
    duration = float(video_info['format']['duration'])

    clips = []
    start_time = 0
    clip_num = 1

    # Découpage dynamique
    while start_time < duration:
        end_time = min(start_time + clip_length, duration)
        output_path = os.path.join(output_dir, f"clip_{clip_num:03d}.mp4")

        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .output(output_path, c="copy")
            .run(overwrite_output=True)
        )
        clips.append(output_path)
        start_time += clip_length
        clip_num += 1

    return clips
