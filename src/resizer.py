import ffmpeg

def resize_clip_for_tiktok(input_path, output_path):
    """
    Redimensionne une vidéo pour TikTok/Shorts (9:16) avec padding.

    Args:
        input_path (str): Chemin de la vidéo source.
        output_path (str): Chemin de la vidéo redimensionnée.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('scale', 1080, 1920)
            .filter('pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2')
            .output(output_path)
            .run(overwrite_output=True)
        )
        print(f"Vidéo redimensionnée avec succès : {output_path}")
    except ffmpeg.Error as e:
        print(f"Erreur lors du redimensionnement : {e.stderr.decode()}")

