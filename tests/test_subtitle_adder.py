import os
from src.subtitle_adder import add_subtitles_ffmpeg
from src.paths import RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR

def test_add_subtitles_ffmpeg():
    # Chemins de test relatifs
    video_path = "data/raw/test.mp4"
    subtitles_path = "data/raw/test.vtt"
    output_path = "data/process/test_subtitled.mp4"

    # Cas 1 : Avec chemin de sortie explicite
    result_with_output = add_subtitles_ffmpeg(video_path, subtitles_path)
    assert result_with_output is not None, "L'ajout des sous-titres avec chemin explicite a échoué."
    assert os.path.exists(result_with_output), f"Le fichier {result_with_output} n'existe pas."

    # Cas 2 : Sans chemin de sortie explicite
    result_auto_output = add_subtitles_ffmpeg(video_path, subtitles_path)
    assert result_auto_output is not None, "L'ajout des sous-titres sans chemin explicite a échoué."
    assert os.path.exists(result_auto_output), f"Le fichier {result_auto_output} n'existe pas."
