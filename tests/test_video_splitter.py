import os
import shutil
from src.video_splitter import split_video

def test_split_video():
    # Préparation des fichiers de test
    video_path = "data/raw/test.mp4"
    output_dir = "data/clips_test"
    clip_length = 25  # Durée en secondes

    # Assurez-vous que le fichier vidéo de test existe
    assert os.path.exists(video_path), f"Le fichier de test {video_path} est manquant."

    # Supprime le dossier de sortie s'il existe déjà (pour un test propre)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Exécuter la fonction de découpage
    clips = split_video(video_path, clip_length, output_dir)

    # Vérifications
    assert len(clips) > 0, "Aucun clip généré."
    assert os.path.exists(output_dir), "Le répertoire de sortie n'a pas été créé."
    for clip in clips:
        assert os.path.exists(clip), f"Le clip {clip} n'existe pas."
        assert clip.endswith(".mp4"), f"Le fichier {clip} n'est pas au format MP4."

    # Vérifier la durée des clips
    for clip in clips[:-1]:  # Tous sauf le dernier clip doivent avoir la bonne durée
        duration = float(os.popen(f"ffprobe -i {clip} -show_entries format=duration -v quiet -of csv=p=0").read().strip())
        assert round(duration) == clip_length, f"La durée du clip {clip} n'est pas correcte."

    print(f"Test terminé avec succès : {len(clips)} clips générés.")

# Exécuter manuellement pour le débogage
if __name__ == "__main__":
    test_split_video()
