# Clip Creator ✦

Application locale pour télécharger ou importer une vidéo, la découper en clips et produire des exports verticaux prêts pour **CapCut** ou **Camtasia**.

## Fonctionnalités

- téléchargement via une URL prise en charge par `yt-dlp` ;
- import de fichiers MP4, MOV, MKV et WebM ;
- découpage précis avec une durée configurable ;
- export vertical sans déformation en 720p, 1080p ou 4K ;
- conservation intégrale de la vidéo paysage avec des bandes noires en 9:16 ;
- accélération matérielle automatique NVIDIA NVENC, Intel Quick Sync ou AMD AMF ;
- redimensionnement CUDA sur NVIDIA pour le recadrage vertical ;
- profils d'encodage rapide, équilibré ou qualité maximale ;
- découpage et conversion verticale en une seule passe d'encodage ;
- aperçu et téléchargement individuel ou groupé en ZIP ;
- un dossier distinct par traitement dans `data/projects/`.

Les sous-titres ne sont volontairement pas intégrés : ils sont ajoutés ensuite dans CapCut ou Camtasia.

## Prérequis

- Python 3.10 ou plus récent ;
- `ffmpeg` et `ffprobe` disponibles dans le `PATH`.
- Deno pour résoudre les protections JavaScript de YouTube.

Sous Windows :

```powershell
winget install Gyan.FFmpeg
winget install DenoLand.Deno
```

## Installation

```bash
git clone https://github.com/julian240502/clip_creator.git
cd clip_creator
python -m venv .venv
```

Activez l'environnement (`.venv\\Scripts\\activate` sous Windows ou `source .venv/bin/activate` sous macOS/Linux), puis :

```bash
python -m pip install -r requirements.txt
python main.py
```

L'interface s'ouvre normalement sur `http://localhost:8501`.

## Accélération GPU

Au démarrage, l'application teste réellement les encodeurs disponibles et
sélectionne automatiquement le premier GPU utilisable, dans cet ordre :
NVIDIA NVENC, Intel Quick Sync, AMD AMF, puis CPU x264. Le choix actif est
visible dans la barre latérale. Chaque GPU détecté et le CPU restent aussi
sélectionnables manuellement.

Avec une carte NVIDIA, le redimensionnement est confié à CUDA et l'encodage à
NVENC. Le profil **Rapide** utilise le preset NVENC P1. La vidéo paysage reste
entièrement visible dans le cadre vertical : aucun recadrage n'est appliqué.

La qualité 4K télécharge la meilleure source disponible jusqu'à 2160p et
produit un export vertical 2160 × 3840. Si la source n'existe pas en 4K,
`yt-dlp` utilise la meilleure qualité inférieure disponible.

Pour vérifier les encodeurs présents sous Windows :

```powershell
ffmpeg -hide_banner -encoders | Select-String "h264_nvenc|h264_qsv|h264_amf"
```

## Tests

```bash
pytest
```

Les tests génèrent leurs propres médias avec FFmpeg et ne téléchargent aucune vidéo.
