# Clip Creator ✦

Application locale pour télécharger ou importer une vidéo, la découper en clips et produire des exports verticaux prêts pour **CapCut** ou **Camtasia**.

## Fonctionnalités

- téléchargement via une URL prise en charge par `yt-dlp` ;
- import de fichiers MP4, MOV, MKV et WebM ;
- découpage précis avec une durée configurable ;
- export vertical 1080 × 1920 sans déformation ;
- choix entre recadrage plein écran et conservation de l'image entière ;
- aperçu et téléchargement individuel ou groupé en ZIP ;
- un dossier distinct par traitement dans `data/projects/`.

Les sous-titres ne sont volontairement pas intégrés : ils sont ajoutés ensuite dans CapCut ou Camtasia.

## Prérequis

- Python 3.10 ou plus récent ;
- `ffmpeg` et `ffprobe` disponibles dans le `PATH`.

Sous Windows :

```powershell
winget install Gyan.FFmpeg
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

## Tests

```bash
pytest
```

Les tests génèrent leurs propres médias avec FFmpeg et ne téléchargent aucune vidéo.
