# Clip Creator ✦

Application locale pour télécharger ou importer une vidéo, la découper en clips et produire des exports verticaux prêts pour **CapCut** ou **Camtasia**.

Le flux se fait en deux temps : on **charge et prévisualise** la vidéo (lecteur intégré, titre, durée, résolution), on **choisit la portion à traiter** et les réglages, puis on **génère les clips**, affichés au fur et à mesure avec une vignette et un aperçu à la demande.

## Fonctionnalités

- téléchargement via une URL prise en charge par `yt-dlp` ;
- import de fichiers MP4, MOV, MKV et WebM ;
- prévisualisation de la source avant traitement, avec métadonnées ;
- sélection d'une portion de la vidéo à clipper et estimation du nombre de clips ;
- découpage précis avec une durée configurable ;
- export vertical sans déformation en 720p, 1080p ou 4K ;
- conservation intégrale de la vidéo paysage en 9:16 ;
- choix entre un arrière-plan vidéo flouté et des bandes noires ;
- accélération matérielle automatique NVIDIA NVENC, Intel Quick Sync ou AMD AMF ;
- redimensionnement CUDA sur NVIDIA pour le recadrage vertical ;
- profils d'encodage rapide, équilibré ou qualité maximale ;
- découpage et conversion verticale en une seule passe d'encodage ;
- aperçu et téléchargement individuel ou groupé en ZIP ;
- un dossier distinct par traitement dans `data/projects/` ;
- sous-titres animés incrustés (optionnel) : styles, police, taille, couleur, position,
  et modes d'apparition mot actif / karaoké / mot par mot / ligne par ligne.

## Sous-titres incrustés (optionnel)

Nécessite `faster-whisper` :

```bash
python -m pip install -r requirements-transcribe.txt
```

Une fois installé, la case **Sous-titres incrustés** apparaît dans les réglages (export 9:16
uniquement). La vidéo est transcrite mot par mot une seule fois (résultat mis en cache dans
`data/transcriptions/`, `transcript.json` copié dans le dossier projet), puis un fichier `.ass`
est généré par clip et gravé dans l'image par FFmpeg.

Réglages : template de style, police, taille, position, couleur du texte, couleur du mot actif,
mode d'apparition (mot actif, karaoké, mot par mot, ligne par ligne), majuscules. Le bouton
**Aperçu du style** rend un extrait de ~4 s avec les réglages courants avant de lancer tous les clips.

Sur GPU NVIDIA, la transcription utilise CUDA (`float16`) ; sinon elle bascule sur le CPU (`int8`).

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
Par défaut, une copie agrandie et floutée de la vidéo remplit l'espace libre
derrière l'image nette. Cette composition utilise les filtres compatibles de
FFmpeg pour éviter les corruptions de couleur de certains builds Windows ;
l'encodage final reste accéléré par NVENC.

La qualité 4K télécharge la meilleure source disponible jusqu'à 2160p et
produit un export vertical 2160 × 3840. Si la source n'existe pas en 4K,
`yt-dlp` utilise la meilleure qualité inférieure disponible.

Pour vérifier les encodeurs présents sous Windows :

```powershell
ffmpeg -hide_banner -encoders | Select-String "h264_nvenc|h264_qsv|h264_amf"
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Les tests génèrent leurs propres médias avec FFmpeg et ne téléchargent aucune vidéo.
Le lint (`ruff`) et les tests tournent aussi en CI sur chaque push et pull request
(voir `.github/workflows/ci.yml`).
