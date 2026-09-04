# Clip Creator ✦

Application locale pour télécharger ou importer une vidéo, la découper en clips et produire des exports verticaux prêts pour **CapCut** ou **Camtasia**.

Le flux se fait en deux temps : on **charge et prévisualise** la vidéo (lecteur intégré, titre, durée, résolution), on **choisit la portion à traiter** et les réglages, puis on **génère les clips**, affichés au fur et à mesure avec une vignette et un aperçu à la demande.

## Fonctionnalités

- téléchargement via une URL prise en charge par `yt-dlp` ;
- import de fichiers MP4, MOV, MKV et WebM ;
- prévisualisation de la source avant traitement, avec métadonnées ;
- sélection d'une portion de la vidéo à clipper et estimation du nombre de clips ;
- découpage précis avec une durée configurable ;
- choix du format d'export : 9:16, 4:5, 1:1, 16:9 ou format d'origine, en 720p / 1080p / 4K ;
- la vidéo source tient entièrement dans le cadre, sans déformation ;
- arrière-plan : vidéo floutée, bandes noires, ou **recadrage qui suit le visage**
  (podcasts / interviews — optionnel, voir plus bas) ;
- accélération matérielle automatique NVIDIA NVENC, Intel Quick Sync ou AMD AMF ;
- redimensionnement CUDA sur NVIDIA pour le recadrage vertical ;
- profils d'encodage rapide, équilibré ou qualité maximale ;
- découpage et conversion verticale en une seule passe d'encodage ;
- aperçu et téléchargement individuel ou groupé en ZIP ;
- un dossier distinct par traitement dans `data/projects/` ;
- sous-titres animés incrustés (optionnel) : styles, police, taille, couleur, position,
  et modes d'apparition mot actif / karaoké / mot par mot / ligne par ligne ;
- deux modes de découpage : **régulier** (durée fixe) et **sélection intelligente**
  (repère les extraits au plus fort potentiel viral, les note et les classe) ;
- titres, descriptions et hashtags générés par clip (optionnel), dans un `.txt`
  joint à chaque vidéo et à l'archive ZIP — dans la langue de la vidéo ;
- **sous-titres traduits** (optionnel) : garder la langue parlée, ou traduire en
  français / anglais / chinois via l'IA locale — le texte traduit s'affiche en
  lignes (pas d'animation mot à mot) ;
- après génération, **choix des clips** à envoyer vers un dossier au choix, rangés
  `<dossier>/<LANG>/<créateur>/<date lisible>/{clips,textes}/` — pointer vers un
  dossier Google Drive synchronisé permet de les récupérer sur le téléphone.

## Recadrage sur le visage (optionnel)

Pour les plans fixes de personne(s) qui parlent (podcasts, interviews), l'arrière-plan
« Recadrage sur le visage » remplace le letterbox/flou par un **recadrage plein cadre qui
suit le visage principal**. Nécessite OpenCV :

```bash
python -m pip install -r requirements-reframe.txt
```

Avant le rendu, une passe d'analyse (~5 s pour 10 min — seules les images-clés sont
décodées) détecte le visage frontal le plus grand (Haar / OpenCV, CPU, multi-thread),
rééchantillonne et **lisse** la trajectoire (fenêtre glissante ~1,2 s + limite de vitesse),
puis pilote le filtre `crop` via une **expression `x(t)` continue** (évaluée à chaque image,
pas de saccades). Sans OpenCV, ou si aucun visage n'est détecté, on retombe sur un
recadrage centré.

## Sélection intelligente (optionnel)

Nécessite `faster-whisper` (voir plus bas) et, pour la notation, **Ollama** :

```bash
ollama serve
ollama pull qwen2.5:7b   # ou llama3 / mistral déjà présents
```

Choisir « Sélection intelligente » dans les réglages, régler le nombre de clips visés
et la durée cible, puis **Analyser les moments** : la vidéo est transcrite, recomposée en
phrases (ponctuation, pauses, capitales de Whisper) puis découpée en fenêtres candidates
calées sur ces frontières — un clip ne démarre jamais en plein milieu d'une phrase ni sur
un connecteur suspendu (« Et donc… »). Les fenêtres sont pré-filtrées par heuristiques, puis chaque
finaliste reçoit un **score de viralité /100**, un **titre**, un **résumé** et une
**justification** via Ollama (repli sur une notation heuristique si Ollama n'est pas lancé).
Chaque extrait est aussi jaugé sur son **accroche** (les toutes premières secondes) : ceux
qui ouvrent fort portent un badge **⚡ Accroche forte** et affichent la phrase d'accroche.
Le hook ne filtre rien et ne change pas le classement — il ne fait que mettre en avant.

L'écran « moments détectés » liste les extraits classés par score ; on coche ceux à
produire et seuls ceux-là sont rendus (sous-titres compris). Le modèle Ollama est choisi
automatiquement (`qwen2.5` > `llama3.1` > `llama3` > `mistral` > …).

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

### Sous-titres dans une autre langue

Menu **« Langue des sous-titres »** : *Auto* (langue parlée, comportement par défaut)
ou une cible **FR / EN / ZH**. Si la cible diffère de la langue parlée, la vidéo est
transcrite normalement puis chaque segment est **traduit par Ollama** (mis en cache).
Le texte traduit n'a pas de timing mot à mot : les sous-titres passent alors en mode
**lignes** (une ligne par segment, pas d'effet karaoké / mot actif). En chinois, la
police *Microsoft YaHei* est imposée pour les glyphes CJK. Nécessite Ollama.

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

## Lancement rapide sous Windows

Le dossier `windows/` contient des lanceurs qui évitent de retaper les commandes
ci-dessus à chaque fois (nécessite que `.venv` existe déjà, voir *Installation*) :

- **`windows/start.bat`** — double-clic pour lancer l'app (fenêtre visible,
  pratique pour voir les erreurs). Créez-en un raccourci sur le bureau
  (clic droit → *Créer un raccourci*) ; vous pouvez lui donner une icône
  personnalisée via les propriétés du raccourci.
- **`windows/start_silent.vbs`** — identique mais sans fenêtre de console.
  Pour un lancement automatique à la connexion : appuyez sur `Win+R`, tapez
  `shell:startup`, puis déposez-y un raccourci vers ce fichier.
- **`windows/stop.bat`** — arrête le serveur lancé en silencieux (le VBS ne
  laissant pas de fenêtre à fermer, cherchez et tue le processus sur le port
  `8501`).

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

## URL YouTube : « Sign in to confirm you're not a bot »

YouTube exige de plus en plus une authentification pour télécharger. Avant de
lancer l'app, indiquer le navigateur connecté à YouTube dont `yt-dlp` doit lire
les cookies :

```powershell
$env:CLIP_CREATOR_YTDLP_COOKIES_BROWSER = "chrome"   # ou firefox, edge, brave
# profil précis : "chrome:Profile 1"
```

Autres variables : `CLIP_CREATOR_YTDLP_COOKIES_FILE` (chemin d'un `cookies.txt`
exporté) et `CLIP_CREATOR_YTDLP_PLAYER_CLIENT` (ex. `android,web`, dépannage).
Le mode avancé (barre latérale) affiche si des cookies sont configurés.

## Tests

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Les tests génèrent leurs propres médias avec FFmpeg et ne téléchargent aucune vidéo.
Le lint (`ruff`) et les tests tournent aussi en CI sur chaque push et pull request
(voir `.github/workflows/ci.yml`).
