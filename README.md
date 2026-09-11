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
- arrière-plan : vidéo floutée, bandes noires, **recadrage qui suit le visage**
  (podcasts / interviews), ou **disposition réaction haut / bas** (facecam du
  streamer en haut, gameplay centré en bas — voir plus bas) ;
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
  français / anglais / chinois / coréen via l'IA locale — timing synthétique, tous
  les modes d'apparition (mot actif, karaoké…) restent utilisables ;
- après génération, **choix des clips** à envoyer vers un dossier au choix, rangés
  `<dossier>/<LANG>/<créateur>/<date lisible>/{clips,textes}/` — pointer vers un
  dossier Google Drive synchronisé permet de les récupérer sur le téléphone.
  Une fois copiés avec succès, le clip et son `.txt` sont **supprimés du cache
  local** pour libérer de la place (le dossier envoyé devient la référence).

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

## Disposition réaction haut / bas (optionnel)

Pour les clips de streamers qui réagissent à du contenu : l'arrière-plan
« Réaction haut / bas » découpe **deux zones dans la même vidéo source** et les
empile — **facecam** en haut, **gameplay** en bas. Sur une image de repère de la
vidéo, on place le cadre du facecam (position X/Y et taille en % de la source) ;
le gameplay est ensuite **recadré et centré automatiquement** pour remplir le
panneau du bas, sans déformation ni bandes. Réglages : cadre du facecam et
**part verticale** du panneau haut dans le clip (40 % par défaut). Un bouton
*Aperçu de la disposition* rend ~3 s pour vérifier le cadrage ; l'aperçu de style
des sous-titres montre aussi le split. Recadrage **statique** (pas de suivi) —
pensé pour un facecam fixe dans un coin.

Le facecam est un petit rectangle très agrandi : l'agrandissement se fait en
**Lanczos** avec un léger **renforcement de netteté** sur ce panneau. Pour un
résultat plus net encore, exporter en **1080p ou 4K** (la source est alors
téléchargée en meilleure définition, donc plus de pixels pour la webcam).

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

### Signaux non textuels (rires / cris / chat)

En plus du texte, deux signaux d'intensité entrent dans le score :

- **Enveloppe de volume** (`ffmpeg` + `numpy`) : les rires, cris et moments de hype
  se traduisent par des pics de volume et des transitions silence→explosion. Les
  passages intenses sont bonifiés et portent un **🔊 Pic d'intensité**. Marche
  aussi sur les fichiers importés.
- **Chat Twitch** (rediff uniquement) — **désactivé par défaut** : cocher
  **« Utiliser le chat Twitch »** dans les réglages (la case n'apparaît que sur
  un VOD Twitch), ou forcer avec **`CLIP_CREATOR_ENABLE_CHAT=1`**. Les
  commentaires du VOD sont récupérés directement depuis l'API web de Twitch
  (aucune dépendance, pas de compte) ; quand le débit de messages dépasse
  nettement sa base locale — surtout en emotes de rire (KEKW, OMEGALUL…) —
  c'est un **moment potentiellement viral** ;
  ces instants (recalés du délai de réaction du chat) **créent une fenêtre
  candidate** et portent un **⚡ Le chat s'emballe**. L'écran « moments détectés »
  liste les pics repérés (horodatage + intensité). La collecte est bornée
  (75 s d'horloge / `CLIP_CREATOR_CHAT_MAX_SECONDS`, 150 k messages) et tourne
  dans un thread **abandonné au bout de 90 s** s'il traîne (l'analyse continue
  sans le chat) — utile sur un très gros stream où le chat est énorme. Sur une
  **longue portion et un chat très dense** (mega-streamer), lire depuis le
  début n'atteindrait jamais la fin dans ce budget : le débit réel est mesuré
  d'abord, et si tout couvrir en continu ne tient pas dans le temps imparti,
  la collecte passe automatiquement à des **sondes réparties sur toute la
  portion analysée** plutôt que de tout consommer sur les premières minutes.

L'écran « moments détectés » liste les extraits classés par score ; on coche ceux à
produire et seuls ceux-là sont rendus (sous-titres compris).

**Rapidité.** L'analyse télécharge la fenêtre en **basse déf (480p)** — elle ne lit que
l'audio + des vignettes. La **génération** ne retélécharge alors **que chaque fenêtre de
clip** (± quelques secondes de marge), en pleine qualité, et **réutilise le transcript de
l'analyse** — plus de re-téléchargement + re-transcription de l'heure entière pour n'en
garder que quelques minutes. Juste avant la notation, le **modèle Whisper est déchargé
de la VRAM** pour ne
pas étouffer Ollama sur une carte de 8 Go. La notation utilise un **petit modèle** s'il
y en a un d'installé (`qwen2.5:3b`, `llama3.2:3b`, `gemma2:2b`… — bien plus rapide, et
noter la viralité n'a pas besoin d'un gros modèle) ; sinon repli sur le modèle habituel
(`qwen2.5:7b` > `qwen2.5` > `llama3.1` > `llama3` > …). `CLIP_CREATOR_RATING_MODEL`
force le modèle de notation ; `CLIP_CREATOR_LLM_MODEL` force celui de la traduction /
des titres (qui, lui, prend le plus **gros** qwen installé — `qwen2.5:7b` avant
`qwen2.5:3b`).

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

### Longues vidéos bruyantes (rediff de live)

Le curseur **« Portion à analyser »** (sélection intelligente) ou **« Portion à
clipper »** (mode régulier) limite le **téléchargement** *et* la **transcription** :
sur une rediff de 5 h (~30 Go) dont on ne garde que 30 min, yt-dlp ne télécharge
que les fragments de cette fenêtre puis seule elle est transcrite. Les horodatages
affichés restent dans le temps absolu de la source ; l'extrait téléchargé est mis
en cache par URL + qualité + fenêtre.

Sur un VOD où la voix se mêle au son du jeu / aux alertes, la transcription peut
dériver, décaler ou sauter des passages. Les réglages par défaut sont calibrés
pour ça :

- **pipeline batché** (faster-whisper) : le VAD découpe d'abord en énoncés, chacun
  transcrit indépendamment → pas de dérive qui s'accumule sur la durée ;
- `condition_on_previous_text` **désactivé** : une fenêtre ratée n'empoisonne plus
  les suivantes (fin des boucles / répétitions) ;
- **seuils assouplis** (`no_speech_threshold`, VAD) : la parole faible sous le bruit
  n'est plus jetée comme « silence » ;
- **pré-nettoyage audio** avant Whisper (`highpass` + `afftdn` + `dynaudnorm`,
  ~0 VRAM) — désactivable avec `CLIP_CREATOR_WHISPER_AUDIO_CLEAN=0`.

Variables d'environnement : `CLIP_CREATOR_WHISPER_MODEL` (ex. `large-v3` complet,
plus robuste que `large-v3-turbo` sur audio dégradé — tient sur 8 Go de VRAM en
`int8_float16` si Ollama n'est pas chargé en même temps), `CLIP_CREATOR_WHISPER_BATCH`
(taille de lot, 8 par défaut ; baisser si mémoire GPU limitée).

### Sous-titres dans une autre langue

Menu **« Langue des sous-titres »** : *Auto* (langue parlée, comportement par défaut)
ou une cible **FR / EN / ZH / KO**. Si la cible diffère de la langue parlée, seuls les
segments des clips réellement exportés sont traités (pas tout le transcript d'une
longue vidéo), puis :

- **calage sur la parole** : chaque segment Whisper est redécoupé en unités ~phrases
  via le minutage réel des mots, et chaque phrase traduite s'affiche sur sa fenêtre
  `[premier mot, dernier mot]`. Un bloc unique couvrant plusieurs phrases débordait
  sur les silences et les fragments dérivaient ; ce n'est plus le cas ;
- avant traduction : seules les **annotations non parlées connues** (`[Music]`,
  `(rires)`, `♪`…) sont retirées — un aparté du streamer entre parenthèses est
  conservé — les **micro-unités collées** sont fusionnées, et un **mot isolé
  court et plat** (« saint », « the »… — un bout de transcription attrapé sur du
  bruit) est écarté (mais « Quoi ?! », « Wow », « Non » restent) ;
- consigne au modèle : **fidélité avant tout** — garder tous les éléments concrets
  (noms, marques, lieux, chiffres, titres de jeux), ne raccourcir que les
  hésitations ; garder un terme d'argot / une expression anglaise tel quel s'il n'a
  pas d'équivalent courant. La durée à l'écran et la réplique précédente sont
  fournies en contexte ;
- un lot mal répondu est retenté par plus petits lots, puis segment par segment,
  avant d'être laissé en VO (un modèle local renvoie parfois un tableau JSON
  incomplet) ;
- un fichier **`translation.<langue>.txt`** est écrit dans le dossier du projet :
  `[m:ss] VO -> traduction` par phrase, pour vérifier à l'œil ce que Whisper a
  entendu et comment ça a été traduit.

Le texte traduit n'a pas de minutage mot à mot (l'audio est dans une autre langue) :
affichage **en bloc** par unité, sans mode mot par mot / karaoké. Pour le chinois /
coréen / japonais, une police à glyphes adaptés est imposée (*Microsoft YaHei* /
*Malgun Gothic* / *Yu Gothic*, livrées avec Windows). Nécessite Ollama.

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

## Dossier de données hors OneDrive / Drive (recommandé)

Si le dépôt se trouve dans un dossier synchronisé (OneDrive, Google Drive…),
la synchronisation peut verrouiller un clip ou l'archive ZIP juste après leur
écriture et faire échouer un téléchargement (`OSError: [Errno 22]`). L'appli
retente automatiquement puis affiche un message clair si ça persiste, mais le
plus fiable est de sortir `data/` du dossier synchronisé :

```powershell
[Environment]::SetEnvironmentVariable("CLIP_CREATOR_DATA_DIR", "C:\ClipCreatorData", "User")
```

Relancer un nouveau terminal (ou `start.bat`) pour que la variable s'applique.

## Tests

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Les tests génèrent leurs propres médias avec FFmpeg et ne téléchargent aucune vidéo.
Le lint (`ruff`) et les tests tournent aussi en CI sur chaque push et pull request
(voir `.github/workflows/ci.yml`).
