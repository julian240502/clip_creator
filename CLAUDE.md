# Clip Creator — notes pour Claude

Outil local qui transforme une vidéo (URL yt-dlp ou fichier) en clips verticaux
sous-titrés. Interface Streamlit : `app.py`. Moteur : `src/`. Pilotage sans
interface : `cli.py` (JSON sur stdout, progression sur stderr).

## Fabriquer des clips

1. **Analyser** (télécharge en 480p, transcrit, repère les moments) :
   `python cli.py analyze <url|fichier> [--start H:MM:SS --end H:MM:SS] [--chat] [--count 10]`
   - longue rediff : toujours borner avec `--start/--end` ;
   - `--chat` seulement sur un VOD Twitch ;
   - `--no-llm` si tu choisis toi-même les moments (plus rapide, pas d'Ollama).
2. **Choisir** : lire `analysis.json` (extraits notés, `loud_moments` = pics de
   volume, `chat_spikes` = chat qui s'emballe) et surtout `transcript.txt`.
   Ne pas se limiter aux extraits proposés : un bon clip
   - démarre sur une phrase complète, idéalement l'accroche, sans contexte manquant ;
   - finit juste après la chute ou la réaction, pas au milieu d'une phrase ;
   - dure en général 20 à 60 s.
   En cas de doute visuel : `python cli.py frames <source> --window DEBUT-FIN`
   puis regarder les images.
3. **Écrire `choix.json`** : `[{"start", "end", "title", "description", "hashtags"}]`,
   dans la langue de la vidéo. Titre court et accrocheur, sans spoiler la chute.
4. **Rendre** :
   `python cli.py render --analysis <analysis.json> --clips choix.json --captions active`
   (+ `--publish <dossier> --label <créateur>` si un dossier d'envoi est donné).
5. **Vérifier** : extraire une image de chaque clip produit (ffmpeg) et
   contrôler le cadrage et les sous-titres avant d'annoncer le résultat.

Ne jamais publier sur un réseau social : livrer les fichiers, l'utilisateur
valide et poste lui-même. Attention aux droits sur le contenu d'autres créateurs.

## Développement

- Tests : `pytest` (médias générés par FFmpeg, aucun téléchargement) ; lint : `ruff check .`
- Messages et commentaires en français.
- Les données vont dans `data/` (ou `CLIP_CREATOR_DATA_DIR`).
