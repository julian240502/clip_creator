Date début de projet : 15/01/2025

# Clip Creator

installer ffmpeg avec winget et pip
```bash
winget install ffmpeg 
```
## Description
Clip Creator est un outil pour créer des clips vidéo à partir de séquences brutes. Il permet de découper, éditer et assembler des vidéos facilement et rapidement.

## Fonctionnalités
- Téléchargement de vidéos [OK]
- Sous-titre des vidéos [OK]
- Découpage en Clips pour des formats courts [OK]
- Optimisation pour TikTok/Shorts : [1/2]<br/> 
    - Redimensionnement [OK]<br/> 
    - Effet visuel de sous-titres / Gameplay(sous la video) [en cours]
## Next step
- Génération de titres et de hashtag
- Automatisation de Publication sur les plateformes.
- Analyse des Performances pour des itérations futures.
- Exportation en différents formats 

## Framework utilisé
- yt-dlp : pour le download <br/>
- ffmpeg : ajouts de sous-ttire 

## Installation
Clonez le dépôt et installez les dépendances :
```bash
git clone https://github.com/votre-utilisateur/clip_creator.git
cd clip_creator
npm install
```

## Utilisation
Pour utiliser Clip Creator, exécutez la commande suivante :
```bash
npm start
```

## Licence
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteurs
- Julian AKUESON - Créateur et Mainteneur Principal

