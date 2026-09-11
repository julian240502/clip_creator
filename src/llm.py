"""Client Ollama minimal (bibliothèque standard, aucune dépendance).

Sert à noter/résumer les extraits en mode « sélection intelligente ». Si Ollama
n'est pas lancé, l'appelant retombe sur une notation heuristique.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
# Ordre de préférence pour traduction / titres : gros modèles en tête (finesse
# FR + suivi de consignes). Les variantes explicites passent AVANT `qwen2.5`
# générique, sinon Ollama peut renvoyer `qwen2.5:3b` avant `qwen2.5:7b`.
_MODEL_PREFERENCE = (
    "qwen2.5:72b", "qwen2.5:32b", "qwen2.5:14b", "qwen2.5:7b", "qwen2.5",
    "llama3.1", "llama3", "mistral", "gemma2", "phi3", "llama2",
)
# Notation des extraits : score + titre court -> un petit modèle suffit et tient
# en VRAM à côté de Whisper (bien plus rapide sur carte serrée).
_RATING_PREFERENCE = (
    "qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b", "llama3.2",
    "gemma2:2b", "phi3.5", "phi3:mini", "qwen2.5:0.5b",
)

# Nom (en français) des langues renvoyées par Whisper — pour cadrer la langue de
# sortie des prompts (titres, résumés, hashtags dans la langue de la vidéo).
_LANGUAGE_NAMES = {
    "en": "anglais", "fr": "français", "es": "espagnol", "de": "allemand",
    "it": "italien", "pt": "portugais", "nl": "néerlandais", "pl": "polonais",
    "ru": "russe", "uk": "ukrainien", "tr": "turc", "ar": "arabe", "hi": "hindi",
    "ja": "japonais", "ko": "coréen", "zh": "chinois", "ro": "roumain",
    "sv": "suédois", "da": "danois", "no": "norvégien", "fi": "finnois",
    "id": "indonésien", "vi": "vietnamien", "cs": "tchèque", "el": "grec",
}


def language_name(code: str | None) -> str:
    """Nom lisible d'une langue ISO (« en » → « anglais »).

    Repli neutre quand la langue est inconnue ou absente, pour que le prompt
    dise quand même « garde la langue de la transcription ».
    """
    key = (code or "").strip().lower().replace("_", "-").split("-")[0]
    return _LANGUAGE_NAMES.get(key, "la langue de la transcription")


def ollama_available(host: str = DEFAULT_HOST, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models(host: str = DEFAULT_HOST, timeout: float = 3.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []
    return [model.get("name", "") for model in data.get("models", []) if model.get("name")]


def pick_model(host: str = DEFAULT_HOST) -> str | None:
    """Modèle pour la **traduction / les titres** : `CLIP_CREATOR_LLM_MODEL` force
    le choix, sinon on suit l'ordre de préférence (variantes précises d'abord,
    puis générique)."""
    forced = os.environ.get("CLIP_CREATOR_LLM_MODEL", "").strip()
    if forced:
        return forced
    models = list_models(host)
    if not models:
        return None
    for wanted in _MODEL_PREFERENCE:
        for name in models:
            if name.split(":")[0] == wanted or name.startswith(wanted):
                return name
    return models[0]


def pick_rating_model(host: str = DEFAULT_HOST) -> str | None:
    """Modèle pour la **notation** des extraits : préfère un petit modèle (rapide,
    tient à côté de Whisper). `CLIP_CREATOR_RATING_MODEL` force le choix ; sinon
    repli sur `pick_model()` si aucun petit modèle n'est installé."""
    forced = os.environ.get("CLIP_CREATOR_RATING_MODEL", "").strip()
    if forced:
        return forced
    models = list_models(host)
    if not models:
        return None
    for wanted in _RATING_PREFERENCE:
        for name in models:
            if name.startswith(wanted):
                return name
    return pick_model(host)


def prewarm(model: str | None = None, host: str = DEFAULT_HOST) -> None:
    """Charge le modèle en VRAM et le garde résident. À lancer dans un thread."""
    name = model or pick_model(host)
    if not name:
        return
    payload = {"model": name, "prompt": "ok", "stream": False, "keep_alive": "30m"}
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180):
            pass
    except (urllib.error.URLError, OSError):
        pass


def chat_json(
    system: str,
    user: str,
    *,
    model: str,
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
    num_ctx: int | None = None,
) -> dict:
    """Un appel /api/chat en mode JSON. Lève une exception en cas d'échec.

    `num_ctx` (tokens) force la taille de contexte Ollama pour CET appel. Sans
    ça, Ollama retombe sur le défaut du modèle (souvent 2048-4096 tokens) et
    **tronque silencieusement** un prompt trop long — pas d'erreur, juste une
    notation faite sur un texte incomplet. À fixer explicitement dès que le
    texte envoyé (extraits longs, plusieurs par lot) peut dépasser ce défaut.
    """
    options: dict = {"temperature": 0.2}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": options,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    content = body.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Réponse Ollama vide.")
    return json.loads(content)
