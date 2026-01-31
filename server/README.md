# Backend TabScore (FastAPI + RQ)

Génération réelle de tablatures à partir d’audio ou de liens YouTube.

## Démarrage local

### Prérequis système

- Python 3.10+
- ffmpeg + ffprobe
- Redis (local ou Docker)
- yt-dlp (pour YouTube)
- demucs + basic-pitch
- music21 (optionnel pour MusicXML)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### Lancer l’API + worker

```bash
export TABSERVER_REDIS_URL=redis://localhost:6379/0
export TABSERVER_PUBLIC_URL=http://localhost:8000
uvicorn server.main:app --reload --port 8000
```

Dans un second terminal :

```bash
python -m server.worker
```

## Docker

```bash
cd server
docker compose up --build
```

## Script de test

```bash
export TEST_YT_URL="https://www.youtube.com/watch?v=..."
export TEST_AUDIO_FILE="/chemin/vers/audio.mp3"
./server/scripts/dev_test.sh
```

## Endpoints REST

- `POST /jobs` (multipart `audio` ou JSON `{ youtubeUrl }`)
  - Query: `outputType=tab|score|both`, `tuning=EADGBE`, `capo=0`, `quality=fast|accurate`, `mode`, `target=GUITAR_BEST_EFFORT`, `inputIsIsolatedGuitar=false`, `startSeconds`, `endSeconds`
- `GET /jobs/{jobId}`
- `GET /jobs/{jobId}/result`
- `DELETE /jobs/{jobId}`
- `GET /files/{jobId}/{fileName}`
- `GET /health`

## Pipeline (v1)

1. Validation + limites (taille 50MB, durée 12 min).
2. Conversion WAV mono 22.05kHz.
3. Isolation Demucs (sauf piste isolée).
4. Basic Pitch → MIDI.
5. Post-traitement notes + tablature.
6. Export tab.txt / tab.json / MIDI / MusicXML (optionnel).

## Stockage

Fichiers dans `server/data/{jobId}/` :

- `input/` : fichiers bruts + wav
- `output/` : `tab.txt`, `tab.json`, `output.mid`, `score.musicxml` (si dispo)
- `logs.txt` : logs détaillés par job

## Erreurs

Aucun résultat par défaut en cas d’échec.
En cas de problème : `status=FAILED` + `errorMessage` clair + logs.
