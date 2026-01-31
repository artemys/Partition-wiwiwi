# Backend TabScore (FastAPI + RQ)

Génération réelle de tablatures à partir d’audio ou de liens YouTube.

## Démarrage local

### Prérequis système

- Python 3.10+
- ffmpeg + ffprobe
- Redis (local ou Docker)
- yt-dlp (pour YouTube)
- demucs + basic-pitch
- MuseScore CLI (mscore) pour le rendu PDF

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

## Rendu PDF (MusicXML -> PDF)

```bash
./server/scripts/render_pdf.sh /chemin/vers/result.musicxml /chemin/vers/result.pdf
```

## Endpoints REST

- `POST /jobs` (multipart `audio` ou JSON `{ youtubeUrl }`)
  - Query: `outputType=tab|score|both`, `tuning=EADGBE`, `capo=0`, `quality=fast|accurate`, `mode`, `target=GUITAR_BEST_EFFORT`, `inputIsIsolatedGuitar=false`, `startSeconds`, `endSeconds`
- `GET /jobs/{jobId}`
- `GET /jobs/{jobId}/result`
- `DELETE /jobs/{jobId}`
- `GET /files/{jobId}/{fileName}`
- `GET /health`
- `GET /test/render-pdf` (rend un PDF depuis un MusicXML exemple)

## Pipeline (v1)

1. Validation + limites (taille 50MB, durée 12 min).
2. Conversion WAV mono 22.05kHz.
3. Isolation Demucs (sauf piste isolée).
4. Basic Pitch → MIDI.
5. Post-traitement notes + tablature.
6. Export tab.txt / tab.json / MIDI / MusicXML (result.musicxml) / PDF (result.pdf).

## Stockage

Fichiers dans `server/data/{jobId}/` :

- `input/` : fichiers bruts + wav
- `output/` : `tab.txt`, `tab.json`, `output.mid`, `result.musicxml`, `result.pdf`
- `logs.txt` : logs détaillés par job

## Erreurs

Aucun résultat par défaut en cas d’échec.
En cas de problème : `status=FAILED` + `errorMessage` clair + logs.

## Résultat PDF

`GET /jobs/{jobId}/result` retourne `pdfUrl` (obligatoire) et `musicXmlUrl` (optionnel).
Le fichier PDF est disponible via `/files/{jobId}/result.pdf`.
