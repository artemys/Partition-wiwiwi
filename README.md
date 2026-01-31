# TabScore (Web)

Application web pour transcrire un audio ou un lien YouTube en tablature et/ou partition.

## Structure

- `/server` : API FastAPI + worker RQ
- `/web` : front-end Next.js (App Router)
- `/android_legacy` : archive de l’ancienne app Android

## Démarrage rapide (local)

### Backend

Voir `server/README.md` pour les prérequis (ffmpeg, Redis, MuseScore, etc.).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
export TABSERVER_PUBLIC_URL=http://localhost:8000
uvicorn server.main:app --reload --port 8000
```

Dans un second terminal :

```bash
python -m server.worker
```

### Frontend

```bash
cd web
pnpm install
pnpm dev
```

Ouvrir `http://localhost:3000`.

## Docker (API + Web)

```bash
cp .env.example .env
docker compose up --build
```

## Variables d’environnement

Voir `.env.example` pour la liste complète.

## Android legacy

Le projet Android d’origine est archivé dans `/android_legacy`.
