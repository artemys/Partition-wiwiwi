# TabScore (Web)

Application web pour transcrire un audio ou un lien YouTube en tablature et/ou partition.

## Structure

- `/server` : API FastAPI + worker RQ
- `/web` : front-end Next.js (App Router)

## Démarrage rapide (local)

### Frontend (dev)

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

