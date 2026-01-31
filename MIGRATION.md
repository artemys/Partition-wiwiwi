# Migration Android -> Web

## Ce qui existe (actuel)

### Backend (/server)
- API FastAPI avec jobs asynchrones via RQ + Redis.
- Endpoints existants :
  - `POST /jobs` (multipart `audio` OU JSON `{ youtubeUrl }`)
  - `GET /jobs/{jobId}`
  - `GET /jobs/{jobId}/result`
  - `DELETE /jobs/{jobId}`
  - `GET /files/{jobId}/{fileName}`
  - `GET /health`, `GET /test/render-pdf`
- Modèle de données : table `jobs` (SQLite par défaut).
  - Champs clés : `status`, `stage`, `progress`, `error_message`, `confidence`, `created_at`
  - Entrée : `input_type`, `source_url`, `input_path`, `input_filename`
  - Sorties : `tab_txt_path`, `tab_json_path`, `musicxml_path`, `pdf_path`, `midi_path`
- Fichiers persistés dans `server/data/{jobId}/` (input/output/logs).
- Validation : formats audio, taille max, URL YouTube.

### Android (supprimé)
- L’app Android et sa logique locale ont été retirées du repo.

## Ce qui change (cible)

### Architecture monorepo
- `/server` conservé et stabilisé (contrat API + persistance DB).
- `/web` ajouté : app web Next.js (App Router, TS).
- L’app Android n’est plus présente dans le repo.

### Contrat API à stabiliser
- `POST /jobs` : upload multipart OU `{ youtubeUrl }`
  - params : `outputType`, `tuning`, `capo`, `quality`
  - réponse : `{ jobId }`
- `GET /jobs/{jobId}` : `{ status, stage, progress, errorMessage?, confidence?, createdAt }`
- `GET /jobs/{jobId}/result` : `{ pdfUrl?, musicXmlUrl?, tabTxtUrl?, tabJsonUrl?, midiUrl? }`
- `GET /library` : liste paginée des jobs terminés (DONE/FAILED) + metadata
- `DELETE /jobs/{jobId}` : supprime job + fichiers
- `GET /files/...` : URLs stables pour téléchargement

### Frontend web (nouveau)
- Pages : Library, New, Details, Settings.
- Data fetching : fetch + React Query (ou SWR).
- Form : React Hook Form + Zod.
- Upload : multipart/form-data.
- UI en français, code en anglais.

### Persistance
- Source de vérité backend (DB).
- Pas de persistance côté frontend (state local uniquement).
