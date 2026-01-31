# TabScore

Application Android (Kotlin/Compose) pour transcrire un audio ou un lien YouTube en partition ou tablature.

## Démarrage Android

1. Ouvrir le projet dans Android Studio.
2. Lancer l’app sur un émulateur ou un device.

### Configuration backend

Dans `app/build.gradle.kts` :

- `USE_FAKE_BACKEND = true` pour l’API mockée locale.
- `BASE_URL = "http://10.0.2.2:8000/"` pour pointer le backend `server/` (émulateur).

## Tests

- Unit tests: `./gradlew test`
- Tests instrumentés: `./gradlew connectedAndroidTest`

## Backend (Option B)

Voir `server/README.md` pour démarrer FastAPI.

## Exports

Les fichiers générés sont enregistrés dans le répertoire de l’app :
`Android/data/com.tabscore.app/files/Download/`

## Limites connues

- Rendu MusicXML affiché sous forme de texte (pas de rendu graphique).
- Pas d’extraction audio côté app pour YouTube (délégué au backend).
- Progression simulée pour la version mockée.
- Options guitare (accordage/capo/mode) gérées côté app et transmises au backend.
