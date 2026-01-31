from fastapi import FastAPI, Request


def install_auth_middleware(app: FastAPI) -> None:
    """
    Middleware placeholder for future auth.
    Currently allows all requests (local mode).
    """

    @app.middleware("http")
    async def optional_auth(request: Request, call_next):  # type: ignore[override]
        return await call_next(request)
