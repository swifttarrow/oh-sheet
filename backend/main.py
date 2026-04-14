"""FastAPI application factory and uvicorn entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import artifacts, capabilities, health, jobs, stages, uploads, ws
from backend.config import settings
from backend.contracts import SCHEMA_VERSION

# Flutter web build output — present in the Docker image at /app/static.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _configure_app_logging() -> None:
    """Attach a handler to ``backend.*`` so INFO logs show up in the server console.

    Python's root logger defaults to WARNING; Uvicorn does not raise it for
    application loggers. We scope to the ``backend`` package so library noise
    stays at default levels unless you tune those loggers separately.
    """
    level = getattr(logging, settings.log_level.upper(), None)
    if not isinstance(level, int):
        level = logging.INFO
    backend_logger = logging.getLogger("backend")
    backend_logger.setLevel(level)
    if not backend_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s [%(name)s] %(message)s"),
        )
        backend_logger.addHandler(handler)
    backend_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_app_logging()
    settings.blob_root.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Oh Sheet — Pipeline API",
        description="REST + WebSocket API for the Song→Humanized-Piano-Sheet-Music pipeline.",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/v1", tags=["health"])
    app.include_router(uploads.router, prefix="/v1", tags=["uploads"])
    app.include_router(jobs.router, prefix="/v1", tags=["jobs"])
    app.include_router(artifacts.router, prefix="/v1", tags=["artifacts"])
    app.include_router(stages.router, prefix="/v1", tags=["stages"])
    app.include_router(ws.router, prefix="/v1", tags=["websocket"])
    app.include_router(capabilities.router, prefix="/v1", tags=["capabilities"])

    # IMPORTANT: mount AFTER API routers — StaticFiles at "/" is a catch-all.
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app


app = create_app()


def run() -> None:
    """Entry point for the ``ohsheet`` console script."""
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
