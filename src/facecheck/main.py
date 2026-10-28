import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .api import create_router
from .database import TaskRepository
from .service import VideoTaskService


def create_app(
    database_path: Optional[Path] = None,
    storage_root: Optional[Path] = None,
    encryption_key: Optional[bytes] = None,
    mac_key: Optional[bytes] = None,
) -> FastAPI:
    data_root = Path(os.getenv("FACECHECK_DATA_ROOT", "data"))
    database = database_path or Path(
        os.getenv("FACECHECK_DATABASE", str(data_root / "facecheck.sqlite3"))
    )
    storage = storage_root or Path(
        os.getenv("FACECHECK_STORAGE_ROOT", str(data_root / "storage"))
    )
    repository = TaskRepository(database)
    service = VideoTaskService(
        repository,
        storage,
        encryption_key=encryption_key,
        mac_key=mac_key,
    )
    application = FastAPI(
        title="Facecheck",
        description="Privacy-preserving portrait video storage prototype",
        version="0.1.0",
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(create_router(repository, service))
    return application


app = create_app()
