import os
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api import create_router
from .database import TaskRepository
from .key_management import VideoKeyProvider, load_video_key_provider_bundle
from .service import VideoTaskService
from .platform_jobs import JobManager
from .platform_api import create_platform_router
from .security import AccessPolicy, AccessMiddleware, auth_router, validate_cloud_keys
from .rotation_control import RotationControlRepository, VideoKeyRotationWorker


def create_app(
    database_path: Optional[Path] = None,
    storage_root: Optional[Path] = None,
    encryption_key: Optional[bytes] = None,
    mac_key: Optional[bytes] = None,
    key_provider_resolver: Optional[Callable[[str], VideoKeyProvider]] = None,
    write_key_provider: Optional[VideoKeyProvider] = None,
    key_revoker: Optional[Callable[[str], str]] = None,
) -> FastAPI:
    provider_bundle = load_video_key_provider_bundle()
    if provider_bundle is not None:
        write_key_provider = write_key_provider or provider_bundle["target_provider"]
        key_provider_resolver = key_provider_resolver or provider_bundle["source_provider_resolver"]
        key_revoker = key_revoker or provider_bundle.get("revoke_key")
    data_root = Path(os.getenv("FACECHECK_DATA_ROOT", "data"))
    database = database_path or Path(
        os.getenv("FACECHECK_DATABASE", str(data_root / "facecheck.sqlite3"))
    )
    storage = storage_root or Path(
        os.getenv("FACECHECK_STORAGE_ROOT", str(data_root / "storage"))
    )
    repository = TaskRepository(database)
    policy = AccessPolicy(repository)
    service = VideoTaskService(
        repository,
        storage,
        encryption_key=encryption_key,
        mac_key=mac_key,
        key_provider_resolver=key_provider_resolver,
        write_key_provider=write_key_provider,
    )
    validate_cloud_keys(policy, service.encryption_key, service.mac_key)
    if policy.cloud:
        from .crypto_backends import get_crypto_backend
        if get_crypto_backend().name != "gmssl-native":
            raise ValueError("cloud deployment requires the GmSSL native backend")
    application = FastAPI(
        title="Facecheck",
        description="Privacy-preserving portrait video storage prototype",
        version="0.1.0",
    )
    application.state.security = policy
    application.state.video_service = service
    application.include_router(auth_router(policy))
    application.add_middleware(AccessMiddleware, policy=policy)
    if policy.cloud:
        application.add_middleware(TrustedHostMiddleware,
                                   allowed_hosts=policy.allowed_hosts + ["localhost", "127.0.0.1", "[::1]"])

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(create_router(repository, service))
    manager = JobManager(database, storage.parent / "platform_runs")
    rotation_repository = RotationControlRepository(database)
    rotation_worker = None
    if write_key_provider is not None and key_provider_resolver is not None:
        rotation_worker = VideoKeyRotationWorker(
            rotation_repository, storage,
            target_provider=write_key_provider,
            source_provider_resolver=key_provider_resolver,
            legacy_encryption_key=service.encryption_key,
            legacy_mac_key=service.mac_key,
            feature_store=service.feature_store,
        )
    application.state.platform_manager = manager
    application.state.video_rotation_repository = rotation_repository
    application.state.video_rotation_worker = rotation_worker
    application.state.video_key_revoker = key_revoker
    application.include_router(create_platform_router(
        repository, service, manager, rotation_repository=rotation_repository,
        key_revoker=key_revoker,
    ))
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    if frontend.is_dir():
        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return application


app = create_app()
