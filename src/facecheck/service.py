from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .database import TaskRepository
from .storage import StorageError, store_encrypted_video
from .video import VideoProcessingError, read_video_info

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
KEY_VERSION = "api-video-key-v1"
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def clean_filename(filename: Optional[str]) -> str:
    value = Path(filename or "upload.bin").name
    value = SAFE_FILENAME.sub("_", value).strip("._")
    return value[:255] or "upload.bin"


def clean_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").replace("\r", " ").strip()
    message = re.sub(r"[/\\][^\s]+", "<path>", message)
    return message[:500] or "processing failed"


class VideoTaskService:
    def __init__(
        self,
        repository: TaskRepository,
        storage_root: Path,
        *,
        encryption_key: Optional[bytes] = None,
        mac_key: Optional[bytes] = None,
    ) -> None:
        self.repository = repository
        self.storage_root = storage_root
        self.encryption_key = encryption_key or self._key_from_env("FACECHECK_SM4_KEY")
        self.mac_key = mac_key or self._key_from_env("FACECHECK_MAC_KEY")

    @staticmethod
    def _key_from_env(name: str) -> Optional[bytes]:
        value = os.getenv(name)
        return value.encode("utf-8") if value else None

    def create_and_process(
        self, source: Path, original_filename: str, source_size: int
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        self.repository.create_task(task_id, clean_filename(original_filename), source_size)
        try:
            self.repository.transition(task_id, "processing")
            if self.encryption_key is None or self.mac_key is None:
                raise StorageError("encryption configuration is unavailable")
            read_video_info(source)
            stored = store_encrypted_video(
                source,
                self.storage_root,
                self.encryption_key,
                self.mac_key,
                chunk_size=CHUNK_SIZE,
                key_version=KEY_VERSION,
                video_id=task_id,
            )
            self.repository.transition(task_id, "completed", storage_id=stored.video_id)
        except Exception as error:
            self.repository.transition(task_id, "failed", error_message=clean_error(error))
        task = self.repository.get(task_id)
        if task is None:
            raise RuntimeError("created task disappeared")
        return task
