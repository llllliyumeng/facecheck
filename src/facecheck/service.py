from __future__ import annotations

import os
import re
import shutil
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .database import TaskRepository
from .face_detection import FaceDetection, FaceDetectionConfig, load_yunet_detector
from .face_embedding import SFaceConfig, extract_embedding
from .face_search import FaceSearchConfig, FaceSearchError, search_encrypted_video
from .storage import StorageError, store_encrypted_video
from .video import VideoProcessingError, read_video_info

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
MAX_QUERY_IMAGE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
KEY_VERSION = "api-video-key-v1"
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


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
        self.last_search_metrics: Dict[str, Any] = {}
        self.encryption_key = encryption_key or self._key_from_env("FACECHECK_SM4_KEY")
        self.mac_key = mac_key or self._key_from_env("FACECHECK_MAC_KEY")

    @staticmethod
    def _path_from_env(name: str) -> Optional[Path]:
        value = os.getenv(name)
        return Path(value) if value else None

    @staticmethod
    def _key_from_env(name: str) -> Optional[bytes]:
        value = os.getenv(name)
        return value.encode("utf-8") if value else None

    @staticmethod
    def _remove_storage_directory(directory: Path) -> None:
        if directory.is_symlink():
            directory.unlink()
        elif directory.is_dir():
            shutil.rmtree(directory)

    def create_and_process(
        self,
        source: Path,
        original_filename: str,
        source_size: int,
        *,
        upload_receive_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        started = time.perf_counter()
        metrics: Dict[str, Any] = {"upload_bytes": source_size}
        if upload_receive_seconds is not None:
            metrics["upload_receive_seconds"] = round(upload_receive_seconds, 6)
        self.repository.create_task(task_id, clean_filename(original_filename), source_size)
        try:
            video_started = time.perf_counter()
            self.repository.transition(task_id, "processing")
            if self.encryption_key is None or self.mac_key is None:
                raise StorageError("encryption configuration is unavailable")
            read_video_info(source)
            metrics["video_read_seconds"] = round(time.perf_counter() - video_started, 6)
            encryption_started = time.perf_counter()
            stored = store_encrypted_video(
                source,
                self.storage_root,
                self.encryption_key,
                self.mac_key,
                chunk_size=CHUNK_SIZE,
                key_version=KEY_VERSION,
                video_id=task_id,
                stats=metrics,
            )
            metrics["encryption_storage_seconds"] = round(
                time.perf_counter() - encryption_started, 6
            )
            metrics["chunk_count"] = stored.manifest.get("chunk_count", 0)
            metrics["key_version"] = stored.manifest.get("key_version")
            try:
                self.repository.transition(
                    task_id, "completed", storage_id=stored.video_id
                )
            except Exception:
                self._remove_storage_directory(stored.directory)
                raise
            metrics["total_seconds"] = round(time.perf_counter() - started, 6)
            self.repository.update_metrics(task_id, metrics)
        except Exception as error:
            metrics["total_seconds"] = round(time.perf_counter() - started, 6)
            try:
                self.repository.update_metrics(task_id, metrics)
            except Exception:
                pass
            try:
                self.repository.transition(
                    task_id, "failed", error_message=clean_error(error)
                )
            except Exception as status_error:
                raise RuntimeError("task status could not be updated") from status_error
        task = self.repository.get(task_id)
        if task is None:
            raise RuntimeError("created task disappeared")
        return task

    def search_task(
        self,
        task_id: str,
        query_image: bytes,
        *,
        top_k: int,
        min_similarity: Optional[float],
    ) -> list[Dict[str, Any]]:
        if not SAFE_ID.fullmatch(task_id):
            raise StorageError("invalid task identifier")
        if len(query_image) > MAX_QUERY_IMAGE_SIZE:
            raise StorageError("query image exceeds the size limit")
        if self.encryption_key is None or self.mac_key is None:
            raise StorageError("encryption configuration is unavailable")
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["status"] != "completed" or not task.get("storage_id"):
            raise StorageError("video task is not completed")
        storage_id = task["storage_id"]
        if not isinstance(storage_id, str) or not SAFE_ID.fullmatch(storage_id):
            raise StorageError("stored video identifier is invalid")
        storage_directory = self.storage_root / storage_id
        if not storage_directory.is_dir():
            raise StorageError("stored video is unavailable")

        started = time.perf_counter()
        image_array = np.frombuffer(query_image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("query file is not a readable image")
        query_decode_seconds = time.perf_counter() - started
        yunet_model = self._path_from_env("FACECHECK_YUNET_MODEL")
        sface_model = self._path_from_env("FACECHECK_SFACE_MODEL")
        if yunet_model is None or sface_model is None:
            raise StorageError("face model configuration is unavailable")
        detection_config = FaceDetectionConfig(
            model_path=yunet_model,
            frame_interval_seconds=float(
                os.getenv("FACECHECK_SEARCH_INTERVAL", "1.0")
            ),
        )
        embedding_config = SFaceConfig(model_path=sface_model)
        detector = load_yunet_detector(detection_config)
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, raw_faces = detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            raise ValueError("no face found in query image")
        face = np.asarray(raw_faces[0], dtype=np.float32)
        if face.ndim != 1 or face.shape[0] < 15:
            raise ValueError("face detector returned invalid query data")
        detection = FaceDetection(
            frame_index=0,
            timestamp_seconds=0.0,
            video_width=image.shape[1],
            video_height=image.shape[0],
            x=float(face[0]),
            y=float(face[1]),
            width=float(face[2]),
            height=float(face[3]),
            confidence=float(face[14]),
            landmarks=tuple(
                (float(face[index]), float(face[index + 1]))
                for index in range(4, 14, 2)
            ),
        )
        query_vector = extract_embedding(
            image, detection, embedding_config
        ).vector
        config = FaceSearchConfig(
            detection=detection_config,
            embedding=embedding_config,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        temporary_root = self.storage_root.parent / "search-temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        metrics: Dict[str, Any] = {}
        try:
            results = search_encrypted_video(
                storage_directory,
                query_vector,
                config,
                self.encryption_key,
                self.mac_key,
                temp_root=temporary_root,
                stats=metrics,
            )
        except Exception as error:
            event_type = "face_search_failed"
            if isinstance(error, FaceSearchError) and isinstance(
                error.__cause__, StorageError
            ):
                event_type = "integrity_validation_failed"
            self.repository.record_event(
                task_id, event_type, clean_error(error)
            )
            raise
        metrics["query_image_decode_seconds"] = round(query_decode_seconds, 6)
        metrics["total_seconds"] = round(time.perf_counter() - started, 6)
        self.last_search_metrics = metrics
        self.repository.record_event(
            task_id,
            "face_search_completed",
            json.dumps(metrics, sort_keys=True),
        )
        return [
            {
                "candidate_id": result.candidate_id,
                "frame_index": result.frame_index,
                "timestamp_seconds": result.timestamp_seconds,
                "similarity": result.similarity,
            }
            for result in results
        ]
