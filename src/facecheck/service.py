from __future__ import annotations

import os
import re
import shutil
import json
import time
import uuid
import threading
import hashlib
import hmac
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from .database import TaskRepository
from .face_detection import FaceDetection, FaceDetectionConfig, load_yunet_detector
from .face_embedding import clip_detection_to_image
from .face_search import FaceSearchConfig, FaceSearchError, search_encrypted_video, rank_candidates
from .recognition import (
    DEFAULT_ARCFACE_PATH, DEFAULT_SFACE_PATH, DEFAULT_RECOGNITION_MODEL,
    DEFAULT_ARCFACE_THRESHOLD, recognition_config, model_name,
    extract_recognition_embedding,
)
from .key_management import VideoKeyProvider
from .storage import StorageError, restore_video_object, store_encrypted_video, store_enveloped_video
from .video import VideoProcessingError, read_video_info
from .security import secret_file

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
        key_provider_resolver: Optional[Callable[[str], VideoKeyProvider]] = None,
        write_key_provider: Optional[VideoKeyProvider] = None,
    ) -> None:
        self.repository = repository
        self.storage_root = storage_root
        self._request_metrics = threading.local()
        self.encryption_key = encryption_key or self._key_from_env("FACECHECK_SM4_KEY")
        self.mac_key = mac_key or self._key_from_env("FACECHECK_MAC_KEY")
        self.key_provider_resolver = key_provider_resolver
        self.write_key_provider = write_key_provider
        self.feature_store = None
        use_cache = os.getenv("FACECHECK_FEATURE_CACHE", "1" if os.getenv("FACECHECK_DEPLOYMENT") == "cloud" else "0")
        if use_cache not in {"0", "1"}:
            raise ValueError("FACECHECK_FEATURE_CACHE must be 0 or 1")
        if use_cache == "1" and self.encryption_key and self.mac_key:
            from .feature_store import FeatureStore
            self.feature_store = FeatureStore(storage_root.parent / "protected-features", self.encryption_key, self.mac_key)
        raw_workers = os.getenv("FACECHECK_ENCRYPT_WORKERS", "4")
        try:
            self.encryption_workers = min(32, max(1, int(raw_workers)))
        except ValueError:
            self.encryption_workers = 4

    @staticmethod
    def _path_from_env(name: str) -> Optional[Path]:
        value = os.getenv(name)
        return Path(value) if value else None

    @staticmethod
    def _key_from_env(name: str) -> Optional[bytes]:
        return secret_file(name)

    @property
    def last_search_metrics(self) -> Dict[str, Any]:
        return getattr(self._request_metrics, "value", {})

    @last_search_metrics.setter
    def last_search_metrics(self, metrics: Dict[str, Any]) -> None:
        self._request_metrics.value = metrics

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
            if (self.write_key_provider is None
                    and (self.encryption_key is None or self.mac_key is None)):
                raise StorageError("encryption configuration is unavailable")
            read_video_info(source)
            metrics["video_read_seconds"] = round(time.perf_counter() - video_started, 6)
            encryption_started = time.perf_counter()
            if self.write_key_provider is not None:
                stored = store_enveloped_video(
                    source, self.storage_root, self.write_key_provider,
                    chunk_size=CHUNK_SIZE, key_version="enveloped-" + self.write_key_provider.key_id,
                    video_id=task_id, stats=metrics, workers=self.encryption_workers,
                )
                metrics["key_id"] = self.write_key_provider.key_id
            else:
                stored = store_encrypted_video(
                    source,
                    self.storage_root,
                    self.encryption_key,
                    self.mac_key,
                    chunk_size=CHUNK_SIZE,
                    key_version=KEY_VERSION,
                    video_id=task_id,
                    stats=metrics,
                    workers=self.encryption_workers,
                )
                metrics["key_id"] = "legacy-api-v1"
            metrics["encryption_storage_seconds"] = round(
                time.perf_counter() - encryption_started, 6
            )
            metrics["chunk_count"] = stored.manifest.get("chunk_count", 0)
            metrics["key_version"] = stored.manifest.get("key_version")
            # Trust anchor is outside the storage object; the legacy format remains unchanged.
            metrics["manifest_sha256"] = hashlib.sha256((stored.directory / "manifest.json").read_bytes()).hexdigest()
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
        try:
            self.check_storage_manifest(task, storage_directory)
        except StorageError:
            self.repository.record_event(task_id, "manifest_integrity_rejected", "stored manifest does not match the trusted task record")
            raise

        started = time.perf_counter()
        image_array = np.frombuffer(query_image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("query file is not a readable image")
        query_decode_seconds = time.perf_counter() - started
        yunet_model = self._path_from_env("FACECHECK_YUNET_MODEL")
        if yunet_model is None:
            raise StorageError("face model configuration is unavailable")
        detection_config = FaceDetectionConfig(
            model_path=yunet_model,
            frame_interval_seconds=float(
                os.getenv("FACECHECK_SEARCH_INTERVAL", "1.0")
            ),
        )
        embedding_config = recognition_config(
            os.getenv("FACECHECK_RECOGNITION_MODEL", DEFAULT_RECOGNITION_MODEL),
            arcface_path=self._path_from_env("FACECHECK_ARCFACE_MODEL") or DEFAULT_ARCFACE_PATH,
            sface_path=self._path_from_env("FACECHECK_SFACE_MODEL") or DEFAULT_SFACE_PATH,
        )
        if min_similarity is None and model_name(embedding_config) == "arcface":
            min_similarity = DEFAULT_ARCFACE_THRESHOLD
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
        detection = clip_detection_to_image(detection, image.shape)
        query_vector = extract_recognition_embedding(
            image, detection, embedding_config
        ).vector
        config = FaceSearchConfig(
            detection=detection_config,
            embedding=embedding_config,
            top_k=top_k,
            min_similarity=min_similarity,
            restore_workers=self.encryption_workers,
            expected_manifest_sha256=(task.get("processing_metrics") or {}).get("manifest_sha256"),
        )
        temporary_root = self.storage_root.parent / "search-temp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        metrics: Dict[str, Any] = {}
        try:
            results = self._search_stored(
                storage_directory,
                query_vector,
                config,
                temporary_root,
                metrics,
                expected_video_id=task_id,
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
        metrics["recognition_model"] = model_name(embedding_config)
        metrics["embedding_dimension"] = embedding_config.expected_dimension
        metrics["min_similarity"] = min_similarity
        metrics["threshold_source"] = ("ArcFace complete plaintext selection; frozen"
                                       if min_similarity == DEFAULT_ARCFACE_THRESHOLD and model_name(embedding_config) == "arcface"
                                       else "request override or no filter")
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

    def _search_stored(self, directory, query, config, temporary_root, metrics,
                       *, expected_video_id):
        if self.feature_store is None:
            return search_encrypted_video(directory, query, config, self.encryption_key,
                                          self.mac_key, temp_root=temporary_root, stats=metrics,
                                          expected_video_id=expected_video_id,
                                          key_provider_resolver=self.key_provider_resolver)
        from .feature_store import storage_fingerprint, representation_fingerprint
        store = self.feature_store
        with store.lock(directory.name):
            started = time.perf_counter()
            source = storage_fingerprint(directory, config.expected_manifest_sha256)
            representation = representation_fingerprint(config)
            path, aad = store.context(directory.name, source, representation)
            cached = store.get(path, aad, config.embedding.expected_dimension)
            metrics["feature_cache_lookup_seconds"] = round(time.perf_counter() - started, 6)
            metrics["feature_cache_hit"] = cached is not None
            if cached is not None:
                metrics["feature_count"] = len(cached)
                metrics["cache_integrity"] = "authenticated cache; full ciphertext/manifest SHA-256 equals the previously authenticated source"
                return rank_candidates(query, cached, config, stats=metrics)

            def publish(candidates):
                # Reject changes during extraction; never bind a feature to a new or partial source.
                if storage_fingerprint(directory, config.expected_manifest_sha256) != source or representation_fingerprint(config) != representation:
                    raise StorageError("feature source or model changed during extraction")
                store.put(path, aad, candidates, config.embedding.expected_dimension)
            return search_encrypted_video(directory, query, config, self.encryption_key,
                                          self.mac_key, temp_root=temporary_root, stats=metrics,
                                          candidate_sink=publish,
                                          expected_video_id=expected_video_id,
                                          key_provider_resolver=self.key_provider_resolver)

    def restore_task_storage(self, task, directory: Path, destination: Path,
                             *, stats=None):
        """Restore a DB-anchored v1 or v2 task object with format-specific keys."""
        self.check_storage_manifest(task, directory)
        return restore_video_object(
            directory, destination, self.encryption_key, self.mac_key,
            expected_video_id=task["task_id"],
            expected_manifest_sha256=(task.get("processing_metrics") or {}).get("manifest_sha256"),
            key_provider_resolver=self.key_provider_resolver,
            stats=stats, workers=self.encryption_workers,
        )

    def check_storage_manifest(self, task, directory):
        """Bind new API storage to a trusted DB record without changing its format."""
        expected = (task.get("processing_metrics") or {}).get("manifest_sha256")
        if expected is None and os.getenv("FACECHECK_DEPLOYMENT", "local") != "cloud":
            return False  # read-only legacy compatibility; no unverified anchor is invented
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise StorageError("trusted manifest anchor is unavailable")
        path = directory / "manifest.json"
        if directory.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise StorageError("stored manifest is unavailable or unsafe")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(expected, actual):
            raise StorageError("stored manifest does not match its trusted anchor")
        return True
