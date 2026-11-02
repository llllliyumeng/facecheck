"""Cached, offline YTF Query-Gallery experiment runner.

This module is deliberately independent of FastAPI and SQLite. Gallery MP4s in
the YTF manifest are reconstructed 1 fps inputs, not original YouTube videos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .experiment import load_manifest
from .face_detection import FaceDetection, FaceDetectionConfig, load_yunet_detector
from .face_embedding import (
    ArcFaceConfig,
    FaceEmbeddingError,
    SFaceConfig,
    extract_embedding,
    extract_arcface_embedding,
    clip_detection_to_image,
    load_arcface_recognizer,
    load_sface_recognizer,
)
from .storage import restore_encrypted_video, store_encrypted_video
from .recognition import (
    DEFAULT_ARCFACE_PATH, DEFAULT_SFACE_PATH, DEFAULT_RECOGNITION_MODEL, DEFAULT_ARCFACE_THRESHOLD,
    RecognitionConfig, model_name,
)
from .video import extract_frames
from .ytf_qg_protocol import (
    THRESHOLD_SELECTION, TEST, build_similarity_pairs,
    evaluate_protocol, validate_layout, retrieval_metrics, select_threshold_at_far,
    evaluate_test_threshold, compare_mode_scores,
)


class YTFRunnerError(ValueError):
    """Invalid runner configuration, inputs, or experiment output."""


@dataclass(frozen=True)
class RunnerConfig:
    manifest_path: Path
    identity_split_path: Path
    output_dir: Path
    detection: FaceDetectionConfig
    embedding: RecognitionConfig
    top_k: int = 5
    target_far: float = 1e-3
    encryption_chunk_size: int = 1024 * 1024
    encryption_key_version: str = "ytf-runner-v1"
    similarity_query_batch_size: int = 16
    score_aggregation: str = "max"
    score_aggregation_top_n: int = 3
    encryption_workers: int = 4
    query_use_annotation_box: bool = False
    gallery_max_sampled_frames: int = 12
    gallery_min_detection_confidence: float = 0.9
    gallery_min_face_area_ratio: float = 0.015
    gallery_min_bbox_visibility: float = 0.75
    gallery_quality_filter: bool = False
    gallery_feature_aggregation: str = "none"
    evaluation_scope: str = "full"
    frozen_selection_threshold: float | None = None
    recognition_model: str | None = None
    arcface_model_path: Path | None = None

    def __post_init__(self) -> None:
        # Explicit SFaceConfig objects remain supported for compatibility callers.
        if self.recognition_model is None:
            object.__setattr__(self, "recognition_model", model_name(self.embedding))
        if isinstance(self.embedding, ArcFaceConfig):
            if self.recognition_model != "arcface":
                raise YTFRunnerError("ArcFaceConfig requires recognition_model=arcface")
            if self.arcface_model_path is None:
                object.__setattr__(self, "arcface_model_path", self.embedding.model_path)
            elif self.arcface_model_path != self.embedding.model_path:
                raise YTFRunnerError("ArcFace model paths must agree")

    def validate(self) -> None:
        if self.top_k <= 0 or self.similarity_query_batch_size <= 0:
            raise YTFRunnerError("top_k and similarity_query_batch_size must be positive")
        if not 0 < self.target_far < 1:
            raise YTFRunnerError("target_far must be between zero and one")
        if self.encryption_chunk_size <= 0:
            raise YTFRunnerError("encryption_chunk_size must be positive")
        if self.score_aggregation not in {"max", "top_k_mean"}:
            raise YTFRunnerError("score_aggregation must be max or top_k_mean")
        if self.score_aggregation_top_n <= 0:
            raise YTFRunnerError("score_aggregation_top_n must be positive")
        if (isinstance(self.encryption_workers, bool)
                or not isinstance(self.encryption_workers, int)
                or self.encryption_workers <= 0):
            raise YTFRunnerError("encryption_workers must be positive")
        if not isinstance(self.query_use_annotation_box, bool):
            raise YTFRunnerError("query_use_annotation_box must be boolean")
        if self.gallery_max_sampled_frames <= 0:
            raise YTFRunnerError("gallery_max_sampled_frames must be positive")
        if not 0 <= self.gallery_min_detection_confidence <= 1:
            raise YTFRunnerError("gallery_min_detection_confidence must be between zero and one")
        if not 0 <= self.gallery_min_face_area_ratio <= 1:
            raise YTFRunnerError("gallery_min_face_area_ratio must be between zero and one")
        if not 0 < self.gallery_min_bbox_visibility <= 1:
            raise YTFRunnerError("gallery_min_bbox_visibility must be in (0, 1]")
        if not isinstance(self.gallery_quality_filter, bool):
            raise YTFRunnerError("gallery_quality_filter must be boolean")
        if self.gallery_feature_aggregation not in {"none", "normalized_mean"}:
            raise YTFRunnerError("gallery_feature_aggregation must be none or normalized_mean")
        if self.evaluation_scope not in {"full", THRESHOLD_SELECTION, TEST}:
            raise YTFRunnerError("evaluation_scope must be full, threshold_selection, or test")
        if self.evaluation_scope == TEST:
            if (self.frozen_selection_threshold is None or
                    not np.isfinite(self.frozen_selection_threshold)):
                raise YTFRunnerError("test scope requires a finite frozen selection threshold")
        elif self.frozen_selection_threshold is not None:
            raise YTFRunnerError("frozen_selection_threshold is only valid for test scope")
        if self.recognition_model not in {"sface", "arcface"}:
            raise YTFRunnerError("recognition_model must be sface or arcface")
        if self.recognition_model == "arcface":
            if self.arcface_model_path is None:
                raise YTFRunnerError("arcface_model_path is required when recognition_model=arcface")
            ArcFaceConfig(self.arcface_model_path).validate()
        self.detection.validate()
        self.embedding.validate()


@dataclass(frozen=True)
class FaceFeature:
    vector: np.ndarray
    frame_index: int
    timestamp_seconds: float
    detection_index: int | None = None


@dataclass(frozen=True)
class CachedMediaFeatures:
    identity_id: str
    video_id: str
    split: str
    media_type: str
    features: tuple[FaceFeature, ...]
    sampled_frame_count: int
    source_sha256: str | None = None
    restored_sha256: str | None = None


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: tuple[FaceFeature, ...]
    sampled_frame_count: int
    detected_face_count: int
    feature_extraction_failure_count: int
    invalid_detection_record_count: int
    timings: dict[str, float | None]
    failures: tuple[dict[str, Any], ...] = ()
    detection_failure_count: int = 0
    gallery_quality_rejection_count: int = 0
    gallery_valid_frame_feature_count: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_split_file(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        assignments = data["assignments"]
        split = {str(row["identity_id"]): str(row["split"]) for row in assignments}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise YTFRunnerError(f"invalid identity split file: {exc}") from exc
    if len(split) != len(assignments) or any(value not in {THRESHOLD_SELECTION, TEST} for value in split.values()):
        raise YTFRunnerError("identity split assignments contain duplicates or invalid split names")
    return split, data


def _load_inputs(manifest_path: Path, split_path: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    try:
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest_json["records"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise YTFRunnerError(f"invalid manifest JSON: {exc}") from exc
    # load_manifest enforces the established manifest contract and resolves
    # paths relative to the manifest directory. Keep raw metadata alongside it.
    manifest = load_manifest(manifest_path)
    split, split_json = _read_split_file(split_path)
    if len(rows) != len(manifest.records):
        raise YTFRunnerError("manifest rows do not match validated records")
    records: list[dict[str, Any]] = []
    for raw, parsed in zip(rows, manifest.records):
        item = dict(raw)
        item["video_path"] = str(parsed.video_path)
        item["_manifest_root"] = str(manifest_path.parent)
        records.append(item)
    validate_layout(records, split)
    return records, split, manifest_json, split_json


def _raw_faces(raw_faces: Any) -> list[np.ndarray]:
    if raw_faces is None:
        return []
    result = []
    for raw in raw_faces:
        row = np.asarray(raw, dtype=np.float32).reshape(-1)
        if row.size >= 15:
            result.append(row)
    return result


def _raw_face_entries(raw_faces: Any) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, str]]]:
    valid: list[tuple[int, np.ndarray]] = []
    invalid: list[tuple[int, str]] = []
    if raw_faces is None:
        return valid, invalid
    for index, raw in enumerate(raw_faces):
        try:
            row = np.asarray(raw, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            invalid.append((index, f"YuNet detection row cannot be converted: {exc}"))
            continue
        if row.size < 15:
            invalid.append((index, f"YuNet detection row has {row.size} values; expected at least 15"))
            continue
        valid.append((index, row))
    return valid, invalid


def _failure_record(record: Mapping[str, Any], *, stage: str, frame_index: int | None,
                    face_index: int | None, exception_type: str, exception_message: str) -> dict[str, Any]:
    return {"mode": record.get("_experiment_mode"), "media_id": str(record.get("video_id", "")),
            "identity_id": record.get("identity_id"), "frame_index": frame_index,
            "face_index": face_index, "stage": stage, "exception_type": exception_type,
            "exception_message": exception_message}


def _uniform_sample_frames(frames: Sequence[Any], maximum: int) -> list[Any]:
    if len(frames) <= maximum:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, maximum, dtype=int)
    return [frames[index] for index in indices]


def _aggregate_gallery_features(features: Sequence[FaceFeature]) -> tuple[tuple[FaceFeature, ...], int]:
    unit_vectors: list[np.ndarray] = []
    for feature in features:
        vector = np.asarray(feature.vector, dtype=np.float64)
        if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
            continue
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            continue
        unit_vectors.append(vector / norm)
    if not unit_vectors:
        return (), 0
    mean = np.mean(np.stack(unit_vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 0:
        return (), len(unit_vectors)
    first = features[0]
    return (FaceFeature((mean / norm).astype(np.float32), first.frame_index,
                        first.timestamp_seconds, None),), len(unit_vectors)


class OpenCVFeatureBackend:
    """Loads one detector/recognizer per mode and processes each media item once."""

    def __init__(self, detection: FaceDetectionConfig, embedding: RecognitionConfig,
                 recognition_model: str | None = None, arcface_model_path: Path | None = None):
        recognition_model = recognition_model or model_name(embedding)
        if recognition_model == "arcface" and arcface_model_path is None:
            arcface_model_path = embedding.model_path if isinstance(embedding, ArcFaceConfig) else DEFAULT_ARCFACE_PATH
        self.detection_config = detection
        self.embedding_config = embedding
        self.recognition_model = recognition_model
        self.arcface_config = ArcFaceConfig(arcface_model_path) if recognition_model == "arcface" else None
        self.query_use_annotation_box = False
        self.detector = load_yunet_detector(detection)
        self.recognizer = (load_arcface_recognizer(self.arcface_config)
                           if self.arcface_config is not None else load_sface_recognizer(embedding))

    @staticmethod
    def _annotation_box(record: Mapping[str, Any], image_shape: tuple[int, ...]) -> tuple[float, float, float, float] | None:
        """Read the official YTF box for the selected query frame, if present."""
        annotation_file = record.get("ytf_annotation_file")
        frame_name = record.get("ytf_selected_query_frame_name")
        source_name = record.get("ytf_original_video_name")
        if not annotation_file or not frame_name or not source_name:
            return None
        path = Path(str(annotation_file))
        if not path.is_absolute():
            manifest_root = record.get("_manifest_root")
            path = (Path(str(manifest_root)) / path) if manifest_root else path
        try:
            lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        except (OSError, UnicodeError):
            return None
        normalized_source_name = str(source_name).replace("\\", "/")
        wanted = f"{normalized_source_name}/{frame_name}"
        selected: tuple[float, float, float, float] | None = None
        for line in lines:
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6 or fields[0].replace("\\", "/") != wanted:
                continue
            try:
                x, y, width, height = (float(fields[index]) for index in range(2, 6))
            except (TypeError, ValueError):
                continue
            if all(np.isfinite(value) for value in (x, y, width, height)) and width > 0 and height > 0:
                selected = (x, y, width, height)
                break
        if selected is None:
            return None
        image_height, image_width = image_shape[:2]
        x, y, width, height = selected
        x0, y0 = max(0.0, min(float(image_width), x)), max(0.0, min(float(image_height), y))
        x1 = max(x0, min(float(image_width), x + width))
        y1 = max(y0, min(float(image_height), y + height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    @classmethod
    def _select_query_entries(
        cls, entries: list[tuple[int, np.ndarray]], record: Mapping[str, Any], image_shape: tuple[int, ...],
    ) -> list[tuple[int, np.ndarray]]:
        if not entries or not record.get("_use_annotation_box", False):
            return entries
        target = cls._annotation_box(record, image_shape)
        if target is None:
            return []
        tx, ty, tw, th = target
        tx1, ty1 = tx + tw, ty + th
        best: tuple[float, int, np.ndarray] | None = None
        for index, row in entries:
            x, y, width, height = (float(row[offset]) for offset in range(4))
            x1, y1 = x + max(0.0, width), y + max(0.0, height)
            ix0, iy0, ix1, iy1 = max(tx, x), max(ty, y), min(tx1, x1), min(ty1, y1)
            intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = tw * th + max(0.0, width) * max(0.0, height) - intersection
            iou = intersection / union if union > 0 else 0.0
            candidate = (iou, -index, row)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return [(-best[1], best[2])] if best is not None and best[0] > 0 else []

    def extract(self, record: Mapping[str, Any]) -> FeatureExtractionResult:
        path = Path(str(record["video_path"]))
        media_type = str(record.get("media_type", "video"))
        is_gallery = (str(record.get("split", "")) == "gallery" and media_type == "video" and
                      bool(record.get("_gallery_representation_enabled", False)))
        decode_start = time.perf_counter()
        if media_type == "image":
            image = cv2.imread(str(path))
            if image is None:
                raise YTFRunnerError(f"cannot decode query image: {record['video_id']}")
            frames = [(0, 0.0, image)]
        elif media_type == "video":
            extracted = extract_frames(path, self.detection_config.frame_interval_seconds)
            if (is_gallery and record.get("_gallery_quality_filter", False) and
                    len(extracted) > int(record.get("_gallery_max_sampled_frames", 12))):
                extracted = _uniform_sample_frames(
                    extracted, int(record.get("_gallery_max_sampled_frames", 12)))
            frames = [(frame.index, frame.timestamp_seconds, frame.image) for frame in extracted]
        else:
            raise YTFRunnerError(f"unsupported media_type for {record['video_id']}: {media_type}")
        # extract_frames includes metadata inspection and frame decoding. Do not
        # claim a separately measured decode or frame-selection duration.
        decode_and_frames = time.perf_counter() - decode_start
        detect_seconds = embed_seconds = 0.0
        detected_face_count = feature_failures = invalid_detection_records = detection_failures = 0
        found: list[FaceFeature] = []
        quality_rejections = 0
        failures: list[dict[str, Any]] = []
        for frame_index, timestamp, image in frames:
            height, width = image.shape[:2]
            self.detector.setInputSize((width, height))
            started = time.perf_counter()
            try:
                _, raw_faces = self.detector.detect(image)
            except Exception as exc:
                detect_seconds += time.perf_counter() - started
                detection_failures += 1
                failures.append(_failure_record(
                    record, stage="face_detection", frame_index=frame_index, face_index=None,
                    exception_type=type(exc).__name__, exception_message=str(exc)))
                continue
            detect_seconds += time.perf_counter() - started
            detected_face_count += 0 if raw_faces is None else len(raw_faces)
            entries, invalid_entries = _raw_face_entries(raw_faces)
            invalid_detection_records += len(invalid_entries)
            for face_index, message in invalid_entries:
                failures.append(_failure_record(
                    record, stage="detection_validation", frame_index=frame_index,
                    face_index=face_index, exception_type="InvalidDetectionRecord",
                    exception_message=message))
            if media_type == "image" and self.query_use_annotation_box:
                entries = self._select_query_entries(entries, record, image.shape)
            if is_gallery and record.get("_gallery_quality_filter", False):
                qualified: list[tuple[int, np.ndarray]] = []
                for face_index, row in entries:
                    x, y, box_width, box_height = (float(row[offset]) for offset in range(4))
                    confidence = float(row[14])
                    area = max(0.0, box_width) * max(0.0, box_height)
                    intersection = max(0.0, min(width, x + box_width) - max(0.0, x)) * max(
                        0.0, min(height, y + box_height) - max(0.0, y))
                    visibility = intersection / area if area > 0 else 0.0
                    if (not np.all(np.isfinite([x, y, box_width, box_height, confidence])) or
                            box_width <= 0 or box_height <= 0 or
                            confidence < float(record.get("_gallery_min_detection_confidence", 0.9)) or
                            area / (width * height) < float(record.get("_gallery_min_face_area_ratio", 0.015)) or
                            visibility < float(record.get("_gallery_min_bbox_visibility", 0.75))):
                        quality_rejections += 1
                    else:
                        qualified.append((face_index, row))
                # A video contributes at most one face per sampled frame: the
                # highest-confidence eligible detection, with area as tie-break.
                entries = sorted(qualified, key=lambda item: (
                    float(item[1][14]), float(item[1][2]) * float(item[1][3]), -item[0]),
                    reverse=True)[:1]
            for face_index, row in entries:
                detection = FaceDetection(
                    frame_index=frame_index, timestamp_seconds=timestamp,
                    video_width=width, video_height=height,
                    x=float(row[0]), y=float(row[1]), width=float(row[2]), height=float(row[3]),
                    confidence=float(row[14]),
                    landmarks=tuple((float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)),
                )
                started = time.perf_counter()
                try:
                    # YuNet can return a box or landmark a few pixels beyond
                    # the decoded frame at image boundaries. Validate and
                    # clamp that geometry before invoking SFace. Keeping the
                    # compatibility call below free of new keyword arguments
                    # also supports injected test backends.
                    detection = clip_detection_to_image(detection, image.shape)
                    if self.arcface_config is not None:
                        embedding = extract_arcface_embedding(
                            image, detection, self.arcface_config,
                            recognizer_factory=lambda _config: self.recognizer,
                        )
                    else:
                        embedding = extract_embedding(
                            image, detection, self.embedding_config,
                            recognizer_factory=lambda _config: self.recognizer,
                        )
                except FaceEmbeddingError as exc:
                    embed_seconds += time.perf_counter() - started
                    feature_failures += 1
                    failures.append(_failure_record(
                        record, stage=getattr(exc, "stage", "feature_extraction"),
                        frame_index=frame_index, face_index=face_index,
                        exception_type=type(exc).__name__, exception_message=str(exc)))
                    continue
                embed_seconds += time.perf_counter() - started
                found.append(FaceFeature(embedding.vector, frame_index, timestamp, face_index))
        frame_feature_count = len(found) if is_gallery else 0
        if is_gallery and found and record.get("_gallery_feature_aggregation", "normalized_mean") == "normalized_mean":
            aggregated, frame_feature_count = _aggregate_gallery_features(found)
            found = list(aggregated)
        return FeatureExtractionResult(
            tuple(found), len(frames), detected_face_count, feature_failures,
            invalid_detection_records,
            {"video_read_decode_extract_seconds": decode_and_frames if media_type == "video" else None,
             "frame_extraction_seconds": None,
             "query_image_decode_seconds": decode_and_frames if media_type == "image" else None,
             "face_detection_seconds": detect_seconds,
             "feature_extraction_seconds": embed_seconds}, tuple(failures), detection_failures,
            quality_rejections, frame_feature_count,
        )


def _matrix_diagnostics(matrix: np.ndarray, *, vector_count: int, rejected: Sequence[Mapping[str, Any]],
                        norms: np.ndarray) -> dict[str, Any]:
    finite_norms = norms[np.isfinite(norms)]
    return {
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "vector_count": vector_count, "valid_vector_count": int(matrix.shape[0]),
        "rejected_vector_count": len(rejected),
        "finite_vector_count": int(matrix.shape[0]),
        "non_finite_vector_count": sum(row["reason"] == "non_finite" for row in rejected),
        "zero_norm_vector_count": sum(row["reason"] == "zero_norm" for row in rejected),
        "invalid_shape_vector_count": sum(row["reason"] == "invalid_shape" for row in rejected),
        "norm_min": float(finite_norms.min()) if finite_norms.size else None,
        "norm_max": float(finite_norms.max()) if finite_norms.size else None,
        "max_absolute_value": float(np.max(np.abs(matrix))) if matrix.size else None,
        "finite": bool(np.all(np.isfinite(matrix))),
        "rejected_vectors": list(rejected),
    }


def _prepare_vectors(items: Sequence[tuple[str, int, FaceFeature]], *, expected_dimension: int | None,
                     role: str) -> tuple[np.ndarray, list[tuple[str, int, FaceFeature]], dict[str, Any]]:
    accepted: list[tuple[str, int, FaceFeature]] = []
    rows: list[np.ndarray] = []
    rejected: list[dict[str, Any]] = []
    observed_dimension = expected_dimension
    for media_id, feature_index, feature in items:
        try:
            raw = np.asarray(feature.vector)
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError("invalid_shape")
            if not np.issubdtype(raw.dtype, np.number):
                raise TypeError("non_numeric")
            if np.iscomplexobj(raw):
                raise TypeError("non_real")
            if observed_dimension is None:
                observed_dimension = int(raw.size)
            if raw.size != observed_dimension:
                raise ValueError("invalid_shape")
            if not np.all(np.isfinite(raw)):
                raise FloatingPointError("non_finite")
            # Refuse values that cannot be represented in the established float32
            # feature format instead of allowing a cast to overflow silently.
            float32_limit = np.finfo(np.float32).max
            if np.any(raw > float32_limit) or np.any(raw < -float32_limit):
                raise FloatingPointError("outside_float32_range")
            vector = np.asarray(raw, dtype=np.float32)
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if norm == 0.0:
                raise FloatingPointError("zero_norm")
            if not np.isfinite(norm):
                raise FloatingPointError("non_finite_norm")
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            message = str(exc)
            reason = message if message in {"invalid_shape", "non_numeric", "non_real", "non_finite",
                                          "outside_float32_range", "zero_norm", "non_finite_norm"} else "invalid_input"
            rejected.append({"role": role, "media_id": media_id, "vector_index": feature_index,
                             "feature_index": feature_index,
                             "frame_index": feature.frame_index,
                             "detection_index": feature.detection_index, "reason": reason})
            continue
        accepted.append((media_id, feature_index, feature))
        rows.append(vector)
    dimension = observed_dimension or 0
    matrix = np.stack(rows).astype(np.float32, copy=False) if rows else np.empty((0, dimension), dtype=np.float32)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1) if matrix.size else np.empty((0,), dtype=np.float64)
    diagnostic = _matrix_diagnostics(matrix, vector_count=len(items), rejected=rejected, norms=norms)
    return np.ascontiguousarray(matrix), accepted, diagnostic


def _normalize_matrix(matrix: np.ndarray, norms: np.ndarray, *, role: str) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty(matrix.shape, dtype=np.float32)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise YTFRunnerError(f"{role} normalization input validation failed")
    normalized = (matrix.astype(np.float64) / norms[:, None]).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise YTFRunnerError(f"{role} normalization produced non-finite values")
    return np.ascontiguousarray(normalized)


def _multiply_similarity(query_matrix: np.ndarray, gallery_matrix: np.ndarray) -> np.ndarray:
    # Avoid platform BLAS float32 matmul warnings observed on normalized SFace
    # vectors while retaining the same all-pairs dot-product scoring semantics.
    query64 = query_matrix.astype(np.float64, copy=False)
    gallery64 = gallery_matrix.astype(np.float64, copy=False)
    return np.einsum("ik,jk->ij", query64, gallery64, optimize=False)


def _score_matrix(queries: Sequence[Mapping[str, Any]], galleries: Sequence[Mapping[str, Any]],
                  query_cache: Mapping[str, CachedMediaFeatures], gallery_cache: Mapping[str, CachedMediaFeatures],
                  batch_size: int, *, expected_dimension: int | None = None,
                  diagnostics: list[dict[str, Any]] | None = None,
                  mode: str | None = None, split: str | None = None,
                  score_aggregation: str = "max", score_aggregation_top_n: int = 3) -> tuple[dict[tuple[str, str], float | None],
                                            dict[tuple[str, str], dict[str, Any]],
                                            float, dict[str, float]]:
    """Vectorized cosine scores with per-stage numeric diagnostics and explicit rejection."""
    del batch_size  # Query/Gallery scoring remains bounded one Query media at a time.
    started = time.perf_counter()
    gallery_items: list[tuple[str, int, FaceFeature]] = []
    for gallery in galleries:
        gid = str(gallery["video_id"])
        gallery_items.extend((gid, index, feature)
                             for index, feature in enumerate(gallery_cache[gid].features))
    matrix, valid_gallery_items, gallery_input_diag = _prepare_vectors(
        gallery_items, expected_dimension=expected_dimension, role="gallery")
    # Rebuild media ranges after invalid vectors have been rejected.
    valid_by_media: dict[str, list[tuple[int, FaceFeature]]] = {}
    for media_id, feature_index, feature in valid_gallery_items:
        valid_by_media.setdefault(media_id, []).append((feature_index, feature))
    gallery_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for gallery in galleries:
        gid = str(gallery["video_id"])
        begin = cursor
        cursor += len(valid_by_media.get(gid, []))
        gallery_offsets[gid] = (begin, cursor)
    gallery_norms = np.linalg.norm(matrix.astype(np.float64), axis=1) if matrix.size else np.empty((0,), dtype=np.float64)
    try:
        normalized_gallery = _normalize_matrix(matrix, gallery_norms, role="gallery")
    except Exception as exc:
        raise YTFRunnerError(f"normalization stage failed for Gallery matrix: {exc}") from exc
    gallery_input_diag["normalization"] = {
        "stage": "normalization", "shape": list(normalized_gallery.shape),
        "dtype": str(normalized_gallery.dtype), "finite": bool(np.all(np.isfinite(normalized_gallery))),
        "max_absolute_value": float(np.max(np.abs(normalized_gallery))) if normalized_gallery.size else None,
    }
    if not np.all(np.isfinite(normalized_gallery)):
        raise YTFRunnerError("normalization stage produced non-finite Gallery values")
    scores: dict[tuple[str, str], float | None] = {}
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    per_query_seconds: dict[str, float] = {}
    for query in queries:
        qid = str(query["video_id"])
        query_started = time.perf_counter()
        qitems = [(qid, index, feature) for index, feature in enumerate(query_cache[qid].features)]
        qmat, valid_query_items, query_input_diag = _prepare_vectors(
            qitems, expected_dimension=expected_dimension, role="query")
        qnorms = np.linalg.norm(qmat.astype(np.float64), axis=1) if qmat.size else np.empty((0,), dtype=np.float64)
        try:
            normalized_query = _normalize_matrix(qmat, qnorms, role="query")
        except Exception as exc:
            raise YTFRunnerError(f"normalization stage failed for Query {qid}: {exc}") from exc
        query_input_diag["normalization"] = {
            "stage": "normalization", "shape": list(normalized_query.shape),
            "dtype": str(normalized_query.dtype), "finite": bool(np.all(np.isfinite(normalized_query))),
            "max_absolute_value": float(np.max(np.abs(normalized_query))) if normalized_query.size else None,
        }
        if not np.all(np.isfinite(normalized_query)):
            raise YTFRunnerError(f"normalization stage produced non-finite Query values for {qid}")
        similarity_diag: dict[str, Any] = {
            "mode": mode, "split": split, "query_id": qid,
            "gallery_input": gallery_input_diag, "query_input": query_input_diag,
            "effective_query_vector_count": len(valid_query_items),
            "effective_gallery_vector_count": len(valid_gallery_items),
        }
        if normalized_query.shape[0] and normalized_gallery.shape[0]:
            try:
                similarities = _multiply_similarity(normalized_query, normalized_gallery)
            except Exception as exc:
                similarity_diag["matrix_multiplication"] = {
                    "stage": "matrix_multiplication", "shape": None,
                    "dtype": None, "finite": False, "status": "exception",
                    "exception_type": type(exc).__name__, "exception_message": str(exc),
                }
                if diagnostics is not None:
                    diagnostics.append(similarity_diag)
                raise YTFRunnerError(
                    f"matrix multiplication stage failed for Query {qid}: {type(exc).__name__}: {exc}"
                ) from exc
            matmul_diag = {"stage": "matrix_multiplication", "shape": list(similarities.shape),
                           "dtype": str(similarities.dtype), "finite": bool(np.all(np.isfinite(similarities))),
                           "non_finite_value_count": int(similarities.size - np.count_nonzero(np.isfinite(similarities)))}
            similarity_diag["matrix_multiplication"] = matmul_diag
            expected_shape = (len(valid_query_items), len(valid_gallery_items))
            if similarities.shape != expected_shape:
                matmul_diag["finite"] = False
                matmul_diag["status"] = "invalid_shape"
                if diagnostics is not None:
                    diagnostics.append(similarity_diag)
                raise YTFRunnerError(
                    f"matrix multiplication stage returned shape {similarities.shape}; expected {expected_shape}"
                )
            if not matmul_diag["finite"]:
                if diagnostics is not None:
                    diagnostics.append(similarity_diag)
                raise YTFRunnerError(f"matrix multiplication stage produced non-finite scores for Query {qid}")
            for gallery in galleries:
                gid = str(gallery["video_id"])
                begin, end = gallery_offsets[gid]
                raw = similarities[:, begin:end]
                if end > begin:
                    flat_index = int(np.argmax(raw))
                    query_valid_index, gallery_valid_index = np.unravel_index(flat_index, raw.shape)
                    if score_aggregation == "max":
                        score = float(raw[query_valid_index, gallery_valid_index])
                    else:
                        flattened = np.sort(raw.reshape(-1))
                        count = min(score_aggregation_top_n, flattened.size)
                        score = float(np.mean(flattened[-count:]))
                    scores[(qid, gid)] = score
                    _, gallery_feature_index, gallery_feature = valid_gallery_items[begin + gallery_valid_index]
                    _, query_feature_index, query_feature = valid_query_items[query_valid_index]
                    provenance[(qid, gid)] = {
                        "query_face_index": query_feature_index,
                        "query_frame_index": query_feature.frame_index,
                        "query_timestamp_seconds": query_feature.timestamp_seconds,
                        "candidate_frame_index": gallery_feature.frame_index,
                        "candidate_timestamp_seconds": gallery_feature.timestamp_seconds,
                        "candidate_feature_index": gallery_feature_index,
                    }
                else:
                    scores[(qid, gid)] = None
        else:
            similarity_diag["matrix_multiplication"] = {
                "stage": "matrix_multiplication", "shape": [len(valid_query_items), len(valid_gallery_items)],
                "dtype": "float32", "finite": None, "status": "not_run_no_valid_vectors"}
            for gallery in galleries:
                scores[(qid, str(gallery["video_id"]))] = None
        if diagnostics is not None:
            diagnostics.append(similarity_diag)
        per_query_seconds[qid] = time.perf_counter() - query_started
    return scores, provenance, time.perf_counter() - started, per_query_seconds


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if path.exists():
        raise YTFRunnerError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _timing_rows(mode: str, feature_timings: Sequence[Mapping[str, Any]],
                 crypto_timings: Sequence[Mapping[str, Any]], matrix_seconds: float,
                 ranking_seconds: float, total_seconds: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timing in feature_timings:
        for stage in ("video_read_decode_extract_seconds", "frame_extraction_seconds",
                      "query_image_decode_seconds", "face_detection_seconds", "feature_extraction_seconds",
                      "query_total_seconds"):
            rows.append({"mode": mode, "media_id": timing["media_id"], "split": timing["split"],
                         "stage": stage, "seconds": timing.get(stage), "measurement": "per_media"})
    for timing in crypto_timings:
        for stage in ("encryption_seconds", "restore_integrity_seconds"):
            rows.append({"mode": mode, "media_id": timing["media_id"], "split": timing["split"],
                         "stage": stage, "seconds": timing.get(stage), "measurement": "per_media"})
    for stage, value in (("similarity_calculation_seconds", matrix_seconds),
                         ("ranking_seconds", ranking_seconds), ("experiment_total_seconds", total_seconds)):
        rows.append({"mode": mode, "media_id": None, "split": None, "stage": stage,
                     "seconds": value, "measurement": "mode_total"})
    return rows


def _run_ytf_runner_impl(config: RunnerConfig, *, encryption_key: bytes, mac_key: bytes,
                         backend_factory: Callable[[FaceDetectionConfig, SFaceConfig], Any],
                         encrypt: Callable[..., Any], restore: Callable[..., Any],
                         work_root: Path) -> dict[str, Any]:
    """Run paired plaintext/encrypted-restore modes. Full YTF inference is explicit."""
    config.validate()
    if not encryption_key or not mac_key:
        raise YTFRunnerError("both encryption_key and mac_key are required")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise YTFRunnerError(f"output directory is not empty; refusing to overwrite: {config.output_dir}")
    records, identity_split, manifest_json, split_json = _load_inputs(config.manifest_path, config.identity_split_path)
    active_sides = ((THRESHOLD_SELECTION, TEST) if config.evaluation_scope == "full"
                    else (config.evaluation_scope,))
    if config.evaluation_scope != "full":
        records = [record for record in records
                   if identity_split[record["identity_id"]] == config.evaluation_scope]
        identity_split = {identity: side for identity, side in identity_split.items()
                          if side == config.evaluation_scope}
        validate_layout(records, identity_split)
    # A newly created output path is owned by this run; existing data is never
    # replaced. Feature vectors remain in memory and are not exported.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    gallery_by_split = {side: [r for r in records if r["split"] == "gallery" and identity_split[r["identity_id"]] == side]
                        for side in active_sides}
    query_by_split = {side: [r for r in records if r["split"] == "query" and identity_split[r["identity_id"]] == side]
                      for side in active_sides}
    media_hashes = {str(r["video_id"]): _sha256(Path(r["video_path"])) for r in records}
    mode_data: dict[str, dict[str, Any]] = {}
    global_started = time.perf_counter()
    for mode in ("plaintext", "encrypted_restore"):
        mode_started = time.perf_counter()
        mode_root = work_root / mode
        mode_root.mkdir(exist_ok=False)
        gallery_cache: dict[str, CachedMediaFeatures] = {}
        query_cache: dict[str, CachedMediaFeatures] = {}
        feature_timings: list[dict[str, Any]] = []
        crypto_timings: list[dict[str, Any]] = []
        feature_failures: list[dict[str, Any]] = []
        scoring_diagnostics: list[dict[str, Any]] = []
        restored_gallery_paths: dict[str, Path] = {}
        try:
            if config.recognition_model == "arcface":
                backend = backend_factory(config.detection, config.embedding,
                                          recognition_model=config.recognition_model,
                                          arcface_model_path=config.arcface_model_path)
            else:
                backend = backend_factory(config.detection, config.embedding)
            if hasattr(backend, "query_use_annotation_box"):
                backend.query_use_annotation_box = config.query_use_annotation_box
            # Encrypt and restore each Gallery video exactly once, before its
            # single feature extraction. Query still images are shared inputs.
            for side in active_sides:
                for gallery in gallery_by_split[side]:
                    gid = str(gallery["video_id"])
                    target = Path(gallery["video_path"])
                    enc_seconds = restore_seconds = None
                    restored_hash = None
                    if mode == "encrypted_restore":
                        stats: dict[str, float] = {}
                        started = time.perf_counter()
                        stored = encrypt(target, mode_root / "encrypted", encryption_key, mac_key,
                                         chunk_size=config.encryption_chunk_size,
                                         key_version=config.encryption_key_version,
                                         video_id=f"video_{len(crypto_timings):05d}", stats=stats,
                                         workers=config.encryption_workers)
                        enc_seconds = time.perf_counter() - started
                        restored = mode_root / "restored" / f"{gid}.mp4"
                        restored.parent.mkdir(parents=True, exist_ok=True)
                        started = time.perf_counter()
                        restore(stored.directory, restored, encryption_key, mac_key,
                                workers=config.encryption_workers)
                        restore_seconds = time.perf_counter() - started
                        restored_hash = _sha256(restored)
                        if restored_hash != media_hashes[gid]:
                            raise YTFRunnerError(f"restored SHA-256 mismatch for {gid}")
                        restored_gallery_paths[gid] = restored
                    crypto_timings.append({"media_id": gid, "split": side,
                                           "encryption_seconds": enc_seconds,
                                           "restore_integrity_seconds": restore_seconds,
                                           "source_sha256": media_hashes[gid],
                                           "restored_sha256": restored_hash})
                    extraction_record = dict(gallery, video_path=str(target if mode == "plaintext" else restored_gallery_paths[gid]),
                                             _experiment_mode=mode,
                                             _gallery_representation_enabled=config.gallery_quality_filter or config.gallery_feature_aggregation != "none",
                                             _gallery_quality_filter=config.gallery_quality_filter,
                                             _gallery_feature_aggregation=config.gallery_feature_aggregation,
                                             _gallery_max_sampled_frames=config.gallery_max_sampled_frames,
                                             _gallery_min_detection_confidence=config.gallery_min_detection_confidence,
                                             _gallery_min_face_area_ratio=config.gallery_min_face_area_ratio,
                                             _gallery_min_bbox_visibility=config.gallery_min_bbox_visibility)
                    extraction = backend.extract(extraction_record)
                    feature_failures.extend(extraction.failures)
                    features, frames, times = extraction.features, extraction.sampled_frame_count, extraction.timings
                    gallery_cache[gid] = CachedMediaFeatures(
                        str(gallery["identity_id"]), gid, side, "video", features, frames,
                        media_hashes[gid], restored_hash)
                    feature_timings.append({"media_id": gid, "split": side, "media_type": "video", **times,
                                            "sampled_frame_count": frames,
                                            "detected_face_count": extraction.detected_face_count,
                                            "feature_count": len(features),
                                            "feature_extraction_failure_count": extraction.feature_extraction_failure_count,
                                            "gallery_quality_rejection_count": extraction.gallery_quality_rejection_count,
                                            "gallery_valid_frame_feature_count": extraction.gallery_valid_frame_feature_count,
                                            "detection_failure_count": extraction.detection_failure_count,
                                            "invalid_detection_record_count": extraction.invalid_detection_record_count})
                    if mode == "encrypted_restore":
                        restored_gallery_paths[gid].unlink()
                        del restored_gallery_paths[gid]
            for side in active_sides:
                for query in query_by_split[side]:
                    qid = str(query["video_id"])
                    query_extraction_started = time.perf_counter()
                    extraction = backend.extract(dict(
                        query, _experiment_mode=mode,
                        _use_annotation_box=config.query_use_annotation_box,
                    ))
                    feature_failures.extend(extraction.failures)
                    features, frames, times = extraction.features, extraction.sampled_frame_count, extraction.timings
                    query_extraction_seconds = time.perf_counter() - query_extraction_started
                    # All YuNet detections are retained. Pair scores aggregate
                    # every query face against every gallery face by maximum.
                    query_cache[qid] = CachedMediaFeatures(str(query["identity_id"]), qid, side,
                                                           "image", features, frames, media_hashes[qid], None)
                    feature_timings.append({"media_id": qid, "split": side, "media_type": "image", **times,
                                            "sampled_frame_count": frames,
                                            "detected_face_count": extraction.detected_face_count,
                                            "feature_count": len(features),
                                            "feature_extraction_failure_count": extraction.feature_extraction_failure_count,
                                            "detection_failure_count": extraction.detection_failure_count,
                                            "invalid_detection_record_count": extraction.invalid_detection_record_count,
                                            "query_extraction_total_seconds": query_extraction_seconds})
            pair_scores: dict[tuple[str, str], float | None] = {}
            pair_provenance: dict[tuple[str, str], dict[str, Any]] = {}
            detected_ids: set[str] = set()
            similarity_seconds = ranking_seconds = 0.0
            query_similarity_seconds: dict[str, float] = {}
            query_results: list[dict[str, Any]] = []
            candidate_results: list[dict[str, Any]] = []
            for side in active_sides:
                queries, galleries = query_by_split[side], gallery_by_split[side]
                scores, provenance, elapsed, per_query_similarity = _score_matrix(
                    queries, galleries, query_cache, gallery_cache,
                    config.similarity_query_batch_size,
                    expected_dimension=(512 if config.recognition_model == "arcface"
                                        else config.embedding.expected_dimension),
                    diagnostics=scoring_diagnostics, mode=mode, split=side,
                    score_aggregation=config.score_aggregation,
                    score_aggregation_top_n=config.score_aggregation_top_n)
                similarity_seconds += elapsed
                query_similarity_seconds.update(per_query_similarity)
                detected_ids.update(str(q["video_id"]) for q in queries if query_cache[str(q["video_id"])].features)
                # Keep isolated matrices separate; validate their complete union
                # after both identity partitions have been scored.
                pair_scores.update(scores)
                pair_provenance.update(provenance)
            # Rebuild pairs once against the complete expected within-split key
            # set. The key set excludes all cross-split query/gallery pairs.
            detected_ids = {qid for qid, cache in query_cache.items() if cache.features}
            protocol_pairs = build_similarity_pairs(records, identity_split, pair_scores, detected_ids)
            timing_by_media = {row["media_id"]: row for row in feature_timings}
            for pair in protocol_pairs:
                if pair["similarity"] is not None:
                    continue
                query_count = timing_by_media[pair["query_id"]]["detected_face_count"]
                gallery_count = timing_by_media[pair["gallery_id"]]["detected_face_count"]
                query_features = timing_by_media[pair["query_id"]]["feature_count"]
                gallery_features = timing_by_media[pair["gallery_id"]]["feature_count"]
                if not query_count:
                    pair["score_status"] = "query_face_not_detected"
                elif not query_features:
                    pair["score_status"] = "query_feature_extraction_failed"
                elif not gallery_count:
                    pair["score_status"] = "gallery_face_not_detected"
                elif not gallery_features:
                    pair["score_status"] = "gallery_feature_extraction_failed"
                else:
                    pair["score_status"] = "no_valid_similarity_score"
            by_query_pairs: dict[str, list[dict[str, Any]]] = {}
            for pair in protocol_pairs:
                by_query_pairs.setdefault(pair["query_id"], []).append(pair)
            # Per-query ranking exports are derived from the same complete matrix.
            for side in active_sides:
                for query in query_by_split[side]:
                    qid = str(query["video_id"])
                    qcache = query_cache[qid]
                    ranked = sorted(by_query_pairs[qid], key=lambda row: (
                        row["similarity"] is None,
                        -float(row["similarity"]) if row["similarity"] is not None else 0,
                        str(row["gallery_id"])))
                    start = time.perf_counter()
                    valid_rank = 0
                    for row in ranked:
                        if row["similarity"] is not None:
                            valid_rank += 1
                            row.update(pair_provenance[(qid, row["gallery_id"])])
                            gallery = next(g for g in gallery_by_split[side] if g["video_id"] == row["gallery_id"])
                            row.update({"mode": mode, "query_identity_id": query["identity_id"],
                                        "candidate_identity_id": gallery["identity_id"],
                                        "candidate_video_id": gallery["video_id"],
                                        "rank": valid_rank,
                                        "within_top_k": valid_rank <= config.top_k,
                                        "candidate_detected_face_count": timing_by_media[gallery["video_id"]]["detected_face_count"],
                                        "candidate_feature_count": len(gallery_cache[str(gallery["video_id"])].features),
                                        "candidate_feature_extraction_failure_count": timing_by_media[gallery["video_id"]]["feature_extraction_failure_count"],
                                        "is_correct_match": gallery["identity_id"] == query["identity_id"],
                                        "is_error_match": gallery["identity_id"] != query["identity_id"],
                                        "status": "scored"})
                        else:
                            row.update({"mode": mode, "query_identity_id": query["identity_id"],
                                        "candidate_identity_id": next(g["identity_id"] for g in galleries if g["video_id"] == row["gallery_id"]) if (galleries := gallery_by_split[side]) else None,
                                        "candidate_video_id": row["gallery_id"], "rank": None,
                                        "within_top_k": False,
                                        "candidate_detected_face_count": timing_by_media[row["gallery_id"]]["detected_face_count"],
                                        "candidate_feature_count": len(gallery_cache[str(row["gallery_id"])].features),
                                        "candidate_feature_extraction_failure_count": timing_by_media[row["gallery_id"]]["feature_extraction_failure_count"],
                                        "is_correct_match": None, "is_error_match": None,
                                        "query_face_index": None, "query_frame_index": None,
                                        "query_timestamp_seconds": None, "candidate_frame_index": None,
                                        "candidate_timestamp_seconds": None,
                                        "status": row["score_status"]})
                        pair = dict(row)
                        # Export best matching frame provenance without changing
                        # the query-gallery score unit (one max score per video).
                        candidate_results.append(pair)
                    ranking_seconds += time.perf_counter() - start
                    valid = [row for row in ranked if row["similarity"] is not None]
                    top = valid[0] if valid else None
                    query_results.append({
                        "mode": mode, "query_id": qid, "query_identity_id": query["identity_id"],
                        "query_split": side,
                        "query_detected_face_count": timing_by_media[qid]["detected_face_count"],
                        "query_feature_count": len(qcache.features),
                        "query_feature_extraction_failure_count": timing_by_media[qid]["feature_extraction_failure_count"],
                        "query_detected": timing_by_media[qid]["detected_face_count"] > 0,
                        "query_feature_available": bool(qcache.features),
                        "query_candidate_feature_count": len(qcache.features), "candidate_count": len(ranked),
                        "scored_candidate_count": len(valid),
                        "top1_gallery_id": top["gallery_id"] if top else None,
                        "top1_identity_id": next((g["identity_id"] for g in gallery_by_split[side] if top and g["video_id"] == top["gallery_id"]), None),
                        "top1_similarity": top["similarity"] if top else None,
                        "top1_correct": (next(g["identity_id"] for g in gallery_by_split[side] if g["video_id"] == top["gallery_id"]) == query["identity_id"]) if top else None,
                        "status": "scored" if top else next(
                            (row["score_status"] for row in ranked), "gallery_face_not_detected"),
                    })
                    for pair in ranked:
                        pair["mode"] = mode
                    q_timing = next(item for item in feature_timings if item["media_id"] == qid)
                    q_timing["query_total_seconds"] = (
                        float(q_timing.get("query_extraction_total_seconds") or 0)
                        + float(query_similarity_seconds.get(qid) or 0)
                        + (time.perf_counter() - start)
                    )
            mode_data[mode] = {
                "pairs": protocol_pairs, "detected_ids": detected_ids,
                "query_results": query_results, "candidate_results": candidate_results,
                "feature_cache": {**gallery_cache, **query_cache},
                "feature_timings": feature_timings, "crypto_timings": crypto_timings,
                "feature_failures": feature_failures, "scoring_diagnostics": scoring_diagnostics,
                "similarity_seconds": similarity_seconds, "ranking_seconds": ranking_seconds,
                "query_total_seconds": sum(float(x.get("query_total_seconds") or 0)
                                            for x in feature_timings if x.get("media_type") == "image"),
                "total_seconds": time.perf_counter() - mode_started,
            }
        finally:
            # Work files are confined to this newly created per-mode directory.
            shutil.rmtree(mode_root)
    plain = mode_data["plaintext"]
    encrypted = mode_data["encrypted_restore"]
    stage_timings = {mode: {
            "video_read_decode_extract_seconds": sum(float(x["video_read_decode_extract_seconds"] or 0) for x in data["feature_timings"]),
            "frame_extraction_seconds": None,
            "face_detection_seconds": sum(float(x["face_detection_seconds"] or 0) for x in data["feature_timings"]),
            "feature_extraction_seconds": sum(float(x["feature_extraction_seconds"] or 0) for x in data["feature_timings"]),
            "similarity_calculation_seconds": data["similarity_seconds"],
            "ranking_seconds": data["ranking_seconds"],
            "encryption_seconds": sum(float(x["encryption_seconds"] or 0) for x in data["crypto_timings"]),
            "restore_integrity_seconds": sum(float(x["restore_integrity_seconds"] or 0) for x in data["crypto_timings"]),
            "query_total_seconds": data["query_total_seconds"],
            "experiment_total_seconds": data["total_seconds"],
        } for mode, data in mode_data.items()}
    if config.evaluation_scope == "full":
        metrics = evaluate_protocol(
            records, identity_split, plain["pairs"], encrypted["pairs"], plain["detected_ids"],
            encrypted["detected_ids"], target_far=config.target_far, stage_timings=stage_timings)
    elif config.evaluation_scope == THRESHOLD_SELECTION:
        threshold = select_threshold_at_far(plain["pairs"], target_far=config.target_far)
        metrics = {
            "protocol": "identity-disjoint-query-gallery", "evaluation_scope": THRESHOLD_SELECTION,
            "target_far": config.target_far,
            "threshold_source": "plaintext threshold_selection pairs only",
            "threshold_selection": threshold,
            "plaintext": {"retrieval": retrieval_metrics(records, identity_split, plain["pairs"], plain["detected_ids"]),
                          "tar_far_selection": threshold},
            "encrypted_restore": {"retrieval": retrieval_metrics(records, identity_split, encrypted["pairs"], encrypted["detected_ids"]),
                                  "tar_far_selection": select_threshold_at_far(encrypted["pairs"], target_far=config.target_far)},
            "mode_consistency": compare_mode_scores(
                {(r["query_id"], r["gallery_id"]): r["similarity"] for r in plain["pairs"]},
                {(r["query_id"], r["gallery_id"]): r["similarity"] for r in encrypted["pairs"]}),
            "cross_partition_comparisons": 0, "stage_timings": stage_timings,
        }
    else:
        frozen = {"status": "estimated", "split": THRESHOLD_SELECTION,
                  "threshold": config.frozen_selection_threshold}
        metrics = {
            "protocol": "identity-disjoint-query-gallery", "evaluation_scope": TEST,
            "target_far": config.target_far,
            "threshold_source": "frozen threshold selected from an independent threshold_selection run",
            "frozen_selection_threshold": config.frozen_selection_threshold,
            "plaintext": {"retrieval": retrieval_metrics(records, identity_split, plain["pairs"], plain["detected_ids"]),
                          "tar_far_test": evaluate_test_threshold(plain["pairs"], frozen, target_far=config.target_far)},
            "encrypted_restore": {"retrieval": retrieval_metrics(records, identity_split, encrypted["pairs"], encrypted["detected_ids"]),
                                  "tar_far_test": evaluate_test_threshold(encrypted["pairs"], frozen, target_far=config.target_far)},
            "mode_consistency": compare_mode_scores(
                {(r["query_id"], r["gallery_id"]): r["similarity"] for r in plain["pairs"]},
                {(r["query_id"], r["gallery_id"]): r["similarity"] for r in encrypted["pairs"]}),
            "cross_partition_comparisons": 0, "stage_timings": stage_timings,
        }
    metrics["total_wall_seconds"] = time.perf_counter() - global_started
    metrics["matrix_audit"] = {
        side: {"expected_pairs": sum(1 for row in plain["pairs"] if row["split"] == side),
               "expected": len(query_by_split[side]) * len(gallery_by_split[side]),
               "cross_partition_pairs": 0}
        for side in active_sides
    }
    metrics["encryption_integrity"] = {
        "gallery_video_count": sum(len(gallery_by_split[side]) for side in active_sides),
        "all_source_restore_sha256_equal": all(x["source_sha256"] == x["restored_sha256"] for x in encrypted["crypto_timings"]),
        "query_images_encrypted": False,
        "description": "Gallery videos encrypted, restored, integrity-verified, then searched as plaintext; query still images are shared unchanged.",
    }
    # Persist only safe, reconstructible configuration and feature metadata.
    config_json = {
        "protocol": "identity-disjoint-query-gallery",
        "manifest_path": str(config.manifest_path.resolve()),
        "manifest_sha256": _sha256(config.manifest_path),
        "identity_split_path": str(config.identity_split_path.resolve()),
        "identity_split_sha256": _sha256(config.identity_split_path),
        "identity_split_seed": split_json.get("seed"),
        "identity_split_counts": split_json.get("split_counts"),
        "ytf_manifest_dataset": manifest_json.get("dataset"),
        "gallery_input_note": "Gallery MP4 files are 1 fps reconstructed experiments inputs from YTF frame images; they are not original YouTube MP4 videos.",
        "query_and_gallery_ids_shared_between_modes": True,
        "model_files": {"yunet_path": str(config.detection.model_path), "yunet_sha256": _sha256(config.detection.model_path),
                        **({"arcface_path": str(config.arcface_model_path), "arcface_sha256": _sha256(config.arcface_model_path)}
                           if config.recognition_model == "arcface" else
                           {"sface_path": str(config.embedding.model_path), "sface_sha256": _sha256(config.embedding.model_path)})},
        "detection": asdict(config.detection), "embedding": asdict(config.embedding),
        "recognition_model": config.recognition_model,
        "arcface": (asdict(ArcFaceConfig(config.arcface_model_path))
                     if config.recognition_model == "arcface" else None),
        "top_k": config.top_k, "target_far": config.target_far,
        "query_use_annotation_box": config.query_use_annotation_box,
        "evaluation_scope": config.evaluation_scope,
        "frozen_selection_threshold": config.frozen_selection_threshold,
        "gallery_representation": {
            "enabled": config.gallery_quality_filter or config.gallery_feature_aggregation != "none",
            "sampling": ("uniformly select at most N frames from existing configured 1 fps samples"
                         if config.gallery_quality_filter else "all existing configured 1 fps samples"),
            "max_sampled_frames": config.gallery_max_sampled_frames,
            "minimum_detection_confidence": config.gallery_min_detection_confidence,
            "minimum_face_area_ratio": config.gallery_min_face_area_ratio,
            "minimum_bbox_visibility": config.gallery_min_bbox_visibility,
            "per_frame_face": ("highest confidence eligible face, area tie-break"
                               if config.gallery_quality_filter else "all YuNet detections"),
            "aggregation": ("L2-normalize each vector, arithmetic mean, then L2-normalize"
                            if config.gallery_feature_aggregation == "normalized_mean" else "none; score-time max over frame features"),
        },
        "similarity_query_batch_size": config.similarity_query_batch_size,
        "feature_cache": "in-memory per media and per mode; embedding vectors are not exported",
        "threshold_source": "plaintext threshold_selection pairs only; frozen for both test modes",
        "encryption": {"chunk_size": config.encryption_chunk_size, "key_version": config.encryption_key_version,
                        "workers": config.encryption_workers},
    }
    feature_summary = {mode: [
        {"cache_index": index, "identity_id": item.identity_id, "video_id": item.video_id,
         "split": item.split, "media_type": item.media_type,
         "feature_dimension": (512 if config.recognition_model == "arcface" else config.embedding.expected_dimension),
         "feature_count": len(item.features), "sampled_frame_count": item.sampled_frame_count,
         "detected_face_count": next(x["detected_face_count"] for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "feature_extraction_failure_count": next(x["feature_extraction_failure_count"] for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "gallery_quality_rejection_count": next(x.get("gallery_quality_rejection_count", 0) for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "gallery_valid_frame_feature_count": next(x.get("gallery_valid_frame_feature_count", 0) for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "detection_failure_count": next(x["detection_failure_count"] for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "invalid_detection_record_count": next(x["invalid_detection_record_count"] for x in data["feature_timings"] if x["media_id"] == item.video_id),
         "source_sha256": item.source_sha256, "restored_sha256": item.restored_sha256}
        for index, item in enumerate(data["feature_cache"].values())
    ] for mode, data in mode_data.items()}
    timing_rows = []
    for mode, data in mode_data.items():
        timing_rows.extend(_timing_rows(mode, data["feature_timings"], data["crypto_timings"],
                                        data["similarity_seconds"], data["ranking_seconds"], data["total_seconds"]))
    pair_export = []
    for mode, data in mode_data.items():
        pair_export.extend({"mode": mode, **row} for row in data["pairs"])
    candidate_export = [row for mode in ("plaintext", "encrypted_restore") for row in mode_data[mode]["candidate_results"]]
    query_export = [row for mode in ("plaintext", "encrypted_restore") for row in mode_data[mode]["query_results"]]
    output_dir = config.output_dir
    failure_export = [row for mode in ("plaintext", "encrypted_restore")
                      for row in mode_data[mode]["feature_failures"]]
    scoring_export = {mode: mode_data[mode]["scoring_diagnostics"]
                      for mode in ("plaintext", "encrypted_restore")}
    json_outputs = {"config.json": config_json, "feature_summary.json": feature_summary,
                    "scoring_diagnostics.json": scoring_export, "metrics.json": metrics}
    for filename, payload in json_outputs.items():
        path = output_dir / filename
        if path.exists():
            raise YTFRunnerError(f"refusing to overwrite: {path}")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    _write_csv(output_dir / "query_results.csv", query_export)
    _write_csv(output_dir / "candidate_results.csv", candidate_export)
    _write_csv(output_dir / "genuine_impostor_pairs.csv", pair_export,
               ("mode", "query_id", "gallery_id", "similarity", "pair_type", "split", "score_status"))
    _write_csv(output_dir / "timing.csv", timing_rows,
               ("mode", "media_id", "split", "stage", "seconds", "measurement"))
    _write_csv(output_dir / "feature_failures.csv", failure_export,
               ("mode", "media_id", "identity_id", "frame_index", "face_index", "stage",
                "exception_type", "exception_message"))
    failure_counts: dict[str, int] = {}
    for failure in failure_export:
        stage = str(failure.get("stage") or "unknown")
        failure_counts[stage] = failure_counts.get(stage, 0) + 1
    failure_media_count = len({(row.get("mode"), row.get("media_id")) for row in failure_export})
    report = [
        "# YTF offline Query-Gallery experiment",
        "",
        "Gallery inputs are MP4 files reconstructed at 1 fps from YTF frame images; they are not original YouTube videos.",
        "The encrypted_restore path encrypts Gallery MP4 files, restores and integrity-verifies them, then searches restored plaintext. It is not ciphertext-domain retrieval. Query still images are shared unencrypted inputs.",
        ("Query annotation-box selection is enabled." if config.query_use_annotation_box else "Query images use every YuNet-detected face with a valid embedding.") +
        f" Gallery quality filter: {config.gallery_quality_filter}; feature aggregation: {config.gallery_feature_aggregation}; pair scoring: {config.score_aggregation}. A non-target Query face may contribute to a pair score.",
        "Unscored pairs are rejected and remain in all-trial TAR/FAR denominators; FAR resolution uses only scored impostor pairs. TAR@FAR=1e-3 is not_estimable below 1,000 scored impostor pairs.",
        "Temporary restored plaintext is stored outside the result bundle and cleaned on normal completion and handled exceptions. Forced process termination or machine/power failure may leave temporary files behind; automatic cleanup is not guaranteed in those cases.",
        "The Runner does not persist feature caches or support resume. Output is staged and published as a complete directory only after bundle validation.",
        "Feature failure records preserve mode, media, identity, frame, face index, stage, exception type, and message when available. YuNet detection exceptions, malformed detection rows, SFace/extraction failures, and feature validation failures are distinguished; failed embeddings remain excluded and are not replaced with scores.",
        f"- Diagnostic failure records: {len(failure_export)}; affected mode/media pairs: {failure_media_count}; stage counts: {json.dumps(failure_counts, ensure_ascii=False, sort_keys=True)}",
        "- scoring_diagnostics.json records per split and query the input matrix dimensions/dtypes, finite and norm summaries, rejected vector indices/reasons, normalization checks, and matrix multiplication output checks. Feature vector values are not exported.",
        "",
        f"- Evaluation scope: {config.evaluation_scope}",
        f"- Identities: {len(identity_split)} (threshold selection {sum(x == THRESHOLD_SELECTION for x in identity_split.values())}; test {sum(x == TEST for x in identity_split.values())})",
        f"- Pair rows per mode: {len(plain['pairs'])}; expected within-partition matrix rows: {sum(len(query_by_split[s]) * len(gallery_by_split[s]) for s in active_sides)}",
        f"- Threshold: {json.dumps(metrics.get('threshold_selection', {'frozen_selection_threshold': config.frozen_selection_threshold}), ensure_ascii=False, sort_keys=True)}",
        f"- Mode consistency: {json.dumps(metrics['mode_consistency'], ensure_ascii=False, sort_keys=True)}",
        f"- Total wall seconds: {metrics['total_wall_seconds']:.6f}",
        "",
        "Metrics and unscored-pair counts are in metrics.json; raw per-query, candidate, and pair records are in the corresponding CSV files.",
    ]
    report_path = output_dir / "report.md"
    if report_path.exists():
        raise YTFRunnerError(f"refusing to overwrite: {report_path}")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def _validate_result_bundle(directory: Path) -> None:
    expected = {"config.json", "feature_summary.json", "query_results.csv",
                "candidate_results.csv", "genuine_impostor_pairs.csv", "metrics.json",
                "timing.csv", "report.md", "feature_failures.csv", "scoring_diagnostics.json"}
    missing = sorted(name for name in expected if not (directory / name).is_file())
    if missing:
        raise YTFRunnerError(f"staged result bundle is incomplete: {missing}")
    payloads = {}
    for filename in ("config.json", "feature_summary.json", "metrics.json", "scoring_diagnostics.json"):
        try:
            payloads[filename] = json.loads((directory / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise YTFRunnerError(f"staged result JSON is invalid: {filename}") from exc
    scoring = payloads["scoring_diagnostics.json"]
    if set(scoring) != {"plaintext", "encrypted_restore"} or any(
            not isinstance(rows, list) for rows in scoring.values()):
        raise YTFRunnerError("staged scoring diagnostics schema is incomplete")
    for filename in expected - {"config.json", "feature_summary.json", "metrics.json"}:
        if (directory / filename).stat().st_size == 0:
            raise YTFRunnerError(f"staged result file is empty: {filename}")
    required_columns = {
        "query_results.csv": {"mode", "query_id", "query_identity_id", "query_detected_face_count", "query_feature_count"},
        "candidate_results.csv": {"mode", "query_id", "candidate_video_id", "similarity", "rank"},
        "genuine_impostor_pairs.csv": {"mode", "query_id", "gallery_id", "similarity", "pair_type", "split"},
        "timing.csv": {"mode", "media_id", "stage", "seconds"},
        "feature_failures.csv": {"mode", "media_id", "identity_id", "frame_index", "face_index", "stage", "exception_type", "exception_message"},
    }
    for filename, columns in required_columns.items():
        with (directory / filename).open("r", encoding="utf-8-sig", newline="") as stream:
            header = set(next(csv.reader(stream), []))
        if not columns <= header:
            raise YTFRunnerError(f"staged CSV schema is incomplete: {filename}")
    audits = payloads["metrics.json"].get("matrix_audit", {})
    if not audits or any(row.get("cross_partition_pairs") != 0 or
                         row.get("expected_pairs") != row.get("expected") for row in audits.values()):
        raise YTFRunnerError("staged similarity matrix audit is incomplete")


def run_ytf_runner(config: RunnerConfig, *, encryption_key: bytes, mac_key: bytes,
                   backend_factory: Callable[[FaceDetectionConfig, SFaceConfig], Any] = OpenCVFeatureBackend,
                   encrypt: Callable[..., Any] = store_encrypted_video,
                   restore: Callable[..., Any] = restore_encrypted_video) -> dict[str, Any]:
    """Run paired modes and publish a complete result directory atomically."""
    config.validate()
    if not encryption_key or not mac_key:
        raise YTFRunnerError("both encryption_key and mac_key are required")
    target = config.output_dir.absolute()
    if target.is_symlink():
        raise YTFRunnerError(f"output path is a symbolic link; refusing to use: {target}")
    if target.exists():
        if target.is_dir() and any(target.iterdir()):
            raise YTFRunnerError(f"output directory is not empty; refusing to overwrite: {target}")
        raise YTFRunnerError(f"output directory already exists; refusing to replace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    published = False
    try:
        # Restored plaintext videos and encrypted chunks stay in a private
        # system temporary directory, never in the published result bundle.
        with tempfile.TemporaryDirectory(prefix="facecheck-ytf-work-") as work:
            result = _run_ytf_runner_impl(
                replace(config, output_dir=staging), encryption_key=encryption_key,
                mac_key=mac_key, backend_factory=backend_factory, encrypt=encrypt,
                restore=restore, work_root=Path(work),
            )
        _validate_result_bundle(staging)
        if target.exists() or target.is_symlink():
            raise YTFRunnerError(f"output directory appeared during run; refusing to overwrite: {target}")
        os.replace(staging, target)
        published = True
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _key_from_env(name: str) -> bytes:
    raw = os.environ.get(name)
    if not raw:
        raise YTFRunnerError(f"environment variable {name} must contain hex key bytes")
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        raise YTFRunnerError(f"environment variable {name} must be hexadecimal") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the YTF identity-disjoint Query-Gallery experiment")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yunet-model", type=Path, required=True)
    parser.add_argument("--sface-model", type=Path, default=DEFAULT_SFACE_PATH)
    parser.add_argument("--recognition-model", choices=("sface", "arcface"), default=DEFAULT_RECOGNITION_MODEL)
    parser.add_argument("--arcface-model", type=Path, default=DEFAULT_ARCFACE_PATH)
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-far", type=float, default=1e-3)
    parser.add_argument("--score-aggregation", choices=("max", "top_k_mean"), default="max")
    parser.add_argument("--score-aggregation-top-n", type=int, default=3)
    parser.add_argument("--encryption-workers", type=int, default=4)
    parser.add_argument("--query-use-annotation-box", action="store_true")
    parser.add_argument("--evaluation-scope", choices=("full", THRESHOLD_SELECTION, TEST), default="full")
    parser.add_argument("--frozen-selection-threshold", type=float)
    parser.add_argument("--gallery-max-sampled-frames", type=int, default=12)
    parser.add_argument("--gallery-min-detection-confidence", type=float, default=0.9)
    parser.add_argument("--gallery-min-face-area-ratio", type=float, default=0.015)
    parser.add_argument("--gallery-min-bbox-visibility", type=float, default=0.75)
    args = parser.parse_args()
    if (args.evaluation_scope == TEST and args.recognition_model == "arcface"
            and args.frozen_selection_threshold is None):
        args.frozen_selection_threshold = DEFAULT_ARCFACE_THRESHOLD
    config = RunnerConfig(
        args.manifest, args.identity_split, args.output,
        FaceDetectionConfig(args.yunet_model, score_threshold=args.score_threshold,
                            frame_interval_seconds=args.frame_interval),
        (ArcFaceConfig(args.arcface_model) if args.recognition_model == "arcface" else SFaceConfig(args.sface_model)),
        top_k=args.top_k, target_far=args.target_far,
        recognition_model=args.recognition_model, arcface_model_path=args.arcface_model,
        score_aggregation=args.score_aggregation,
        score_aggregation_top_n=args.score_aggregation_top_n,
        encryption_workers=args.encryption_workers,
        query_use_annotation_box=args.query_use_annotation_box,
        evaluation_scope=args.evaluation_scope,
        frozen_selection_threshold=args.frozen_selection_threshold,
        gallery_max_sampled_frames=args.gallery_max_sampled_frames,
        gallery_min_detection_confidence=args.gallery_min_detection_confidence,
        gallery_min_face_area_ratio=args.gallery_min_face_area_ratio,
        gallery_min_bbox_visibility=args.gallery_min_bbox_visibility,
    )
    run_ytf_runner(config, encryption_key=_key_from_env("FACECHECK_EXPERIMENT_ENCRYPTION_KEY"),
                   mac_key=_key_from_env("FACECHECK_EXPERIMENT_MAC_KEY"))


if __name__ == "__main__":
    main()
