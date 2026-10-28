from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from .face_detection import FaceDetection


class FaceEmbeddingError(ValueError):
    """Raised when an SFace embedding cannot be produced safely."""


@dataclass(frozen=True)
class SFaceConfig:
    model_path: Path
    expected_dimension: int = 128
    backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV
    target_id: int = cv2.dnn.DNN_TARGET_CPU

    def validate(self) -> None:
        if self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")


class SFaceRecognizer(Protocol):
    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        ...

    def feature(self, aligned_face: np.ndarray) -> np.ndarray:
        ...


RecognizerFactory = Callable[[SFaceConfig], SFaceRecognizer]


@dataclass(frozen=True)
class FaceEmbedding:
    vector: np.ndarray


@dataclass(frozen=True)
class EmbeddingMatch:
    sample_id: str
    similarity: float


def load_sface_recognizer(config: SFaceConfig) -> SFaceRecognizer:
    config.validate()
    if config.model_path.is_symlink() or not config.model_path.is_file():
        raise FaceEmbeddingError("SFace model file is missing")
    try:
        return cv2.FaceRecognizerSF_create(
            str(config.model_path),
            "",
            config.backend_id,
            config.target_id,
        )
    except Exception as exc:
        raise FaceEmbeddingError("SFace model could not be loaded") from exc


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise FaceEmbeddingError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise FaceEmbeddingError("image must be a non-empty BGR image")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise FaceEmbeddingError("image must be a non-empty BGR image")


def face_record_from_detection(
    detection: FaceDetection,
    image_shape: Sequence[int],
) -> np.ndarray:
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise FaceEmbeddingError("image must have three color channels")
    height, width = int(image_shape[0]), int(image_shape[1])
    values = (
        detection.x,
        detection.y,
        detection.width,
        detection.height,
    )
    if not all(np.isfinite(value) for value in values):
        raise FaceEmbeddingError("face bounding box contains invalid values")
    if (
        detection.width <= 0
        or detection.height <= 0
        or detection.x < 0
        or detection.y < 0
        or detection.x + detection.width > width
        or detection.y + detection.height > height
    ):
        raise FaceEmbeddingError("face bounding box is outside the image")
    if detection.landmarks is None or len(detection.landmarks) != 5:
        raise FaceEmbeddingError("SFace requires five face landmarks")
    landmarks = []
    for point in detection.landmarks:
        if len(point) != 2 or not all(np.isfinite(value) for value in point):
            raise FaceEmbeddingError("face landmarks contain invalid values")
        if not 0 <= point[0] < width or not 0 <= point[1] < height:
            raise FaceEmbeddingError("face landmarks are outside the image")
        landmarks.extend(point)
    return np.asarray(
        [
            detection.x,
            detection.y,
            detection.width,
            detection.height,
            *landmarks,
            detection.confidence,
        ],
        dtype=np.float32,
    )


def _normalise_feature(
    raw_feature: np.ndarray,
    expected_dimension: int,
) -> np.ndarray:
    try:
        feature = np.asarray(raw_feature)
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingError("SFace returned an invalid feature") from exc
    if feature.size != expected_dimension:
        raise FaceEmbeddingError("SFace returned an unexpected feature dimension")
    if feature.ndim not in (1, 2):
        raise FaceEmbeddingError("SFace returned an invalid feature shape")
    if feature.ndim == 2 and feature.shape != (1, expected_dimension):
        raise FaceEmbeddingError("SFace returned an invalid feature shape")
    if feature.ndim == 1 and feature.shape != (expected_dimension,):
        raise FaceEmbeddingError("SFace returned an invalid feature shape")
    if not np.issubdtype(feature.dtype, np.number):
        raise FaceEmbeddingError("SFace returned a non-numeric feature")
    feature = feature.astype(np.float32, copy=True).reshape(expected_dimension)
    if not np.all(np.isfinite(feature)):
        raise FaceEmbeddingError("SFace returned a feature with invalid values")
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm == 0:
        raise FaceEmbeddingError("SFace returned a zero feature")
    return np.ascontiguousarray(feature / norm, dtype=np.float32)


def extract_embedding(
    image: np.ndarray,
    detection: FaceDetection,
    config: SFaceConfig,
    *,
    recognizer_factory: Optional[RecognizerFactory] = None,
) -> FaceEmbedding:
    config.validate()
    _validate_image(image)
    face = face_record_from_detection(detection, image.shape)
    recognizer = (
        recognizer_factory(config)
        if recognizer_factory is not None
        else load_sface_recognizer(config)
    )
    try:
        aligned = recognizer.alignCrop(image, face)
        if not isinstance(aligned, np.ndarray) or aligned.size == 0:
            raise FaceEmbeddingError("SFace alignment returned an invalid image")
        raw_feature = recognizer.feature(aligned)
    except FaceEmbeddingError:
        raise
    except Exception as exc:
        raise FaceEmbeddingError("SFace feature extraction failed") from exc
    return FaceEmbedding(
        vector=_normalise_feature(raw_feature, config.expected_dimension)
    )


def _validate_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    if not isinstance(vector, np.ndarray):
        raise FaceEmbeddingError("embedding must be a NumPy array")
    if vector.shape != (dimension,):
        raise FaceEmbeddingError("embedding has an invalid shape")
    if not np.issubdtype(vector.dtype, np.number):
        raise FaceEmbeddingError("embedding must be numeric")
    if not np.all(np.isfinite(vector)):
        raise FaceEmbeddingError("embedding contains invalid values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise FaceEmbeddingError("embedding must have a non-zero norm")
    return vector


def cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    dimension: int = 128,
) -> float:
    left_vector = _validate_vector(left, dimension)
    right_vector = _validate_vector(right, dimension)
    similarity = float(
        np.dot(left_vector, right_vector)
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )
    if not np.isfinite(similarity):
        raise FaceEmbeddingError("similarity is not finite")
    return similarity


def search_embeddings(
    query: np.ndarray,
    candidates: Iterable[Tuple[str, np.ndarray]],
    *,
    top_k: int = 10,
    min_similarity: Optional[float] = None,
    dimension: int = 128,
) -> Tuple[EmbeddingMatch, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if min_similarity is not None and not np.isfinite(min_similarity):
        raise ValueError("min_similarity must be finite")
    _validate_vector(query, dimension)
    matches = []
    for sample_id, vector in candidates:
        if not isinstance(sample_id, str) or not sample_id:
            raise FaceEmbeddingError("sample_id must be a non-empty string")
        similarity = cosine_similarity(query, vector, dimension=dimension)
        if min_similarity is None or similarity >= min_similarity:
            matches.append(EmbeddingMatch(sample_id, similarity))
    matches.sort(key=lambda match: (-match.similarity, match.sample_id))
    return tuple(matches[:top_k])
