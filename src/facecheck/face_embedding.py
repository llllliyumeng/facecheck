from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from .face_detection import FaceDetection


class FaceEmbeddingError(ValueError):
    """Raised when an SFace embedding cannot be produced safely."""

    def __init__(self, message: str, *, stage: str = "feature_extraction"):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class SFaceConfig:
    model_path: Path
    expected_dimension: int = 128
    backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV
    target_id: int = cv2.dnn.DNN_TARGET_CPU

    def validate(self) -> None:
        if self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")


@dataclass(frozen=True)
class ArcFaceConfig:
    """Configuration for an ArcFace ONNX embedding model.

    The public InsightFace w600k_r50 weight uses the standard 112x112,
    five-landmark ArcFace crop and emits a 512-dimensional embedding.
    """
    model_path: Path
    expected_dimension: int = 512
    input_size: tuple[int, int] = (112, 112)
    backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV
    target_id: int = cv2.dnn.DNN_TARGET_CPU

    def validate(self) -> None:
        if self.expected_dimension != 512 or self.input_size != (112, 112):
            raise ValueError("w600k_r50 requires 512 dimensions and 112x112 input")


class SFaceRecognizer(Protocol):
    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        ...

    def feature(self, aligned_face: np.ndarray) -> np.ndarray:
        ...


class ArcFaceRecognizer(Protocol):
    def setInput(self, blob: np.ndarray) -> None:
        ...

    def forward(self) -> np.ndarray:
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


def load_arcface_recognizer(config: ArcFaceConfig) -> ArcFaceRecognizer:
    config.validate()
    if config.model_path.is_symlink() or not config.model_path.is_file():
        raise FaceEmbeddingError("ArcFace model file is missing")
    try:
        net = cv2.dnn.readNetFromONNX(str(config.model_path))
        net.setPreferableBackend(config.backend_id)
        net.setPreferableTarget(config.target_id)
        return net
    except Exception as exc:
        raise FaceEmbeddingError("ArcFace model could not be loaded") from exc


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise FaceEmbeddingError("image must be a NumPy array", stage="feature_validation")
    if image.ndim != 3 or image.shape[2] != 3:
        raise FaceEmbeddingError("image must be a non-empty BGR image", stage="feature_validation")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise FaceEmbeddingError("image must be a non-empty BGR image")


def face_record_from_detection(
    detection: FaceDetection,
    image_shape: Sequence[int],
    *,
    clip_to_image: bool = False,
) -> np.ndarray:
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise FaceEmbeddingError("image must have three color channels", stage="feature_validation")
    height, width = int(image_shape[0]), int(image_shape[1])
    values = (
        detection.x,
        detection.y,
        detection.width,
        detection.height,
    )
    if not all(np.isfinite(value) for value in values):
        raise FaceEmbeddingError("face bounding box contains invalid values", stage="feature_validation")
    x, y, box_width, box_height = (
        float(detection.x), float(detection.y),
        float(detection.width), float(detection.height),
    )
    if box_width <= 0 or box_height <= 0:
        raise FaceEmbeddingError("face bounding box is outside the image", stage="feature_validation")
    if clip_to_image:
        right = min(float(width), x + box_width)
        bottom = min(float(height), y + box_height)
        x = max(0.0, x)
        y = max(0.0, y)
        box_width = right - x
        box_height = bottom - y
        if box_width <= 0 or box_height <= 0:
            raise FaceEmbeddingError("face bounding box is outside the image", stage="feature_validation")
    elif (
        x < 0
        or y < 0
        or x + box_width > width
        or y + box_height > height
    ):
        raise FaceEmbeddingError("face bounding box is outside the image", stage="feature_validation")
    if detection.landmarks is None or len(detection.landmarks) != 5:
        raise FaceEmbeddingError("SFace requires five face landmarks", stage="feature_validation")
    landmarks = []
    max_x = max(0.0, float(width) - 1e-3)
    max_y = max(0.0, float(height) - 1e-3)
    for point in detection.landmarks:
        if len(point) != 2 or not all(np.isfinite(value) for value in point):
            raise FaceEmbeddingError("face landmarks contain invalid values", stage="feature_validation")
        if clip_to_image:
            point_x = min(max(float(point[0]), 0.0), max_x)
            point_y = min(max(float(point[1]), 0.0), max_y)
        else:
            point_x, point_y = float(point[0]), float(point[1])
        if not 0 <= point_x < width or not 0 <= point_y < height:
            raise FaceEmbeddingError("face landmarks are outside the image", stage="feature_validation")
        landmarks.extend((point_x, point_y))
    return np.asarray(
        [
            x,
            y,
            box_width,
            box_height,
            *landmarks,
            detection.confidence,
        ],
        dtype=np.float32,
    )


def clip_detection_to_image(
    detection: FaceDetection,
    image_shape: Sequence[int],
) -> FaceDetection:
    """Validate and clamp YuNet geometry to the decoded image bounds."""
    record = face_record_from_detection(
        detection, image_shape, clip_to_image=True
    )
    return FaceDetection(
        frame_index=detection.frame_index,
        timestamp_seconds=detection.timestamp_seconds,
        video_width=detection.video_width,
        video_height=detection.video_height,
        x=float(record[0]), y=float(record[1]),
        width=float(record[2]), height=float(record[3]),
        confidence=detection.confidence,
        landmarks=tuple(
            (float(record[index]), float(record[index + 1]))
            for index in range(4, 14, 2)
        ),
    )


def _normalise_feature(
    raw_feature: np.ndarray,
    expected_dimension: int,
) -> np.ndarray:
    try:
        feature = np.asarray(raw_feature)
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingError("SFace returned an invalid feature", stage="feature_validation") from exc
    if feature.size != expected_dimension:
        raise FaceEmbeddingError("SFace returned an unexpected feature dimension", stage="feature_validation")
    if feature.ndim not in (1, 2):
        raise FaceEmbeddingError("SFace returned an invalid feature shape", stage="feature_validation")
    if feature.ndim == 2 and feature.shape != (1, expected_dimension):
        raise FaceEmbeddingError("SFace returned an invalid feature shape", stage="feature_validation")
    if feature.ndim == 1 and feature.shape != (expected_dimension,):
        raise FaceEmbeddingError("SFace returned an invalid feature shape")
    if not np.issubdtype(feature.dtype, np.number):
        raise FaceEmbeddingError("SFace returned a non-numeric feature", stage="feature_validation")
    if not np.all(np.isfinite(feature)):
        raise FaceEmbeddingError("SFace returned a feature with invalid values", stage="feature_validation")
    if np.any(np.abs(feature.astype(np.float64, copy=False)) > np.finfo(np.float32).max):
        raise FaceEmbeddingError("SFace returned a feature outside float32 range", stage="feature_validation")
    feature = feature.astype(np.float32, copy=True).reshape(expected_dimension)
    if not np.all(np.isfinite(feature)):
        raise FaceEmbeddingError("SFace returned a feature with invalid values", stage="feature_validation")
    norm = float(np.linalg.norm(feature.astype(np.float64)))
    if not np.isfinite(norm) or norm == 0:
        raise FaceEmbeddingError("SFace returned a zero feature", stage="feature_validation")
    return np.ascontiguousarray((feature.astype(np.float64) / norm).astype(np.float32))


def extract_embedding(
    image: np.ndarray,
    detection: FaceDetection,
    config: SFaceConfig,
    *,
    recognizer_factory: Optional[RecognizerFactory] = None,
    clip_to_image: bool = False,
) -> FaceEmbedding:
    config.validate()
    _validate_image(image)
    face = face_record_from_detection(detection, image.shape, clip_to_image=clip_to_image)
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


_ARCFACE_TEMPLATE = np.asarray([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def _arcface_alignment(source: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform using all five landmarks.

    Like InsightFace norm_crop, this fits scale, rotation and translation;
    there is no random sampling, shear, reflection, or outlier rejection.
    """
    points = np.asarray(source, dtype=np.float64)
    if (points.shape != (5, 2) or not np.all(np.isfinite(points)) or
            np.linalg.matrix_rank(points - points.mean(axis=0)) < 2):
        raise FaceEmbeddingError("ArcFace landmarks are degenerate", stage="feature_validation")
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, 0] = points[:, 0]
    design[0::2, 1] = -points[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 0] = points[:, 1]
    design[1::2, 1] = points[:, 0]
    design[1::2, 3] = 1.0
    ok, parameters = cv2.solve(design, _ARCFACE_TEMPLATE.astype(np.float64).reshape(10, 1),
                               flags=cv2.DECOMP_SVD)
    if not ok or not np.all(np.isfinite(parameters)):
        raise FaceEmbeddingError("ArcFace alignment failed")
    a, b, tx, ty = parameters.reshape(4)
    if a * a + b * b <= np.finfo(np.float64).eps:
        raise FaceEmbeddingError("ArcFace alignment scale is zero")
    return np.asarray([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def extract_arcface_embedding(
    image: np.ndarray,
    detection: FaceDetection,
    config: ArcFaceConfig,
    *,
    recognizer_factory: Optional[Callable[[ArcFaceConfig], ArcFaceRecognizer]] = None,
) -> FaceEmbedding:
    config.validate()
    _validate_image(image)
    face = face_record_from_detection(detection, image.shape, clip_to_image=False)
    recognizer = recognizer_factory(config) if recognizer_factory else load_arcface_recognizer(config)
    source = face[4:14].reshape(5, 2).astype(np.float32)
    try:
        transform = _arcface_alignment(source)
        aligned = cv2.warpAffine(image, transform, config.input_size, borderValue=0)
        blob = cv2.dnn.blobFromImage(
            aligned, scalefactor=1.0 / 127.5, size=config.input_size,
            mean=(127.5, 127.5, 127.5), swapRB=True, crop=False,
        )
        recognizer.setInput(blob)
        raw_feature = recognizer.forward()
    except FaceEmbeddingError:
        raise
    except Exception as exc:
        raise FaceEmbeddingError("ArcFace feature extraction failed") from exc
    return FaceEmbedding(vector=_normalise_feature(raw_feature, config.expected_dimension))


def _validate_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    if not isinstance(vector, np.ndarray):
        raise FaceEmbeddingError("embedding must be a NumPy array", stage="feature_validation")
    if vector.shape != (dimension,):
        raise FaceEmbeddingError("embedding has an invalid shape", stage="feature_validation")
    if not np.issubdtype(vector.dtype, np.number):
        raise FaceEmbeddingError("embedding must be numeric", stage="feature_validation")
    if not np.all(np.isfinite(vector)):
        raise FaceEmbeddingError("embedding contains invalid values", stage="feature_validation")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise FaceEmbeddingError("embedding must have a non-zero norm", stage="feature_validation")
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
