from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .face_detection import (
    FaceDetection,
    FaceDetectionConfig,
    FaceDetectionError,
    FaceDetector,
    DetectorFactory,
    load_yunet_detector,
)
from .face_embedding import (
    EmbeddingMatch,
    FaceEmbeddingError,
    RecognizerFactory,
    SFaceConfig,
    extract_embedding,
    load_sface_recognizer,
    search_embeddings,
)
from .storage import restore_encrypted_video
from .video import ExtractedFrame, VideoProcessingError, extract_frames, read_video_info


class FaceSearchError(ValueError):
    """Raised when an in-memory video face search cannot complete safely."""


@dataclass(frozen=True)
class FaceSearchConfig:
    detection: FaceDetectionConfig
    embedding: SFaceConfig
    top_k: int = 10
    min_similarity: Optional[float] = None
    max_processing_seconds: float = 60.0

    def validate(self) -> None:
        self.detection.validate()
        self.embedding.validate()
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.min_similarity is not None and not np.isfinite(self.min_similarity):
            raise ValueError("min_similarity must be finite")
        if self.max_processing_seconds <= 0:
            raise ValueError("max_processing_seconds must be greater than zero")


@dataclass(frozen=True)
class FaceSearchResult:
    candidate_id: str
    frame_index: int
    timestamp_seconds: float
    similarity: float


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    frame_index: int
    timestamp_seconds: float
    vector: np.ndarray


def _detections_from_raw(
    raw_faces: object,
    frame: ExtractedFrame,
) -> List[FaceDetection]:
    if raw_faces is None:
        return []
    try:
        faces = np.asarray(raw_faces, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise FaceSearchError("YuNet returned invalid detection data") from exc
    if faces.ndim != 2 or faces.shape[1] < 15:
        raise FaceSearchError("YuNet returned incomplete detection data")
    height, width = frame.image.shape[:2]
    detections = []
    for row in faces:
        if not np.all(np.isfinite(row[:15])):
            raise FaceSearchError("YuNet returned non-finite detection data")
        detections.append(
            FaceDetection(
                frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                video_width=width,
                video_height=height,
                x=float(row[0]),
                y=float(row[1]),
                width=float(row[2]),
                height=float(row[3]),
                confidence=float(row[14]),
                landmarks=tuple(
                    (float(row[index]), float(row[index + 1]))
                    for index in range(4, 14, 2)
                ),
            )
        )
    return detections


def _search_frames(
    frames: Iterable[ExtractedFrame],
    query: np.ndarray,
    config: FaceSearchConfig,
    *,
    detector_factory: Optional[DetectorFactory],
    recognizer_factory: Optional[RecognizerFactory],
    clock: Callable[[], float],
) -> Tuple[FaceSearchResult, ...]:
    started = clock()
    detector: FaceDetector = (
        detector_factory(config.detection)
        if detector_factory is not None
        else load_yunet_detector(config.detection)
    )
    recognizer = None
    candidates: List[_Candidate] = []
    for frame in frames:
        if clock() - started > config.max_processing_seconds:
            raise FaceSearchError("face search exceeded the time limit")
        try:
            height, width = frame.image.shape[:2]
            detector.setInputSize((width, height))
            _, raw_faces = detector.detect(frame.image)
            detections = _detections_from_raw(raw_faces, frame)
        except FaceSearchError:
            raise
        except Exception as exc:
            raise FaceSearchError(
                f"face detection failed at frame {frame.index}"
            ) from exc
        for face_index, detection in enumerate(detections):
            try:
                if recognizer is None:
                    recognizer = (
                        recognizer_factory(config.embedding)
                        if recognizer_factory is not None
                        else load_sface_recognizer(config.embedding)
                    )
                embedding = extract_embedding(
                    frame.image,
                    detection,
                    config.embedding,
                    recognizer_factory=lambda _, value=recognizer: value,
                )
            except FaceEmbeddingError as exc:
                raise FaceSearchError(
                    f"feature extraction failed at frame {frame.index}, "
                    f"face {face_index}"
                ) from exc
            candidates.append(
                _Candidate(
                    candidate_id=f"frame-{frame.index}-face-{face_index}",
                    frame_index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    vector=embedding.vector,
                )
            )

    matches = search_embeddings(
        query,
        ((candidate.candidate_id, candidate.vector) for candidate in candidates),
        top_k=config.top_k,
        min_similarity=config.min_similarity,
        dimension=config.embedding.expected_dimension,
    )
    metadata: Dict[str, _Candidate] = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    return tuple(
        FaceSearchResult(
            candidate_id=match.sample_id,
            frame_index=metadata[match.sample_id].frame_index,
            timestamp_seconds=metadata[match.sample_id].timestamp_seconds,
            similarity=match.similarity,
        )
        for match in matches
    )


def search_video(
    video_path: Path,
    query: np.ndarray,
    config: FaceSearchConfig,
    *,
    detector_factory: Optional[DetectorFactory] = None,
    recognizer_factory: Optional[RecognizerFactory] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[FaceSearchResult, ...]:
    config.validate()
    try:
        info = read_video_info(video_path)
        if info.duration_seconds > config.detection.max_duration_seconds:
            raise FaceSearchError("video exceeds the maximum duration")
        frames = extract_frames(
            video_path,
            config.detection.frame_interval_seconds,
        )
        if len(frames) > config.detection.max_frames:
            raise FaceSearchError("video exceeds the maximum frame limit")
        return _search_frames(
            frames,
            query,
            config,
            detector_factory=detector_factory,
            recognizer_factory=recognizer_factory,
            clock=clock,
        )
    except FaceSearchError:
        raise
    except (FaceDetectionError, VideoProcessingError, FaceEmbeddingError) as exc:
        raise FaceSearchError("video face search could not be completed") from exc


def search_encrypted_video(
    stored_directory: Path,
    query: np.ndarray,
    config: FaceSearchConfig,
    encryption_key: bytes,
    mac_key: bytes,
    *,
    temp_root: Optional[Path] = None,
    detector_factory: Optional[DetectorFactory] = None,
    recognizer_factory: Optional[RecognizerFactory] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[FaceSearchResult, ...]:
    config.validate()
    try:
        with tempfile.TemporaryDirectory(
            prefix="facecheck-search-",
            dir=str(temp_root) if temp_root is not None else None,
        ) as directory:
            restored_path = Path(directory) / "restored-video"
            restore_encrypted_video(
                stored_directory,
                restored_path,
                encryption_key,
                mac_key,
            )
            return search_video(
                restored_path,
                query,
                config,
                detector_factory=detector_factory,
                recognizer_factory=recognizer_factory,
                clock=clock,
            )
    except FaceSearchError:
        raise
    except Exception as exc:
        raise FaceSearchError(
            "encrypted video face search could not be completed"
        ) from exc
