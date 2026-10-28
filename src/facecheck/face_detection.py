from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Tuple

import cv2


class FaceDetectionError(ValueError):
    """Raised when face detection cannot complete safely."""


@dataclass(frozen=True)
class FaceDetectionConfig:
    model_path: Path
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000
    frame_interval_seconds: float = 1.0
    max_duration_seconds: float = 600.0
    max_frames: int = 5000
    max_processing_seconds: float = 60.0

    def validate(self) -> None:
        if self.frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be greater than zero")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than zero")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        if self.max_processing_seconds <= 0:
            raise ValueError("max_processing_seconds must be greater than zero")
        if not 0 < self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0 < self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


@dataclass(frozen=True)
class FaceDetection:
    frame_index: int
    timestamp_seconds: float
    video_width: int
    video_height: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    landmarks: Optional[Tuple[Tuple[float, float], ...]]


@dataclass(frozen=True)
class FaceDetectionReport:
    video_width: int
    video_height: int
    fps: float
    frame_count: int
    sampled_frame_count: int
    detections: List[FaceDetection]


class FaceDetector(Protocol):
    def setInputSize(self, input_size: Tuple[int, int]) -> None:
        ...

    def detect(self, image: Any) -> Tuple[Any, Any]:
        ...


DetectorFactory = Callable[[FaceDetectionConfig], FaceDetector]
CaptureFactory = Callable[[str], Any]


def load_yunet_detector(config: FaceDetectionConfig) -> FaceDetector:
    config.validate()
    if config.model_path.is_symlink() or not config.model_path.is_file():
        raise FaceDetectionError("YuNet model file is missing")
    try:
        return cv2.FaceDetectorYN.create(
            str(config.model_path),
            "",
            (320, 320),
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )
    except Exception as exc:
        raise FaceDetectionError("YuNet model could not be loaded") from exc


def _normalise_faces(raw_faces: Any) -> List[Tuple[float, ...]]:
    if raw_faces is None:
        return []
    try:
        return [tuple(float(value) for value in row) for row in raw_faces]
    except (TypeError, ValueError) as exc:
        raise FaceDetectionError("YuNet returned invalid detection data") from exc


def _to_detections(
    faces: List[Tuple[float, ...]],
    frame_index: int,
    timestamp: float,
    width: int,
    height: int,
) -> List[FaceDetection]:
    detections: List[FaceDetection] = []
    for face in faces:
        if len(face) < 15:
            raise FaceDetectionError("YuNet returned an incomplete face record")
        landmarks = tuple(
            (face[index], face[index + 1]) for index in range(4, 14, 2)
        )
        detections.append(
            FaceDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                video_width=width,
                video_height=height,
                x=face[0],
                y=face[1],
                width=face[2],
                height=face[3],
                confidence=face[14],
                landmarks=landmarks,
            )
        )
    return detections


def detect_video(
    video_path: Path,
    config: FaceDetectionConfig,
    *,
    detector_factory: Optional[DetectorFactory] = None,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    clock: Callable[[], float] = time.monotonic,
) -> FaceDetectionReport:
    config.validate()
    detector = (
        detector_factory(config)
        if detector_factory is not None
        else load_yunet_detector(config)
    )
    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FaceDetectionError("video could not be opened")
    started = clock()
    sampled_count = 0
    frame_index = 0
    next_sample = 0.0
    detections: List[FaceDetection] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise FaceDetectionError("video metadata is unavailable or invalid")
        if frame_count / fps > config.max_duration_seconds:
            raise FaceDetectionError("video exceeds the maximum duration")
        while True:
            if clock() - started > config.max_processing_seconds:
                raise FaceDetectionError("face detection exceeded the time limit")
            try:
                success, image = capture.read()
            except Exception as exc:
                raise FaceDetectionError("video frame could not be read") from exc
            if not success:
                break
            if frame_index >= config.max_frames:
                raise FaceDetectionError("video exceeds the maximum frame limit")
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp < 0:
                timestamp = frame_index / fps
            if timestamp + 1e-9 >= next_sample:
                sampled_count += 1
                detector.setInputSize((width, height))
                try:
                    _, raw_faces = detector.detect(image)
                except Exception as exc:
                    raise FaceDetectionError("YuNet detection failed") from exc
                detections.extend(
                    _to_detections(
                        _normalise_faces(raw_faces),
                        frame_index,
                        timestamp,
                        width,
                        height,
                    )
                )
                next_sample += config.frame_interval_seconds
            frame_index += 1
        return FaceDetectionReport(
            video_width=width,
            video_height=height,
            fps=fps,
            frame_count=frame_count,
            sampled_frame_count=sampled_count,
            detections=detections,
        )
    finally:
        capture.release()
