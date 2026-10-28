from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoProcessingError(ValueError):
    """Raised when a video cannot be opened or read reliably."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class ExtractedFrame:
    index: int
    timestamp_seconds: float
    image: object


def read_video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"unable to open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise VideoProcessingError(f"video metadata is unavailable or invalid: {path}")
        return VideoInfo(
            path=path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()


def extract_frames(path: Path, interval_seconds: float) -> list[ExtractedFrame]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")
    read_video_info(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"unable to open video: {path}")
    frames: list[ExtractedFrame] = []
    next_timestamp = 0.0
    frame_index = 0
    try:
        while True:
            success, image = capture.read()
            if not success:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp + 1e-9 >= next_timestamp:
                frames.append(
                    ExtractedFrame(
                        index=frame_index,
                        timestamp_seconds=timestamp,
                        image=image,
                    )
                )
                next_timestamp += interval_seconds
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise VideoProcessingError(f"video contains no readable frames: {path}")
    return frames
