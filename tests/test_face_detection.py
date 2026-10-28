from pathlib import Path

import cv2
import numpy as np
import pytest

from facecheck.face_detection import (
    FaceDetectionConfig,
    FaceDetectionError,
    detect_video,
    load_yunet_detector,
)


def _write_video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (24, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v writer unavailable")
    try:
        for index in range(count):
            writer.write(np.full((16, 24, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.input_sizes = []

    def setInputSize(self, input_size):
        self.input_sizes.append(input_size)

    def detect(self, image):
        return None, self.faces


def _config(tmp_path: Path, **kwargs) -> FaceDetectionConfig:
    return FaceDetectionConfig(model_path=tmp_path / "yunet.onnx", **kwargs)


def test_video_is_sampled_and_face_metadata_is_returned(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(video)
    face = np.array([[1, 2, 8, 7, 3, 4, 5, 4, 7, 4, 4, 7, 6, 8, 0.95]], dtype=np.float32)
    detector = FakeDetector(face)

    report = detect_video(
        video,
        _config(tmp_path, frame_interval_seconds=0.5),
        detector_factory=lambda _: detector,
    )

    assert report.video_width == 24
    assert report.video_height == 16
    assert report.sampled_frame_count >= 2
    assert report.detections[0].frame_index == 0
    assert report.detections[0].video_width == 24
    assert report.detections[0].confidence == pytest.approx(0.95)
    assert report.detections[0].landmarks == ((3.0, 4.0), (5.0, 4.0), (7.0, 4.0), (4.0, 7.0), (6.0, 8.0))


def test_no_face_video_returns_empty_detections(tmp_path: Path) -> None:
    video = tmp_path / "blank.mp4"
    _write_video(video)
    report = detect_video(
        video,
        _config(tmp_path),
        detector_factory=lambda _: FakeDetector(None),
    )
    assert report.detections == []


def test_invalid_video_and_interval_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FaceDetectionError, match="could not be opened"):
        detect_video(
            tmp_path / "missing.mp4",
            _config(tmp_path),
            detector_factory=lambda _: FakeDetector(None),
        )
    with pytest.raises(ValueError, match="greater than zero"):
        detect_video(
            tmp_path / "missing.mp4",
            _config(tmp_path, frame_interval_seconds=0),
            detector_factory=lambda _: FakeDetector(None),
        )


def test_missing_or_unloadable_model_is_explicit(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(FaceDetectionError, match="model file is missing"):
        load_yunet_detector(_config(tmp_path))

    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"not-a-model")
    monkeypatch.setattr(
        cv2.FaceDetectorYN,
        "create",
        lambda *args: (_ for _ in ()).throw(RuntimeError("bad model")),
    )
    with pytest.raises(FaceDetectionError, match="could not be loaded"):
        load_yunet_detector(_config(tmp_path))


def test_capture_is_released_when_detection_fails(tmp_path: Path) -> None:
    class FailingCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FPS: 4.0,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 24.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 16.0,
                cv2.CAP_PROP_POS_MSEC: 0.0,
            }
            return values[prop]

        def read(self):
            raise RuntimeError("read failure")

        def release(self):
            self.released = True

    capture = FailingCapture()
    with pytest.raises(FaceDetectionError, match="frame could not be read"):
        detect_video(
            tmp_path / "sample.mp4",
            _config(tmp_path),
            detector_factory=lambda _: FakeDetector(None),
            capture_factory=lambda _: capture,
        )
    assert capture.released
