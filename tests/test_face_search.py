from pathlib import Path

import cv2
import numpy as np
import pytest

from facecheck.face_detection import FaceDetectionConfig
from facecheck.face_embedding import SFaceConfig
from facecheck.face_search import (
    FaceSearchConfig,
    FaceSearchError,
    search_encrypted_video,
    search_video,
)
from facecheck.storage import store_encrypted_video


def _write_video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v writer unavailable")
    try:
        for index in range(count):
            writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def _config(tmp_path: Path, **changes) -> FaceSearchConfig:
    detection = FaceDetectionConfig(
        model_path=tmp_path / "yunet.onnx",
        frame_interval_seconds=0.25,
    )
    embedding = SFaceConfig(model_path=tmp_path / "sface.onnx")
    values = dict(
        detection=detection,
        embedding=embedding,
        top_k=10,
        min_similarity=None,
        max_processing_seconds=60.0,
    )
    values.update(changes)
    return FaceSearchConfig(**values)


def _raw_face(x: float = 4.0, y: float = 3.0) -> np.ndarray:
    return np.array(
        [[x, y, 12, 12, 7, 7, 13, 7, 10, 10, 7, 14, 13, 14, 0.95]],
        dtype=np.float32,
    )


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def setInputSize(self, input_size):
        pass

    def detect(self, image):
        self.calls += 1
        return None, self.faces


class FakeRecognizer:
    def __init__(self, feature):
        self.feature_value = feature
        self.calls = 0

    def alignCrop(self, image, face):
        return image[:8, :8]

    def feature(self, aligned):
        self.calls += 1
        return self.feature_value


def _query() -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = 1
    return vector


def test_single_face_search_returns_frame_metadata_and_no_vector(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, count=1)
    result = search_video(
        video,
        _query(),
        _config(tmp_path),
        detector_factory=lambda _: FakeDetector(_raw_face()),
        recognizer_factory=lambda _: FakeRecognizer(
            np.eye(1, 128, dtype=np.float32)
        ),
    )
    assert len(result) == 1
    assert result[0].candidate_id == "frame-0-face-0"
    assert result[0].frame_index == 0
    assert result[0].timestamp_seconds >= 0
    assert result[0].similarity == pytest.approx(1.0)
    assert not hasattr(result[0], "vector")


def test_multiple_frames_and_faces_are_independent(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, count=4)
    faces = np.concatenate((_raw_face(), _raw_face(x=16.0)), axis=0)
    result = search_video(
        video,
        _query(),
        _config(tmp_path),
        detector_factory=lambda _: FakeDetector(faces),
        recognizer_factory=lambda _: FakeRecognizer(
            np.eye(1, 128, dtype=np.float32)
        ),
    )
    assert len(result) == 8
    assert {item.candidate_id for item in result} == {
        f"frame-{frame}-face-{face}"
        for frame in (0, 1, 2, 3)
        for face in (0, 1)
    }
    assert len({item.frame_index for item in result}) == 4


def test_no_face_returns_empty_without_loading_sface(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video)
    result = search_video(
        video,
        _query(),
        _config(tmp_path),
        detector_factory=lambda _: FakeDetector(None),
    )
    assert result == ()


def test_search_threshold_and_top_k(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, count=1)
    feature = np.zeros((1, 128), dtype=np.float32)
    feature[0, 0] = 1
    result = search_video(
        video,
        _query(),
        _config(tmp_path, top_k=1, min_similarity=0.9),
        detector_factory=lambda _: FakeDetector(_raw_face()),
        recognizer_factory=lambda _: FakeRecognizer(feature),
    )
    assert len(result) == 1
    with pytest.raises(ValueError, match="top_k"):
        search_video(
            video,
            _query(),
            _config(tmp_path, top_k=0),
            detector_factory=lambda _: FakeDetector(None),
        )


def test_invalid_video_and_detection_or_feature_failures_are_clear(
    tmp_path: Path,
) -> None:
    with pytest.raises(FaceSearchError, match="could not be completed"):
        search_video(
            tmp_path / "missing.mp4",
            _query(),
            _config(tmp_path),
            detector_factory=lambda _: FakeDetector(None),
        )
    video = tmp_path / "video.mp4"
    _write_video(video, count=1)
    with pytest.raises(FaceSearchError, match="feature extraction failed"):
        search_video(
            video,
            _query(),
            _config(tmp_path),
            detector_factory=lambda _: FakeDetector(_raw_face()),
            recognizer_factory=lambda _: FakeRecognizer(
                np.zeros((1, 127), dtype=np.float32)
            ),
        )


def test_encrypted_search_cleans_unique_restore_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, count=1)
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=100,
        key_version="video-key-v1",
        video_id="search-video",
    )
    temp_root = tmp_path / "search-temp"
    temp_root.mkdir()
    result = search_encrypted_video(
        stored.directory,
        _query(),
        _config(tmp_path),
        b"0123456789abcdef",
        b"independent-mac-key",
        temp_root=temp_root,
        detector_factory=lambda _: FakeDetector(_raw_face()),
        recognizer_factory=lambda _: FakeRecognizer(
            np.eye(1, 128, dtype=np.float32)
        ),
    )
    assert len(result) == 1
    assert list(temp_root.iterdir()) == []


def test_encrypted_search_cleans_restore_directory_after_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, count=1)
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=100,
        key_version="video-key-v1",
        video_id="failed-search-video",
    )
    temp_root = tmp_path / "search-temp"
    temp_root.mkdir()
    with pytest.raises(FaceSearchError, match="feature extraction failed"):
        search_encrypted_video(
            stored.directory,
            _query(),
            _config(tmp_path),
            b"0123456789abcdef",
            b"independent-mac-key",
            temp_root=temp_root,
            detector_factory=lambda _: FakeDetector(_raw_face()),
            recognizer_factory=lambda _: FakeRecognizer(
                np.zeros((1, 127), dtype=np.float32)
            ),
        )
    assert list(temp_root.iterdir()) == []


def test_missing_model_is_reported_when_a_face_is_detected(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, count=1)
    with pytest.raises(FaceSearchError, match="feature extraction failed"):
        search_video(
            video,
            _query(),
            _config(tmp_path),
            detector_factory=lambda _: FakeDetector(_raw_face()),
        )
