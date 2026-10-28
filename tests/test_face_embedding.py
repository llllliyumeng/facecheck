from pathlib import Path

import cv2
import numpy as np
import pytest

from facecheck.face_detection import FaceDetection
from facecheck.face_embedding import (
    FaceEmbeddingError,
    SFaceConfig,
    cosine_similarity,
    extract_embedding,
    face_record_from_detection,
    load_sface_recognizer,
    search_embeddings,
)


def _detection(**changes) -> FaceDetection:
    values = dict(
        frame_index=0,
        timestamp_seconds=0.0,
        video_width=32,
        video_height=32,
        x=4.0,
        y=4.0,
        width=16.0,
        height=16.0,
        confidence=0.95,
        landmarks=((8.0, 10.0), (16.0, 10.0), (12.0, 14.0), (9.0, 18.0), (15.0, 18.0)),
    )
    values.update(changes)
    return FaceDetection(**values)


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeRecognizer:
    def __init__(self, feature):
        self.feature_value = feature
        self.aligned_input = None
        self.face_input = None

    def alignCrop(self, image, face):
        self.aligned_input = image
        self.face_input = face
        return image[:8, :8]

    def feature(self, aligned_face):
        return self.feature_value


def _config(tmp_path: Path, **changes) -> SFaceConfig:
    return SFaceConfig(model_path=tmp_path / "sface.onnx", **changes)


def test_face_record_preserves_yunet_order(tmp_path: Path) -> None:
    record = face_record_from_detection(_detection(), (32, 32, 3))
    assert record.dtype == np.float32
    assert record.shape == (15,)
    assert record.tolist() == pytest.approx(
        [4, 4, 16, 16, 8, 10, 16, 10, 12, 14, 9, 18, 15, 18, 0.95]
    )


def test_extract_embedding_aligns_and_returns_normalised_float32(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    raw_feature = np.arange(1, 129, dtype=np.float64).reshape(1, 128)
    recognizer = FakeRecognizer(raw_feature)
    result = extract_embedding(
        image,
        _detection(),
        _config(tmp_path),
        recognizer_factory=lambda _: recognizer,
    )
    assert result.vector.shape == (128,)
    assert result.vector.dtype == np.float32
    assert np.linalg.norm(result.vector) == pytest.approx(1.0)
    assert recognizer.face_input.tolist() == pytest.approx(
        [4, 4, 16, 16, 8, 10, 16, 10, 12, 14, 9, 18, 15, 18, 0.95]
    )


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (_unit([1] + [0] * 127), _unit([1] + [0] * 127), 1.0),
        (_unit([1, 0] + [0] * 126), _unit([0, 1] + [0] * 126), 0.0),
        (_unit([1] + [0] * 127), _unit([-1] + [0] * 127), -1.0),
    ],
)
def test_cosine_similarity(left, right, expected) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(128, dtype=np.float32),
        np.zeros((1, 128), dtype=np.float32),
        np.zeros(127, dtype=np.float32),
        np.zeros(129, dtype=np.float32),
        np.full(128, np.nan, dtype=np.float32),
        np.full(128, np.inf, dtype=np.float32),
    ],
)
def test_invalid_vectors_are_rejected(value) -> None:
    valid = _unit([1] + [0] * 127)
    with pytest.raises(FaceEmbeddingError):
        cosine_similarity(value, valid)


def test_search_sorts_filters_and_does_not_mutate_inputs() -> None:
    query = _unit([1] + [0] * 127)
    candidates = [
        ("low", _unit([0, 1] + [0] * 126)),
        ("high", query.copy()),
        ("middle", _unit([1, 1] + [0] * 126)),
    ]
    original = [candidate[1].copy() for candidate in candidates]
    result = search_embeddings(query, candidates, top_k=2, min_similarity=0.5)
    assert result[0].sample_id == "high"
    assert result[1].sample_id == "middle"
    assert all(
        np.array_equal(candidate[1], before)
        for candidate, before in zip(candidates, original)
    )
    assert search_embeddings(query, [], top_k=2) == ()
    with pytest.raises(ValueError):
        search_embeddings(query, candidates, top_k=0)


def test_invalid_image_box_landmarks_and_feature_are_rejected(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    config = _config(tmp_path)
    with pytest.raises(FaceEmbeddingError, match="BGR"):
        extract_embedding(
            image[:, :, 0],
            _detection(),
            config,
            recognizer_factory=lambda _: FakeRecognizer(np.ones((1, 128))),
        )
    with pytest.raises(FaceEmbeddingError, match="bounding box"):
        extract_embedding(
            image,
            _detection(x=-1),
            config,
            recognizer_factory=lambda _: FakeRecognizer(np.ones((1, 128))),
        )
    with pytest.raises(FaceEmbeddingError, match="five"):
        extract_embedding(
            image,
            _detection(landmarks=None),
            config,
            recognizer_factory=lambda _: FakeRecognizer(np.ones((1, 128))),
        )
    for invalid_feature in (
        np.ones((1, 127), dtype=np.float32),
        np.full((1, 128), np.nan, dtype=np.float32),
        np.zeros((1, 128), dtype=np.float32),
    ):
        with pytest.raises(FaceEmbeddingError):
            extract_embedding(
                image,
                _detection(),
                config,
                recognizer_factory=lambda _, value=invalid_feature: FakeRecognizer(value),
            )


def test_model_missing_or_unloadable_is_explicit(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    with pytest.raises(FaceEmbeddingError, match="model file is missing"):
        load_sface_recognizer(config)
    model = tmp_path / "sface.onnx"
    model.write_bytes(b"not-a-model")
    monkeypatch.setattr(
        cv2,
        "FaceRecognizerSF_create",
        lambda *args: (_ for _ in ()).throw(RuntimeError("bad model")),
    )
    with pytest.raises(FaceEmbeddingError, match="could not be loaded"):
        load_sface_recognizer(config)


def test_matches_do_not_contain_image_path_or_identity_label() -> None:
    query = _unit([1] + [0] * 127)
    result = search_embeddings(query, [("controlled-sample-1", query)])
    assert result[0].sample_id == "controlled-sample-1"
    assert not hasattr(result[0], "image")
    assert not hasattr(result[0], "path")
    assert not hasattr(result[0], "identity")
