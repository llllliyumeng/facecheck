"""Focused checks for the ArcFace default and explicit SFace compatibility."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from facecheck.face_detection import FaceDetectionConfig
from facecheck.face_embedding import ArcFaceConfig, SFaceConfig
from facecheck.face_search import FaceSearchConfig, _search_frames
from facecheck.recognition import DEFAULT_ARCFACE_THRESHOLD, recognition_config
from facecheck.service import VideoTaskService
from facecheck.video import ExtractedFrame
from facecheck.ytf_runner import RunnerConfig


def test_default_and_explicit_compatibility_config(tmp_path):
    assert isinstance(recognition_config(), ArcFaceConfig)
    assert recognition_config().expected_dimension == 512
    assert isinstance(recognition_config("sface"), SFaceConfig)
    with pytest.raises(ValueError):
        recognition_config("unknown")
    base = (tmp_path/'manifest', tmp_path/'split', tmp_path/'out', FaceDetectionConfig(tmp_path/'yunet'))
    arc = RunnerConfig(*base, recognition_config())
    assert arc.recognition_model == "arcface"
    assert arc.arcface_model_path == recognition_config().model_path
    assert RunnerConfig(*base, recognition_config("sface")).recognition_model == "sface"
    with pytest.raises(ValueError):
        RunnerConfig(*base, recognition_config(), recognition_model="sface")


def raw_face():
    return np.array([[10, 10, 50, 50, 25, 25, 45, 25, 35, 35, 25, 45, 45, 45, .99]], dtype=np.float32)


class Detector:
    def setInputSize(self, size):
        pass

    def detect(self, image):
        return None, raw_face()


def test_arcface_video_dispatch_retains_512_dimensions(tmp_path):
    blobs = []
    class Network:
        def setInput(self, blob):
            blobs.append(blob)
        def forward(self):
            return np.eye(1, 512, dtype=np.float32)
    cfg = FaceSearchConfig(FaceDetectionConfig(tmp_path/'yunet'), ArcFaceConfig(tmp_path/'arcface'))
    stats = {}
    result = _search_frames(
        [ExtractedFrame(0, 0., np.zeros((100, 100, 3), dtype=np.uint8))],
        np.eye(1, 512, dtype=np.float32)[0], cfg,
        detector_factory=lambda _: Detector(), recognizer_factory=lambda _: Network(),
        clock=lambda: 0., stats=stats,
    )
    assert len(result) == 1 and result[0].similarity == pytest.approx(1.)
    assert blobs[0].shape == (1, 3, 112, 112)
    assert "feature_extraction_seconds" in stats
    assert "sface_embedding_seconds" not in stats


@pytest.mark.parametrize("model,dimension,request_threshold,expected_threshold", [
    (None, 512, None, DEFAULT_ARCFACE_THRESHOLD),
    ("arcface", 512, .5, .5),
    ("sface", 128, None, None),
])
def test_service_uses_identical_query_gallery_config(monkeypatch, tmp_path, model, dimension,
                                                       request_threshold, expected_threshold):
    monkeypatch.delenv("FACECHECK_RECOGNITION_MODEL", raising=False)
    if model:
        monkeypatch.setenv("FACECHECK_RECOGNITION_MODEL", model)
    monkeypatch.setenv("FACECHECK_YUNET_MODEL", str(tmp_path/'yunet'))
    repository = SimpleNamespace(get=lambda _: {"status":"completed", "storage_id":"stored"},
                                 record_event=lambda *args: None)
    storage = tmp_path/'storage'
    (storage/'stored').mkdir(parents=True)
    service = VideoTaskService(repository, storage, encryption_key=b'e'*16, mac_key=b'm'*32)
    monkeypatch.setattr("facecheck.service.load_yunet_detector", lambda _: Detector())
    captured = []
    def extract(image, detection, cfg):
        captured.append(cfg)
        return SimpleNamespace(vector=np.eye(1, dimension, dtype=np.float32)[0])
    def search(directory, query, cfg, *args, **kwargs):
        assert cfg.embedding is captured[0]
        assert cfg.embedding.expected_dimension == query.size == dimension
        assert cfg.min_similarity == expected_threshold
        return ()
    monkeypatch.setattr("facecheck.service.extract_recognition_embedding", extract)
    monkeypatch.setattr("facecheck.service.search_encrypted_video", search)
    ok, image = cv2.imencode('.png', np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    assert service.search_task("task", image.tobytes(), top_k=5, min_similarity=request_threshold) == []
    assert service.last_search_metrics['recognition_model'] == (model or 'arcface')
    assert service.last_search_metrics['min_similarity'] == expected_threshold
