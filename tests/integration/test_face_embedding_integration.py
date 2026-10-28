import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from facecheck.face_detection import (
    FaceDetection,
    FaceDetectionConfig,
    load_yunet_detector,
)
from facecheck.face_embedding import SFaceConfig, extract_embedding


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} must be set for the explicit SFace integration test")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{name} does not point to a readable file: {path}")
    return path


def _detect_first_face(
    image: np.ndarray,
    model_path: Path,
) -> Optional[FaceDetection]:
    height, width = image.shape[:2]
    detector = load_yunet_detector(
        FaceDetectionConfig(model_path=model_path, score_threshold=0.9)
    )
    detector.setInputSize((width, height))
    _, raw_faces = detector.detect(image)
    if raw_faces is None:
        return None
    face = np.asarray(raw_faces[0], dtype=np.float32)
    if face.shape != (15,):
        pytest.fail(f"YuNet returned unexpected face shape: {face.shape}")
    return FaceDetection(
        frame_index=0,
        timestamp_seconds=0.0,
        video_width=width,
        video_height=height,
        x=float(face[0]),
        y=float(face[1]),
        width=float(face[2]),
        height=float(face[3]),
        confidence=float(face[14]),
        landmarks=tuple(
            (float(face[index]), float(face[index + 1]))
            for index in range(4, 14, 2)
        ),
    )


@pytest.mark.sface_integration
def test_authorized_image_runs_yunet_and_sface_end_to_end() -> None:
    if os.environ.get("FACECHECK_RUN_FACE_INTEGRATION") != "1":
        pytest.skip(
            "set FACECHECK_RUN_FACE_INTEGRATION=1 to run the explicit SFace test"
        )
    model_path = _required_path("FACECHECK_SFACE_MODEL")
    yunet_path = _required_path("FACECHECK_YUNET_MODEL")
    image_path = _required_path("FACECHECK_FACE_IMAGE")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        pytest.fail(f"authorized test image could not be decoded: {image_path}")

    detection = _detect_first_face(image, yunet_path)
    if detection is None:
        pytest.fail("YuNet found no face in the authorized test image")

    embedding = extract_embedding(
        image,
        detection,
        SFaceConfig(model_path=model_path),
    )
    assert embedding.vector.shape == (128,)
    assert embedding.vector.dtype == np.float32
    assert np.all(np.isfinite(embedding.vector))
    assert np.linalg.norm(embedding.vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.sface_integration
def test_authorized_blank_image_has_no_face() -> None:
    if os.environ.get("FACECHECK_RUN_FACE_INTEGRATION") != "1":
        pytest.skip(
            "set FACECHECK_RUN_FACE_INTEGRATION=1 to run the explicit SFace test"
        )
    yunet_path = _required_path("FACECHECK_YUNET_MODEL")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _detect_first_face(blank, yunet_path) is None
