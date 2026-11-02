"""Shared recognition defaults and dispatch; model-specific preprocessing stays intact."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from .face_embedding import (
    ArcFaceConfig, SFaceConfig, extract_arcface_embedding, extract_embedding,
    load_arcface_recognizer, load_sface_recognizer,
)

DEFAULT_RECOGNITION_MODEL = "arcface"
# Frozen from the complete plaintext 502-identity selection run; never fit on test.
DEFAULT_ARCFACE_THRESHOLD = 0.2731515063106565
MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ARCFACE_PATH = MODEL_ROOT / "arcface_w600k_r50.onnx"
DEFAULT_SFACE_PATH = MODEL_ROOT / "face_recognition_sface_2021dec.onnx"
RecognitionConfig = Union[ArcFaceConfig, SFaceConfig]


def recognition_config(model: str = DEFAULT_RECOGNITION_MODEL, *,
                       arcface_path: Path = DEFAULT_ARCFACE_PATH,
                       sface_path: Path = DEFAULT_SFACE_PATH) -> RecognitionConfig:
    if model == "arcface":
        return ArcFaceConfig(arcface_path)
    if model == "sface":
        return SFaceConfig(sface_path)
    raise ValueError("recognition model must be arcface or sface")


def model_name(config: RecognitionConfig) -> str:
    return "arcface" if isinstance(config, ArcFaceConfig) else "sface"


def load_recognizer(config: RecognitionConfig):
    return (load_arcface_recognizer(config) if isinstance(config, ArcFaceConfig)
            else load_sface_recognizer(config))


def extract_recognition_embedding(image, detection, config: RecognitionConfig, **kwargs):
    extract = extract_arcface_embedding if isinstance(config, ArcFaceConfig) else extract_embedding
    return extract(image, detection, config, **kwargs)
