import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace
import cv2

from facecheck.face_detection import FaceDetectionConfig
from facecheck.face_embedding import FaceEmbeddingError, SFaceConfig
from facecheck.ytf_qg_protocol import THRESHOLD_SELECTION, TEST, select_threshold_at_far
from facecheck import ytf_runner as runner_module
from facecheck.ytf_runner import (
    CachedMediaFeatures, FaceFeature, FeatureExtractionResult, RunnerConfig, YTFRunnerError,
    OpenCVFeatureBackend,
    _score_matrix, run_ytf_runner, _uniform_sample_frames, _aggregate_gallery_features,
)


def _cache(identity, video, split, vectors):
    return CachedMediaFeatures(identity, video, split, "video", tuple(
        FaceFeature(np.asarray(vector, dtype=np.float32), i, float(i))
        for i, vector in enumerate(vectors)), len(vectors))


def _build_runner_config(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    records, assignments = [], []
    for index in range(4):
        identity = f"id_{index:03d}"
        side = THRESHOLD_SELECTION if index < 2 else TEST
        assignments.append({"identity_id": identity, "split": side})
        for role in ("gallery", "query"):
            video_id = f"{role}_{index:03d}"
            ext = ".mp4" if role == "gallery" else ".png"
            path = media_root / (video_id + ext)
            path.write_bytes((identity + role).encode())
            records.append({"identity_id": identity, "video_id": video_id,
                            "video_path": str(path.relative_to(tmp_path)), "split": role,
                            "source_group_id": f"source_{role}_{index:03d}", "label": f"label_{index:03d}",
                            "media_type": "video" if role == "gallery" else "image"})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"records": records}))
    split_path = tmp_path / "identity_split.json"
    split_path.write_text(json.dumps({"seed": "test", "assignments": assignments}))
    yunet = tmp_path / "yunet.onnx"; yunet.write_bytes(b"model")
    sface = tmp_path / "sface.onnx"; sface.write_bytes(b"model")
    return RunnerConfig(manifest_path, split_path, tmp_path / "out",
                        FaceDetectionConfig(yunet), SFaceConfig(sface, expected_dimension=4),
                        similarity_query_batch_size=2)


def test_gallery_uniform_sampling_and_normalized_mean_aggregation():
    assert _uniform_sample_frames(list(range(21)), 5) == [0, 5, 10, 15, 20]
    assert _uniform_sample_frames([1, 2], 5) == [1, 2]
    source = (
        FaceFeature(np.array([3, 0], dtype=np.float32), 1, 1.0),
        FaceFeature(np.array([0, 2], dtype=np.float32), 2, 2.0),
    )
    aggregated, valid_count = _aggregate_gallery_features(source)
    assert valid_count == 2
    assert len(aggregated) == 1
    assert np.linalg.norm(aggregated[0].vector) == pytest.approx(1.0)
    assert aggregated[0].vector == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_gallery_aggregation_ignores_invalid_vectors_and_empty_input():
    invalid = FaceFeature(np.array([np.nan, 1], dtype=np.float32), 0, 0.0)
    assert _aggregate_gallery_features([invalid]) == ((), 0)
    assert _aggregate_gallery_features([]) == ((), 0)


def test_runner_split_scopes_require_frozen_selection_threshold(tmp_path):
    config = _build_runner_config(tmp_path)
    with pytest.raises(YTFRunnerError, match="requires a finite frozen selection threshold"):
        replace(config, evaluation_scope=TEST).validate()
    with pytest.raises(YTFRunnerError, match="only valid for test scope"):
        replace(config, frozen_selection_threshold=.4).validate()
    replace(config, evaluation_scope=TEST, frozen_selection_threshold=.4).validate()


class _FakeBackend:
    def __init__(self, mode, calls=None):
        self.mode, self.calls = mode, calls

    def extract(self, record):
        path = Path(record["video_path"])
        if self.calls is not None:
            self.calls[self.mode].append((record["video_id"], path.read_bytes()))
        index = int(record["video_id"][-3:])
        vector = np.zeros(4, dtype=np.float32); vector[index] = 1
        return FeatureExtractionResult(
            (FaceFeature(vector, 0, 0.0),), 1, 2, 1, 0,
            {"video_read_decode_extract_seconds": .01 if record["media_type"] == "video" else None,
             "frame_extraction_seconds": None,
             "query_image_decode_seconds": .001 if record["media_type"] == "image" else None,
             "face_detection_seconds": .002,
             "feature_extraction_seconds": .003},
            ({"mode": self.mode, "media_id": record["video_id"],
              "identity_id": record["identity_id"], "frame_index": 0, "face_index": 1,
              "stage": "feature_extraction", "exception_type": "FaceEmbeddingError",
              "exception_message": "mock failure"},))


def _backend_factory(calls=None):
    backend_number = 0
    def factory(_detection, _embedding):
        nonlocal backend_number
        mode = ("plaintext", "encrypted_restore")[backend_number]
        backend_number += 1
        return _FakeBackend(mode, calls)
    return factory


def _copy_encrypt(source, storage_root, *_args, **_kwargs):
    storage_root.mkdir(parents=True, exist_ok=True)
    return type("Stored", (), {"directory": source})


def _copy_restore(stored_directory, destination, *_args, **_kwargs):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stored_directory, destination)
    return destination


def test_score_matrix_is_complete_with_null_for_undetected_and_no_cross_split():
    queries = [
        {"video_id": "q1", "identity_id": "i1"},
        {"video_id": "q2", "identity_id": "i2"},
    ]
    galleries = [
        {"video_id": "g1", "identity_id": "i1"},
        {"video_id": "g2", "identity_id": "i2"},
    ]
    qcache = {"q1": _cache("i1", "q1", TEST, [[1, 0]]),
              "q2": _cache("i2", "q2", TEST, [])}
    gcache = {"g1": _cache("i1", "g1", TEST, [[1, 0], [0.9, 0.1]]),
              "g2": _cache("i2", "g2", TEST, [])}
    scores, provenance, _, per_query = _score_matrix(queries, galleries, qcache, gcache, 1)
    assert set(scores) == {("q1", "g1"), ("q1", "g2"), ("q2", "g1"), ("q2", "g2")}
    assert scores[("q1", "g1")] == pytest.approx(1.0)
    assert provenance[("q1", "g1")]["candidate_frame_index"] in (0, 1)
    assert scores[("q1", "g2")] is None
    assert scores[("q2", "g1")] is None
    assert scores[("q2", "g2")] is None
    assert set(per_query) == {"q1", "q2"}


@pytest.mark.parametrize("bad_vector,reason", [
    (np.zeros(2, dtype=np.float32), "zero_norm"),
    (np.array([np.nan, 1], dtype=np.float32), "non_finite"),
    (np.array([np.inf, 1], dtype=np.float32), "non_finite"),
])
@pytest.mark.parametrize("role", ["query", "gallery"])
def test_invalid_scoring_vectors_are_rejected_without_zero_substitution(bad_vector, reason, role):
    query_vector = bad_vector if role == "query" else np.array([1, 0], dtype=np.float32)
    gallery_vector = bad_vector if role == "gallery" else np.array([1, 0], dtype=np.float32)
    query = [{"video_id": "q", "identity_id": "i"}]
    gallery = [{"video_id": "g", "identity_id": "i"}]
    diagnostics = []
    scores, _provenance, _, _ = _score_matrix(
        query, gallery, {"q": _cache("i", "q", TEST, [query_vector])},
        {"g": _cache("i", "g", TEST, [gallery_vector])}, 1, diagnostics=diagnostics)
    assert scores[("q", "g")] is None
    side = diagnostics[0]["query_input"] if role == "query" else diagnostics[0]["gallery_input"]
    rejected = side["rejected_vectors"]
    assert rejected[0]["reason"] == reason
    assert rejected[0]["vector_index"] == 0
    assert rejected[0]["role"] == role
    assert side["valid_vector_count"] == 0


def test_finite_vectors_score_and_emit_normalization_and_matmul_diagnostics():
    diagnostics = []
    scores, *_ = _score_matrix(
        [{"video_id": "q", "identity_id": "i"}],
        [{"video_id": "g", "identity_id": "i"}],
        {"q": _cache("i", "q", TEST, [[3, 4]])},
        {"g": _cache("i", "g", TEST, [[3, 4]])}, 1, diagnostics=diagnostics)
    assert scores[("q", "g")] == pytest.approx(1)
    assert diagnostics[0]["query_input"]["normalization"]["finite"]
    assert diagnostics[0]["gallery_input"]["normalization"]["finite"]
    assert diagnostics[0]["matrix_multiplication"]["finite"]


def test_top_k_mean_score_aggregation_is_explicit_and_bounded():
    scores, *_ = _score_matrix(
        [{"video_id": "q", "identity_id": "i"}],
        [{"video_id": "g", "identity_id": "i"}],
        {"q": _cache("i", "q", TEST, [[1, 0]])},
        {"g": _cache("i", "g", TEST, [[1, 0], [0.8, 0.6]])}, 1,
        score_aggregation="top_k_mean", score_aggregation_top_n=2)
    assert scores[("q", "g")] == pytest.approx(0.9)


def test_normalization_exception_identifies_stage(monkeypatch):
    monkeypatch.setattr(runner_module, "_normalize_matrix",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ArithmeticError("normalize failed")))
    with pytest.raises(YTFRunnerError, match="normalization stage failed for Gallery"):
        _score_matrix(
            [{"video_id": "q"}], [{"video_id": "g"}],
            {"q": _cache("i", "q", TEST, [[1, 0]])},
            {"g": _cache("i", "g", TEST, [[1, 0]])}, 1)


@pytest.mark.parametrize("failure", [
    lambda *_args: np.array([[np.nan]], dtype=np.float32),
    lambda *_args: (_ for _ in ()).throw(ArithmeticError("matmul failed")),
])
def test_matrix_multiplication_invalid_results_are_blocked(failure, monkeypatch):
    diagnostics = []
    monkeypatch.setattr(runner_module, "_multiply_similarity", failure)
    with pytest.raises(YTFRunnerError, match="matrix multiplication stage"):
        _score_matrix(
            [{"video_id": "q"}], [{"video_id": "g"}],
            {"q": _cache("i", "q", TEST, [[1, 0]])},
            {"g": _cache("i", "g", TEST, [[1, 0]])}, 1, diagnostics=diagnostics)
    assert diagnostics[0]["matrix_multiplication"]["finite"] is False


def test_threshold_selection_uses_only_selection_rows_and_preserves_ties():
    selection = [
        {"split": THRESHOLD_SELECTION, "pair_type": "genuine", "similarity": .9},
        *[{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": .2}
          for _ in range(1000)],
    ]
    result = select_threshold_at_far(selection)
    assert result["status"] == "estimated"
    assert result["split"] == THRESHOLD_SELECTION
    assert result["threshold"] == .9
    assert result["far_resolution"] == pytest.approx(.001)
    with pytest.raises(ValueError):
        select_threshold_at_far(selection + [{"split": TEST, "pair_type": "genuine", "similarity": 1.0}])


def test_backend_counts_yunet_faces_separately_from_sface_features_and_failures(tmp_path, monkeypatch):
    class Detector:
        def setInputSize(self, _size):
            pass

        def detect(self, _image):
            valid_a = np.zeros(15, dtype=np.float32); valid_a[2:4] = 8; valid_a[14] = .99
            valid_b = valid_a.copy()
            invalid = np.zeros(14, dtype=np.float32)
            return 1, [valid_a, valid_b, invalid]

    monkeypatch.setattr(runner_module, "load_yunet_detector", lambda _config: Detector())
    monkeypatch.setattr(runner_module, "load_sface_recognizer", lambda _config: object())
    monkeypatch.setattr(runner_module.cv2, "imread", lambda _path: np.zeros((20, 20, 3), dtype=np.uint8))
    calls = 0

    def extract(_image, _detection, _config, *, recognizer_factory):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FaceEmbeddingError("mock feature failure")
        return SimpleNamespace(vector=np.ones(128, dtype=np.float32))

    monkeypatch.setattr(runner_module, "extract_embedding", extract)
    backend = OpenCVFeatureBackend(FaceDetectionConfig(tmp_path / "yunet"), SFaceConfig(tmp_path / "sface"))
    result = backend.extract({"video_id": "q", "identity_id": "person-1", "_experiment_mode": "plaintext",
                              "video_path": str(tmp_path / "query.png"), "media_type": "image"})
    assert result.detected_face_count == 3
    assert len(result.features) == 1
    assert result.feature_extraction_failure_count == 1
    assert result.invalid_detection_record_count == 1
    failure = next(row for row in result.failures if row["stage"] == "feature_extraction")
    assert failure == {"mode": "plaintext", "media_id": "q", "identity_id": "person-1",
                       "frame_index": 0, "face_index": 1, "stage": "feature_extraction",
                       "exception_type": "FaceEmbeddingError", "exception_message": "mock feature failure"}
    assert len(result.features) == 1


def test_annotation_box_selects_highest_iou_detection_and_normalizes_windows_path(tmp_path):
    annotation = tmp_path / "person.labeled_faces.txt"
    annotation.write_text("Person\\0\\2.10.jpg,0,10,10,20,20,0,1\n", encoding="utf-8")
    record = {
        "ytf_annotation_file": str(annotation),
        "ytf_original_video_name": r"Person\0",
        "ytf_selected_query_frame_name": "2.10.jpg",
        "_use_annotation_box": True,
    }
    low_iou = np.zeros(15, dtype=np.float32)
    low_iou[:4] = [0, 0, 10, 10]
    high_iou = np.zeros(15, dtype=np.float32)
    high_iou[:4] = [10, 10, 20, 20]

    selected = OpenCVFeatureBackend._select_query_entries(
        [(3, low_iou), (7, high_iou)], record, (100, 100, 3))

    assert len(selected) == 1
    assert selected[0][0] == 7


def test_annotation_box_missing_or_nonoverlapping_returns_no_query_feature(tmp_path):
    annotation = tmp_path / "person.labeled_faces.txt"
    annotation.write_text("Person/0/2.10.jpg,0,10,10,20,20,0,1\n", encoding="utf-8")
    record = {
        "ytf_annotation_file": str(annotation),
        "ytf_original_video_name": "Person/0",
        "ytf_selected_query_frame_name": "2.10.jpg",
        "_use_annotation_box": True,
    }
    outside = np.zeros(15, dtype=np.float32)
    outside[:4] = [50, 50, 10, 10]
    assert OpenCVFeatureBackend._select_query_entries(
        [(0, outside)], record, (100, 100, 3)) == []
    record["ytf_annotation_file"] = str(tmp_path / "missing.txt")
    assert OpenCVFeatureBackend._select_query_entries(
        [(0, outside)], record, (100, 100, 3)) == []


def test_runner_extracts_each_media_once_per_mode_and_exports_raw_results(tmp_path, monkeypatch):
    cfg = _build_runner_config(tmp_path)
    output = cfg.output_dir
    calls = {"plaintext": [], "encrypted_restore": []}
    temp_paths = _track_tempdirs(monkeypatch)
    result = run_ytf_runner(cfg, encryption_key=b"e" * 16, mac_key=b"m" * 32,
                            backend_factory=_backend_factory(calls),
                            encrypt=_copy_encrypt, restore=_copy_restore)
    assert temp_paths and all(not path.exists() for path in temp_paths)
    assert len(calls["plaintext"]) == 8
    assert len(calls["encrypted_restore"]) == 8
    for mode_calls in calls.values():
        ids = [row[0] for row in mode_calls]
        assert len(ids) == len(set(ids))
    # Every same-named input reaches both modes with identical bytes after restore.
    assert dict(calls["plaintext"]) == dict(calls["encrypted_restore"])
    assert result["matrix_audit"][THRESHOLD_SELECTION]["expected_pairs"] == 4
    assert result["matrix_audit"][TEST]["expected_pairs"] == 4
    assert result["cross_partition_comparisons"] == 0
    assert result["mode_consistency"]["consistent"] is True
    assert (output / "config.json").is_file()
    assert (output / "feature_summary.json").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "feature_failures.csv").is_file()
    assert (output / "scoring_diagnostics.json").is_file()
    feature_summary = json.loads((output / "feature_summary.json").read_text())
    assert all(row["detected_face_count"] == 2 and row["feature_count"] == 1
               and row["feature_extraction_failure_count"] == 1
               for rows in feature_summary.values() for row in rows)
    with (output / "genuine_impostor_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row["mode"] for row in rows} == {"plaintext", "encrypted_restore"}
    assert all(row["similarity"] not in ("", "None") for row in rows)
    with (output / "feature_failures.csv").open(encoding="utf-8-sig", newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert len(failures) == 16
    assert {row["mode"] for row in failures} == {"plaintext", "encrypted_restore"}
    assert all(row["identity_id"] and row["frame_index"] == "0" and row["face_index"] == "1"
               and row["exception_type"] == "FaceEmbeddingError" for row in failures)
    diagnostics = json.loads((output / "scoring_diagnostics.json").read_text())
    assert set(diagnostics) == {"plaintext", "encrypted_restore"}
    report = (output / "report.md").read_text()
    assert "not original YouTube" in report
    assert "every YuNet-detected face" in report
    assert "non-target Query face" in report


def test_runner_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    config = RunnerConfig(tmp_path / "missing.json", tmp_path / "missing-split.json", output,
                          FaceDetectionConfig(tmp_path / "model"), SFaceConfig(tmp_path / "sface"))
    with pytest.raises(YTFRunnerError, match="refusing to overwrite"):
        run_ytf_runner(config, encryption_key=b"key", mac_key=b"mac")
    assert (output / "keep.txt").read_text() == "preserve"


def _track_tempdirs(monkeypatch):
    original = runner_module.tempfile.TemporaryDirectory
    paths = []
    def tracked(*args, **kwargs):
        temporary = original(*args, **kwargs)
        paths.append(Path(temporary.name))
        return temporary
    monkeypatch.setattr(runner_module.tempfile, "TemporaryDirectory", tracked)
    return paths


def test_model_initialization_failure_cleans_private_work_and_staging(tmp_path, monkeypatch):
    config = _build_runner_config(tmp_path)
    temp_paths = _track_tempdirs(monkeypatch)

    def fail_model_init(_detection, _embedding):
        raise RuntimeError("mock model initialization failure")

    with pytest.raises(RuntimeError, match="model initialization"):
        run_ytf_runner(config, encryption_key=b"e" * 16, mac_key=b"m" * 32,
                       backend_factory=fail_model_init)
    assert temp_paths and all(not path.exists() for path in temp_paths)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize("failure_stage", ["encryption", "restore"])
def test_crypto_failure_cleans_private_work_and_does_not_publish(tmp_path, monkeypatch, failure_stage):
    config = _build_runner_config(tmp_path)
    temp_paths = _track_tempdirs(monkeypatch)

    def fail_encrypt(*_args, **_kwargs):
        raise RuntimeError("mock encryption failure")

    def fail_restore(*_args, **_kwargs):
        raise RuntimeError("mock restoration failure")

    with pytest.raises(RuntimeError):
        run_ytf_runner(
            config, encryption_key=b"e" * 16, mac_key=b"m" * 32,
            backend_factory=_backend_factory(),
            encrypt=fail_encrypt if failure_stage == "encryption" else _copy_encrypt,
            restore=fail_restore if failure_stage == "restore" else _copy_restore,
        )
    assert temp_paths and all(not path.exists() for path in temp_paths)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_encrypted_feature_extraction_failure_cleans_restored_plaintext(tmp_path, monkeypatch):
    config = _build_runner_config(tmp_path)
    temp_paths = _track_tempdirs(monkeypatch)
    backend_number = 0

    class FailingEncryptedBackend(_FakeBackend):
        def extract(self, record):
            if self.mode == "encrypted_restore":
                raise RuntimeError("mock feature extraction failure")
            return super().extract(record)

    def factory(_detection, _embedding):
        nonlocal backend_number
        mode = ("plaintext", "encrypted_restore")[backend_number]
        backend_number += 1
        return FailingEncryptedBackend(mode)

    with pytest.raises(RuntimeError, match="feature extraction"):
        run_ytf_runner(config, encryption_key=b"e" * 16, mac_key=b"m" * 32,
                       backend_factory=factory, encrypt=_copy_encrypt, restore=_copy_restore)
    assert temp_paths and all(not path.exists() for path in temp_paths)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_result_generation_failure_removes_staged_partial_bundle(tmp_path, monkeypatch):
    config = _build_runner_config(tmp_path)
    monkeypatch.setattr(runner_module, "_write_csv", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mock CSV write failure")))
    with pytest.raises(RuntimeError, match="CSV write"):
        run_ytf_runner(config, encryption_key=b"e" * 16, mac_key=b"m" * 32,
                       backend_factory=_backend_factory(), encrypt=_copy_encrypt, restore=_copy_restore)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))
