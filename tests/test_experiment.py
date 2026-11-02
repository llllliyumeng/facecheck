import csv
import json
from pathlib import Path

import numpy as np
import pytest

from facecheck.experiment import (
    ExperimentError, ExperimentManifest, MediaRecord, aggregate_identity_candidates,
    compute_metrics, load_manifest, run_comparison, validate_manifest,
    select_threshold_at_far,
)
from facecheck.face_detection import FaceDetectionConfig
from facecheck.face_embedding import SFaceConfig


def _record(tmp_path, vid, identity, split, group, label=None):
    p = tmp_path / f"{vid}.mp4"
    p.write_bytes(vid.encode())
    return MediaRecord(identity, vid, p, split, group, label or identity)


def _manifest(tmp_path):
    rows = [_record(tmp_path, "g1", "id_1", "gallery", "source-g1"),
            _record(tmp_path, "g2", "id_2", "gallery", "source-g2"),
            _record(tmp_path, "q1", "id_1", "query", "source-q1")]
    return ExperimentManifest(tuple(rows), tmp_path / "manifest.json")


def test_manifest_load_and_validation(tmp_path):
    (tmp_path / "g.mp4").write_bytes(b"x")
    (tmp_path / "q.png").write_bytes(b"x")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": [
        {"identity_id": "id_1", "video_id": "g", "video_path": "g.mp4", "split": "gallery", "source_group_id": "sg", "label": "person1"},
        {"identity_id": "id_1", "video_id": "q", "video_path": "q.png", "split": "query", "source_group_id": "sq", "label": "person1", "media_type": "image"},
    ]}))
    assert len(load_manifest(path).records) == 2
    path.write_text("{")
    with pytest.raises(ExperimentError, match="invalid experiment manifest"):
        load_manifest(path)


def test_manifest_rejects_leakage_labels_and_illegal_split(tmp_path):
    rows = [_record(tmp_path, "g", "id_1", "gallery", "same"),
            _record(tmp_path, "q", "id_1", "query", "same")]
    with pytest.raises(ExperimentError, match="leakage"):
        validate_manifest(rows)
    rows[1] = MediaRecord("id_1", "q", rows[1].video_path, "other", "different", "id_1")
    with pytest.raises(ExperimentError, match="both gallery and query"):
        validate_manifest(rows)
    rows = [_record(tmp_path, "g", "id_1", "gallery", "g"),
            _record(tmp_path, "q", "id_1", "query", "q", label="wrong")]
    rows.append(MediaRecord("id_1", "g2", rows[0].video_path, "gallery", "g2", "different"))
    with pytest.raises(ExperimentError, match="inconsistent labels"):
        validate_manifest(rows)


def test_identity_aggregation_and_match_error_miss_metrics():
    rows = [
        {"query_id": "q", "candidate_identity_id": "a", "candidate_video_id": "v", "similarity": .9},
        {"query_id": "q", "candidate_identity_id": "a", "candidate_video_id": "v", "similarity": .8},
        {"query_id": "q", "candidate_identity_id": "b", "candidate_video_id": "v2", "similarity": .7},
    ]
    agg = aggregate_identity_candidates(rows)
    assert len(agg) == 2 and agg[0]["similarity"] == .9 and agg[1]["identity_rank"] == 2
    queries = [{"query_id": "q1", "query_detected": True}, {"query_id": "q2", "query_detected": False}]
    result = [dict(agg[0], query_id="q1", is_correct_match=True, is_false_match=False),
              dict(agg[1], query_id="q1", is_correct_match=False, is_false_match=True)]
    m = compute_metrics(queries, result, [])
    assert m["top1_hit_rate"]["value"] == .5
    assert m["recall_at_1"]["value"] == .5
    assert m["recall_at_5"]["value"] == .5
    assert m["precision_at_1"]["value"] == .5
    assert m["precision_at_5"]["value"] == .1
    assert m["miss_rate"]["value"] == .5
    assert m["false_match_rate"]["value"] == 0
    assert m["tar_at_far_1e-3"]["status"] == "not_estimable"


def test_tar_far_sample_minimum_and_fixed_threshold():
    pairs = ([{"pair_type": "genuine", "similarity": .9}] +
             [{"pair_type": "impostor", "similarity": .2} for _ in range(1000)])
    q = [{"query_id": "q", "query_detected": True}]
    m = compute_metrics(q, [], pairs, threshold=.8,
                        threshold_selection={"split": "threshold_selection", "threshold": .8})
    assert m["tar_at_far_1e-3"]["value"] == 1
    assert m["tar_at_far_1e-3"]["far"] == 0
    m = compute_metrics(q, [], pairs[:10], threshold=.8,
                        threshold_selection={"split": "threshold_selection"})
    assert m["tar_at_far_1e-3"]["status"] == "not_estimable"
    selected = select_threshold_at_far(pairs)
    assert selected["status"] == "estimated" and selected["split"] == "threshold_selection"
    assert select_threshold_at_far(pairs[:10])["status"] == "not_estimable"


def test_comparison_exports_raw_files_and_shares_parameters(tmp_path, monkeypatch):
    import facecheck.experiment as exp
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(exp, "_query_vectors", lambda *a: ([(np.ones(128, dtype=np.float32), 0, 0.)], .01, .02, .03))
    class Match:
        candidate_id = "frame-0-face-0"; frame_index = 0; timestamp_seconds = 0.; similarity = .9
    def fake_search(path, vector, config, stats=None):
        if stats is not None:
            stats.update(feature_count=1, video_decode_extract_seconds=.1,
                         face_detection_seconds=.2, sface_embedding_seconds=.3,
                         similarity_sort_seconds=.01)
        return [Match()]
    monkeypatch.setattr(exp, "search_video", fake_search)
    stored_sources = {}
    def fake_store(source, root, *args, **kwargs):
        directory = Path(root) / kwargs["video_id"]
        directory.mkdir(parents=True)
        stored_sources[directory] = Path(source)
        return type("Stored", (), {"directory": directory})()
    def fake_restore(stored, dest, *args, **kwargs):
        return dest.write_bytes(stored_sources[Path(stored)].read_bytes())
    monkeypatch.setattr(exp, "store_encrypted_video", fake_store)
    monkeypatch.setattr(exp, "restore_encrypted_video", fake_restore)
    d = FaceDetectionConfig(tmp_path / "yunet.onnx")
    e = SFaceConfig(tmp_path / "sface.onnx")
    result = run_comparison(manifest, tmp_path / "out", detection=d, embedding=e,
                            encryption_key=b"0123456789abcdef", mac_key=b"independent-mac-key")
    assert result["shared_parameters"]["frame_interval_seconds"] == d.frame_interval_seconds
    for mode in ("plaintext", "encrypted_restore"):
        out = tmp_path / "out" / mode
        assert {"config.json", "metrics.json", "query_results.csv", "candidate_results.csv", "identity_results.csv", "timing.csv", "report.txt"}.issubset({p.name for p in out.iterdir()})
        with (out / "candidate_results.csv").open(encoding="utf-8-sig") as f:
            assert next(csv.DictReader(f))["similarity"] == "0.9"
    with pytest.raises(ExperimentError, match="refusing to overwrite"):
        run_comparison(manifest, tmp_path / "out", detection=d, embedding=e,
                       encryption_key=b"0123456789abcdef", mac_key=b"independent-mac-key")
