"""Run the complete plaintext ArcFace YTF threshold-selection partition once."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from facecheck.face_detection import FaceDetectionConfig  # noqa: E402
from facecheck.face_embedding import ArcFaceConfig, SFaceConfig  # noqa: E402
from facecheck.ytf_qg_protocol import (  # noqa: E402
    THRESHOLD_SELECTION,
    build_similarity_pairs,
    retrieval_metrics,
    select_threshold_at_far,
    validate_layout,
)
from facecheck.ytf_runner import (  # noqa: E402
    CachedMediaFeatures,
    OpenCVFeatureBackend,
    _load_inputs,
    _score_matrix,
)


MANIFEST = ROOT / "data/ytf/experiment_full.json"
SPLIT_PATH = ROOT / "data/ytf/query_gallery_identity_split.json"
YUNET = ROOT / "models/face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models/face_recognition_sface_2021dec.onnx"
ARCFACE = ROOT / "models/arcface_w600k_r50.onnx"
BASELINE = ROOT / "data/ytf/experiments/ytf_full_run_02/metrics.json"
OUTPUT = ROOT / "data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext"
TARGET_FAR = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_status(path: Path, state: str, *, started: str, detail: str | None = None) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"state": state, "started_at_utc": started,
                                     "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "detail": detail}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT}")
    records, all_splits, manifest_json, split_json = _load_inputs(MANIFEST, SPLIT_PATH)
    selection_ids = sorted(identity for identity, side in all_splits.items()
                           if side == THRESHOLD_SELECTION)
    if len(selection_ids) != 502:
        raise SystemExit(f"expected 502 selection identities, got {len(selection_ids)}")
    selected = set(selection_ids)
    selection_records = [record for record in records if record["identity_id"] in selected]
    identity_split = {identity: THRESHOLD_SELECTION for identity in selection_ids}
    layout = validate_layout(selection_records, identity_split)
    if layout["gallery_count"] != 502 or layout["query_count"] != 502:
        raise SystemExit("selection layout must contain exactly 502 gallery and query records")
    if not all(path.is_file() and not path.is_symlink() for path in (YUNET, SFACE, ARCFACE)):
        raise SystemExit("one or more configured model files are missing or unsafe")

    galleries = [record for record in selection_records if record["split"] == "gallery"]
    queries = [record for record in selection_records if record["split"] == "query"]
    frame_count = sum(int(record.get("ytf_frame_count", 0)) for record in galleries)
    model_hashes = {"yunet": sha256(YUNET), "sface_reference": sha256(SFACE),
                    "arcface": sha256(ARCFACE)}
    started_at = datetime.now(timezone.utc).isoformat()
    OUTPUT.mkdir(parents=True, mode=0o700)
    write_json(OUTPUT / "identity_ids.json", {
        "source_split": THRESHOLD_SELECTION, "identity_split_seed": split_json["seed"],
        "identity_split_sha256": sha256(SPLIT_PATH), "identity_count": len(selection_ids),
        "identity_ids": selection_ids,
    })
    write_json(OUTPUT / "config.json", {
        "purpose": "complete plaintext ArcFace selection evaluation; no test identities",
        "started_at_utc": started_at, "evaluation_scope": THRESHOLD_SELECTION,
        "manifest_path": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "identity_split_path": str(SPLIT_PATH), "identity_split_seed": split_json["seed"],
        "identity_count": 502, "gallery_count": 502, "query_count": 502,
        "source_partition": "existing identity-disjoint threshold_selection only",
        "model": {"model_path": str(ARCFACE), "expected_dimension": 512,
                  "input_size": [112, 112], "backend_id": cv2.dnn.DNN_BACKEND_OPENCV,
                  "target_id": cv2.dnn.DNN_TARGET_CPU, "name": "InsightFace buffalo_l w600k_r50",
                  "file_sha256": model_hashes["arcface"],
                  "weight_source": "official InsightFace v0.7 buffalo_l model pack",
                  "license_note": "pretrained model restricted by InsightFace to non-commercial research use",
                  "input_preprocessing": "112x112, YuNet five-landmark ArcFace similarity alignment, BGR to RGB, (pixel-127.5)/127.5",
                  "embedding": "512 float32, L2 normalized; cosine similarity"},
        "reference_sface": {"path": str(SFACE), "sha256": model_hashes["sface_reference"]},
        "yunet": {"path": str(YUNET), "sha256": model_hashes["yunet"],
                  "score_threshold": 0.9, "frame_interval_seconds": 1.0},
        "gallery": {"sampled_frame_count_from_manifest": frame_count,
                    "quality_filter": False, "feature_aggregation": "none",
                    "faces": "all valid YuNet detections",
                    "pair_similarity": "maximum cosine similarity over all Query/Gallery frame-feature pairs"},
        "query": {"current query flow": "unchanged YuNet detections; no annotation box filtering"},
        "target_far": TARGET_FAR,
        "threshold_source": "ArcFace plaintext scores from this threshold_selection split only; SFace threshold not reused",
        "feature_cache": "per-media embeddings held once in memory for pair scoring; no cross-model or encrypted mode reuse",
        "modes": ["plaintext"], "formal_test_run": False,
        "runtime_environment": {"platform": platform.platform(), "machine": platform.machine(),
                                "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
        "model_sha256": model_hashes,
        "source_dataset": manifest_json.get("dataset"),
    })
    status_path = OUTPUT / "run_status.json"
    write_status(status_path, "running", started=started_at)
    total_started = time.perf_counter()

    try:
        backend = OpenCVFeatureBackend(
            FaceDetectionConfig(YUNET, score_threshold=0.9, frame_interval_seconds=1.0),
            SFaceConfig(SFACE), recognition_model="arcface", arcface_model_path=ARCFACE,
        )
        gallery_cache: dict[str, CachedMediaFeatures] = {}
        query_cache: dict[str, CachedMediaFeatures] = {}
        feature_rows: list[dict] = []
        failure_rows: list[dict] = []
        feature_started = time.perf_counter()

        def extract_batch(batch: list[dict], role: str, cache: dict[str, CachedMediaFeatures]) -> None:
            for index, record in enumerate(batch, start=1):
                extraction = backend.extract({
                    **record, "_experiment_mode": "plaintext_selection",
                    "_gallery_representation_enabled": role == "gallery",
                    "_gallery_quality_filter": False,
                    "_gallery_feature_aggregation": "none",
                })
                media_id = str(record["video_id"])
                cache[media_id] = CachedMediaFeatures(
                    str(record["identity_id"]), media_id, THRESHOLD_SELECTION,
                    "video" if role == "gallery" else "image", extraction.features,
                    extraction.sampled_frame_count,
                )
                feature_rows.append({
                    "media_id": media_id, "identity_id": record["identity_id"], "role": role,
                    "sampled_frame_count": extraction.sampled_frame_count,
                    "detected_face_count": extraction.detected_face_count,
                    "feature_count": len(extraction.features),
                    "feature_extraction_failure_count": extraction.feature_extraction_failure_count,
                    "invalid_detection_record_count": extraction.invalid_detection_record_count,
                    "detection_failure_count": extraction.detection_failure_count,
                    "video_read_decode_extract_seconds": extraction.timings.get("video_read_decode_extract_seconds"),
                    "face_detection_seconds": extraction.timings.get("face_detection_seconds"),
                    "feature_extraction_seconds": extraction.timings.get("feature_extraction_seconds"),
                })
                failure_rows.extend(extraction.failures)
                if index % (25 if role == "query" else 10) == 0 or index == len(batch):
                    elapsed = time.perf_counter() - feature_started
                    print(f"{role} {index}/{len(batch)} elapsed={elapsed:.1f}s", flush=True)

        extract_batch(galleries, "gallery", gallery_cache)
        extract_batch(queries, "query", query_cache)
        feature_seconds = time.perf_counter() - feature_started

        scoring_started = time.perf_counter()
        scores, _, matrix_seconds, _ = _score_matrix(
            queries, galleries, query_cache, gallery_cache, 16,
            expected_dimension=512, mode="plaintext_selection", split=THRESHOLD_SELECTION,
            score_aggregation="max",
        )
        detected_queries = {qid for qid, cache in query_cache.items() if cache.features}
        pairs = build_similarity_pairs(selection_records, identity_split, scores, detected_queries)
        retrieval = retrieval_metrics(selection_records, identity_split, pairs, detected_queries)
        threshold = select_threshold_at_far(pairs, target_far=TARGET_FAR)
        scoring_seconds = time.perf_counter() - scoring_started
        total_seconds = time.perf_counter() - total_started
        counts = Counter()
        for row in feature_rows:
            counts[row["role"] + "_with_features"] += row["feature_count"] > 0
            counts[row["role"] + "_without_features"] += row["feature_count"] == 0
            counts[row["role"] + "_features"] += row["feature_count"]
            counts[row["role"] + "_feature_extraction_failures"] += row["feature_extraction_failure_count"]
        unscored = sum(row["similarity"] is None for row in pairs)
        metrics = {
            "status": "complete", "protocol": "identity-disjoint-query-gallery",
            "evaluation_scope": THRESHOLD_SELECTION, "target_far": TARGET_FAR,
            "identity_count": len(selection_ids), "gallery_count": len(galleries),
            "query_count": len(queries), "model": "InsightFace buffalo_l w600k_r50",
            "retrieval": retrieval, "tar_far_selection": threshold,
            "threshold_source": "ArcFace plaintext threshold_selection only; selected independently",
            "pair_counts": {"total": len(pairs), "scored": len(pairs) - unscored,
                            "unscored": unscored,
                            "genuine_total": sum(row["pair_type"] == "genuine" for row in pairs),
                            "genuine_scored": sum(row["pair_type"] == "genuine" and row["similarity"] is not None for row in pairs),
                            "impostor_total": sum(row["pair_type"] == "impostor" for row in pairs),
                            "impostor_scored": sum(row["pair_type"] == "impostor" and row["similarity"] is not None for row in pairs),
                            "unscored_rate": unscored / len(pairs)},
            "feature_availability": dict(counts),
            "feature_extraction_failure_record_count": len(failure_rows),
            "timing_seconds": {"feature_extraction": feature_seconds,
                               "similarity_calculation": matrix_seconds,
                               "scoring_and_metrics": scoring_seconds,
                               "total_wall": total_seconds},
            "matrix_audit": {"expected_pairs": 502 * 502, "actual_pairs": len(pairs),
                             "cross_partition_pairs": 0},
            "test_evaluation_run": False,
        }
        if len(pairs) != 502 * 502 or unscored != len(pairs) - metrics["pair_counts"]["scored"]:
            raise RuntimeError("selection score matrix failed completeness checks")

        write_json(OUTPUT / "metrics.json", metrics)
        with (OUTPUT / "feature_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = list(feature_rows[0]) if feature_rows else []
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(feature_rows)
        with (OUTPUT / "feature_failures.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ("mode", "media_id", "identity_id", "frame_index", "face_index", "stage",
                      "exception_type", "exception_message")
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failure_rows)
        with (OUTPUT / "genuine_impostor_pairs.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ("query_id", "gallery_id", "similarity", "pair_type", "split", "score_status")
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pairs)

        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        baseline_selection_queries = {}
        with (BASELINE.parent / "query_results.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("mode") == "plaintext" and row.get("query_split") == THRESHOLD_SELECTION:
                    baseline_selection_queries[row["query_id"]] = row
        baseline_selection_candidates: dict[str, list[dict]] = {}
        with (BASELINE.parent / "candidate_results.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("mode") == "plaintext" and row.get("split") == THRESHOLD_SELECTION:
                    baseline_selection_candidates.setdefault(row["query_id"], []).append(row)
        baseline_hits_top1 = sum(row.get("top1_correct", "").lower() == "true"
                                 for row in baseline_selection_queries.values())
        baseline_hits_top5 = sum(any(candidate.get("is_correct_match", "").lower() == "true" and
                                     candidate.get("rank", "").isdigit() and int(candidate["rank"]) <= 5
                                     for candidate in baseline_selection_candidates.get(qid, []))
                                 for qid in baseline_selection_queries)
        baseline_retrieval = {
            "query_count": len(baseline_selection_queries),
            "top1_accuracy": {"hits": baseline_hits_top1, "denominator": len(baseline_selection_queries),
                              "value": baseline_hits_top1 / len(baseline_selection_queries)},
            "recall_at_5": {"hits": baseline_hits_top5, "denominator": len(baseline_selection_queries),
                             "value": baseline_hits_top5 / len(baseline_selection_queries)},
        }
        baseline_threshold = baseline["threshold_selection"]
        if len(baseline_selection_queries) != 502:
            raise RuntimeError(f"expected 502 baseline selection queries, got {len(baseline_selection_queries)}")
        comparison = {
            "baseline_source": str(BASELINE),
            "baseline_model": "OpenCV SFace",
            "baseline_top1": baseline_retrieval["top1_accuracy"],
            "baseline_recall_at_5": baseline_retrieval["recall_at_5"],
            "arcface_top1_percentage_point_delta": 100 * (retrieval["top1_accuracy"]["value"] - baseline_retrieval["top1_accuracy"]["value"]),
            "arcface_recall_at_5_percentage_point_delta": 100 * (retrieval["recall_at_5"]["value"] - baseline_retrieval["recall_at_5"]["value"]),
            "baseline_selection_tar": baseline_threshold.get("selection_tar"),
            "baseline_selection_far": baseline_threshold.get("selection_far"),
            "arcface_tar_percentage_point_delta": (100 * (threshold.get("selection_tar", 0) - baseline_threshold["selection_tar"])
                                                    if threshold.get("status") == "estimated" else None),
            "arcface_far_delta": (threshold.get("selection_far") - baseline_threshold["selection_far"]
                                  if threshold.get("status") == "estimated" else None),
            "test_gate_met": bool(
                retrieval["top1_accuracy"]["value"] > baseline_retrieval["top1_accuracy"]["value"] and
                retrieval["recall_at_5"]["value"] > baseline_retrieval["recall_at_5"]["value"] and
                threshold.get("status") == "estimated" and threshold.get("selection_far", 1) <= TARGET_FAR and
                threshold.get("selection_tar", 0) > baseline_threshold["selection_tar"]),
            "test_evaluation_run": False,
        }
        write_json(OUTPUT / "comparison.json", comparison)
        report = [
            "# ArcFace Complete YTF Selection Evaluation",
            "",
            "Scope: complete 502-identity threshold-selection partition; plaintext only. The test partition was not accessed.",
            "Gallery used the formal baseline path: all configured 1 fps samples, all valid YuNet detections, per-frame features retained, and maximum cosine score per Query/Gallery video pair. No quality filter or feature mean aggregation was enabled.",
            f"ArcFace threshold was selected from this run's plaintext selection pairs only: `{threshold.get('threshold')}`. The SFace threshold was not reused.",
            "",
            "| Metric | SFace selection baseline | ArcFace selection | Difference |",
            "|---|---:|---:|---:|",
            f"| Top-1 | {baseline_retrieval['top1_accuracy']['hits']}/502 ({baseline_retrieval['top1_accuracy']['value']:.2%}) | {retrieval['top1_accuracy']['hits']}/502 ({retrieval['top1_accuracy']['value']:.2%}) | {comparison['arcface_top1_percentage_point_delta']:+.2f} pp |",
            f"| Recall@5 | {baseline_retrieval['recall_at_5']['hits']}/502 ({baseline_retrieval['recall_at_5']['value']:.2%}) | {retrieval['recall_at_5']['hits']}/502 ({retrieval['recall_at_5']['value']:.2%}) | {comparison['arcface_recall_at_5_percentage_point_delta']:+.2f} pp |",
            f"| TAR at selection FAR<=1e-3 | {baseline_threshold['selection_tar']:.4%} | {threshold.get('selection_tar', 0):.4%} | {comparison['arcface_tar_percentage_point_delta']:+.2f} pp |",
            f"| Selection FAR | {baseline_threshold['selection_far']:.8f} | {threshold.get('selection_far')} | {comparison['arcface_far_delta']:+.8f} |",
            "",
            f"Pairs: {len(pairs)} total, {len(pairs) - unscored} scored, {unscored} unscored. Effective impostors: {threshold.get('scored_impostor_count')}; FAR resolution: {threshold.get('far_resolution')}; estimable: {threshold.get('status') == 'estimated'}.",
            f"Feature extraction failures: {len(failure_rows)} records. Effective queries: {retrieval['detected_query_count']}/502. Galleries with at least one embedding: {counts['gallery_with_features']}/502.",
            f"Timing: features {feature_seconds:.1f}s; similarity {matrix_seconds:.1f}s; total {total_seconds:.1f}s.",
            f"Entry gate comparison: {'met' if comparison['test_gate_met'] else 'not met'}; no test run was performed.",
            "",
            "The selection TAR/FAR is a threshold-selection operating point, not an independent test estimate. Any later test evaluation must use the frozen ArcFace threshold recorded here and cannot retune it.",
        ]
        with (OUTPUT / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(report) + "\n")
        write_status(status_path, "complete", started=started_at,
                     detail=f"published {len(pairs)} selection pairs; test not run")
        print(f"completed output={OUTPUT}", flush=True)
        print(json.dumps({"retrieval": retrieval, "tar_far_selection": threshold,
                          "pair_counts": metrics["pair_counts"], "timing_seconds": metrics["timing_seconds"],
                          "comparison": comparison}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_status(status_path, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     started=started_at, detail=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
