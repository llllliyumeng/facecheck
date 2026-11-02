#!/usr/bin/env python3
"""Run a small, plaintext ArcFace YuNet threshold smoke comparison.

Only YuNet's confidence threshold varies. Exactly 20 deterministic identities
are sampled from the existing threshold-selection cohort and compared with the
same identities in the recorded 0.90 baseline. This is a smoke check, not an
accuracy or FAR/TAR estimate. Test media is never opened or scored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from facecheck.face_detection import FaceDetectionConfig  # noqa: E402
from facecheck.face_embedding import SFaceConfig  # noqa: E402
from facecheck.ytf_qg_protocol import (  # noqa: E402
    THRESHOLD_SELECTION,
    build_similarity_pairs,
    retrieval_metrics,
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
BASELINE = ROOT / "data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext"
DEFAULT_OUTPUT = ROOT / "data/ytf/experiments/ytf_arcface_yunet085_smoke20_20261101"
SMOKE_IDENTITY_COUNT = 20
SMOKE_SAMPLE_SEED = "facecheck-arcface-yunet-threshold-smoke-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_smoke_identities(identity_ids: list[str]) -> list[str]:
    ranked = sorted(
        identity_ids,
        key=lambda identity_id: hashlib.sha256(
            f"{SMOKE_SAMPLE_SEED}:{identity_id}".encode("utf-8")
        ).hexdigest(),
    )
    return sorted(ranked[:SMOKE_IDENTITY_COUNT])


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_status(path: Path, state: str, *, started: str, detail: str | None = None) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "state": state,
        "started_at_utc": started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pair_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw = row.get("similarity", "")
            similarity = None if raw in ("", "None", "null") else float(raw)
            rows.append({
                "query_id": row["query_id"],
                "gallery_id": row["gallery_id"],
                "similarity": similarity,
                "pair_type": row["pair_type"],
                "split": row["split"],
            })
    return rows


def load_detected_queries(path: Path) -> set[str]:
    detected: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("role") == "query" and int(row.get("feature_count") or 0) > 0:
                detected.add(row["media_id"])
    return detected


def query_decisions(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    detected: set[str],
) -> dict[str, dict[str, Any]]:
    identity_by_media = {str(row["video_id"]): str(row["identity_id"]) for row in records}
    queries = {str(row["video_id"]): str(row["identity_id"])
               for row in records if row["split"] == "query"}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_query[str(pair["query_id"])].append(pair)

    result: dict[str, dict[str, Any]] = {}
    for query_id, identity_id in queries.items():
        ranked = sorted(
            (row for row in by_query.get(query_id, []) if row["similarity"] is not None),
            key=lambda row: (-float(row["similarity"]), str(row["gallery_id"])),
        )
        winner = ranked[0] if query_id in detected and ranked else None
        correct_rank = next(
            (index for index, row in enumerate(ranked, 1)
             if identity_by_media[str(row["gallery_id"])] == identity_id),
            None,
        )
        result[query_id] = {
            "identity_id": identity_id,
            "detected": query_id in detected,
            "top1_gallery_id": str(winner["gallery_id"]) if winner else None,
            "top1_correct": bool(winner and identity_by_media[str(winner["gallery_id"])] == identity_id),
            "correct_identity_rank": correct_rank,
            "recall_at_5": bool(correct_rank is not None and correct_rank <= 5),
        }
    return result


def compare_decisions(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> dict[str, int]:
    if set(baseline) != set(candidate):
        raise RuntimeError("paired Query IDs differ between threshold arms")
    counts: Counter[str] = Counter()
    for query_id, before in baseline.items():
        after = candidate[query_id]
        counts["baseline_detected"] += before["detected"]
        counts["candidate_detected"] += after["detected"]
        counts["newly_detected"] += not before["detected"] and after["detected"]
        counts["lost_detection"] += before["detected"] and not after["detected"]
        counts["baseline_top1_correct"] += before["top1_correct"]
        counts["candidate_top1_correct"] += after["top1_correct"]
        counts["newly_correct_top1"] += not before["top1_correct"] and after["top1_correct"]
        counts["regressed_top1"] += before["top1_correct"] and not after["top1_correct"]
        counts["baseline_recall_at_5"] += before["recall_at_5"]
        counts["candidate_recall_at_5"] += after["recall_at_5"]
    counts["query_count"] = len(baseline)
    return dict(counts)


def validate_pair_matrix(
    records: list[dict[str, Any]], pairs: list[dict[str, Any]],
) -> dict[str, int]:
    queries = {str(row["video_id"]): str(row["identity_id"])
               for row in records if row["split"] == "query"}
    galleries = {str(row["video_id"]): str(row["identity_id"])
                 for row in records if row["split"] == "gallery"}
    expected_count = len(queries) * len(galleries)
    seen: set[tuple[str, str]] = set()
    genuine = 0
    impostor = 0
    for pair in pairs:
        query_id = str(pair["query_id"])
        gallery_id = str(pair["gallery_id"])
        key = (query_id, gallery_id)
        if query_id not in queries or gallery_id not in galleries or key in seen:
            raise RuntimeError("pair matrix contains an unknown or duplicate Query/Gallery pair")
        if pair.get("split") != THRESHOLD_SELECTION:
            raise RuntimeError("pair matrix contains a record outside threshold_selection")
        expected_type = "genuine" if queries[query_id] == galleries[gallery_id] else "impostor"
        if pair.get("pair_type") != expected_type:
            raise RuntimeError("pair matrix has an incorrect genuine/impostor label")
        seen.add(key)
        if expected_type == "genuine":
            genuine += 1
        else:
            impostor += 1
    if len(pairs) != expected_count or len(seen) != expected_count:
        raise RuntimeError(f"incomplete pair matrix: expected {expected_count}, found {len(seen)}")
    return {"total": len(pairs), "genuine": genuine, "impostor": impostor}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="new, nonexistent experiment directory")
    parser.add_argument("--score-threshold", type=float, default=0.85,
                        help="candidate YuNet confidence threshold; comparison is the recorded 0.90 arm")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    threshold = float(args.score_threshold)
    if not 0 < threshold < 1:
        raise SystemExit("--score-threshold must be between zero and one")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    if output.resolve().parent != (ROOT / "data/ytf/experiments").resolve():
        raise SystemExit("output must be a new direct child of data/ytf/experiments")

    manifest_sha256 = sha256(MANIFEST)
    split_sha256 = sha256(SPLIT_PATH)
    records, all_splits, manifest_json, split_json = _load_inputs(MANIFEST, SPLIT_PATH)
    cohort_ids = sorted(identity for identity, side in all_splits.items()
                        if side == THRESHOLD_SELECTION)
    cohort_id_set = set(cohort_ids)
    cohort_records = [row for row in records if row["identity_id"] in cohort_id_set]
    cohort_split = {identity: THRESHOLD_SELECTION for identity in cohort_ids}
    cohort_layout = validate_layout(cohort_records, cohort_split)
    if (len(cohort_ids) != 502 or cohort_layout["gallery_count"] != 502 or
            cohort_layout["query_count"] != 502):
        raise SystemExit("expected the unchanged 502-identity selection cohort")
    if sha256(MANIFEST) != manifest_sha256 or sha256(SPLIT_PATH) != split_sha256:
        raise SystemExit("manifest or identity split changed while loading experiment metadata")

    selection_ids = choose_smoke_identities(cohort_ids)
    selected = set(selection_ids)
    selection_records = [row for row in cohort_records if row["identity_id"] in selected]
    identity_split = {identity: THRESHOLD_SELECTION for identity in selection_ids}
    layout = validate_layout(selection_records, identity_split)
    if (len(selection_ids) != SMOKE_IDENTITY_COUNT or
            layout["gallery_count"] != SMOKE_IDENTITY_COUNT or
            layout["query_count"] != SMOKE_IDENTITY_COUNT):
        raise SystemExit("smoke cohort must contain exactly 20 selection identities")
    if not all(path.is_file() and not path.is_symlink()
               for path in (YUNET, SFACE, ARCFACE)):
        raise SystemExit("one or more configured model files are missing or unsafe")

    baseline_config_path = BASELINE / "config.json"
    baseline_identities_path = BASELINE / "identity_ids.json"
    baseline_metrics_path = BASELINE / "metrics.json"
    baseline_validation_path = BASELINE / "validation.json"
    baseline_status_path = BASELINE / "run_status.json"
    baseline_pairs_path = BASELINE / "genuine_impostor_pairs.csv"
    baseline_features_path = BASELINE / "feature_summary.csv"
    if not all(path.is_file() and not path.is_symlink() for path in (
        baseline_config_path, baseline_identities_path, baseline_metrics_path,
        baseline_validation_path, baseline_status_path, baseline_pairs_path,
        baseline_features_path,
    )):
        raise SystemExit("the recorded ArcFace 0.90 selection baseline is incomplete")
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    baseline_identities = json.loads(baseline_identities_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_validation = json.loads(baseline_validation_path.read_text(encoding="utf-8"))
    baseline_status = json.loads(baseline_status_path.read_text(encoding="utf-8"))
    if (baseline_config.get("manifest_sha256") != manifest_sha256 or
            baseline_identities.get("identity_split_sha256") != split_sha256 or
            baseline_identities.get("identity_ids") != cohort_ids or
            baseline_identities.get("identity_count") != len(cohort_ids)):
        raise SystemExit("baseline manifest or identity partition differs from current selection inputs")
    if (baseline_config.get("identity_count") != 502 or
            baseline_config.get("gallery_count") != 502 or
            baseline_config.get("query_count") != 502 or
            baseline_config.get("source_partition") != "existing identity-disjoint threshold_selection only" or
            baseline_config.get("yunet", {}).get("score_threshold") != 0.9 or
            baseline_config.get("model_sha256", {}).get("yunet") != sha256(YUNET) or
            baseline_config.get("model_sha256", {}).get("sface_reference") != sha256(SFACE) or
            baseline_config.get("model_sha256", {}).get("arcface") != sha256(ARCFACE)):
        raise SystemExit("the recorded baseline is not the expected 502-identity ArcFace 0.90 arm")
    if (baseline_status.get("state") != "complete" or
            baseline_validation.get("status") != "passed" or
            baseline_validation.get("test_evaluation_run") is not False or
            baseline_metrics.get("status") != "complete" or
            baseline_metrics.get("pair_counts", {}).get("total") != 502 * 502):
        raise SystemExit("the recorded baseline did not pass its stored completeness validation")

    smoke_media_ids = {str(row["video_id"]) for row in selection_records}
    baseline_pairs_all = load_pair_rows(baseline_pairs_path)
    baseline_pairs = [pair for pair in baseline_pairs_all
                      if pair["query_id"] in smoke_media_ids and
                      pair["gallery_id"] in smoke_media_ids]
    baseline_pair_audit = validate_pair_matrix(selection_records, baseline_pairs)
    baseline_detected_all = load_detected_queries(baseline_features_path)
    query_ids = {str(row["video_id"]) for row in selection_records
                 if row["split"] == "query"}
    baseline_detected = baseline_detected_all & query_ids
    baseline_retrieval = retrieval_metrics(
        selection_records, identity_split, baseline_pairs, baseline_detected
    )
    baseline_query_decisions = query_decisions(
        selection_records, baseline_pairs, baseline_detected
    )
    baseline_started = datetime.now(timezone.utc).isoformat()

    output.mkdir(parents=True, mode=0o700)
    status_path = output / "run_status.json"
    write_status(status_path, "running", started=baseline_started)
    started = time.perf_counter()

    try:
        hash_paths = {
            str(path.relative_to(ROOT)): path
            for path in (
                MANIFEST, SPLIT_PATH, YUNET, SFACE, ARCFACE,
                baseline_config_path, baseline_identities_path,
                baseline_metrics_path, baseline_validation_path,
                baseline_status_path, baseline_pairs_path, baseline_features_path,
                Path(__file__).resolve(),
                ROOT / "src/facecheck/face_detection.py",
                ROOT / "src/facecheck/face_embedding.py",
                ROOT / "src/facecheck/ytf_qg_protocol.py",
                ROOT / "src/facecheck/ytf_runner.py",
            )
        }
        run_input_sha256 = {name: sha256(path) for name, path in hash_paths.items()}
        model_hashes = {
            "yunet": run_input_sha256[str(YUNET.relative_to(ROOT))],
            "sface_reference": run_input_sha256[str(SFACE.relative_to(ROOT))],
            "arcface": run_input_sha256[str(ARCFACE.relative_to(ROOT))],
        }
        source_hashes = {
            name: digest for name, digest in run_input_sha256.items()
            if name.startswith("src/facecheck/") or name == str(Path(__file__).resolve().relative_to(ROOT))
        }
        write_json(output / "identities.json", {
            "source_split": THRESHOLD_SELECTION,
            "identity_split_seed": split_json["seed"],
            "identity_split_sha256": split_sha256,
            "sampling_seed": SMOKE_SAMPLE_SEED,
            "sampling_method": "20 identities with the lowest SHA-256 of seed:identity_id, then sorted by ID",
            "selection_cohort_identity_count": len(cohort_ids),
            "identity_count": len(selection_ids),
            "identity_ids": selection_ids,
            "test_media_loaded_or_scored": False,
        })
        write_json(output / "config.json", {
            "purpose": "20-identity smoke comparison of YuNet confidence thresholds using existing ArcFace selection data",
            "created_at_utc": baseline_started,
            "evaluation_scope": THRESHOLD_SELECTION,
            "formal_test_run": False,
            "test_media_loaded_or_scored": False,
            "smoke_test_only": True,
            "candidate_variable": "YuNet score_threshold",
            "candidate_score_threshold": threshold,
            "baseline_score_threshold": 0.9,
            "sample": {
                "selection_cohort_identity_count": len(cohort_ids),
                "identity_count": len(selection_ids),
                "sampling_seed": SMOKE_SAMPLE_SEED,
                "sampling_method": "20 identities with the lowest SHA-256 of seed:identity_id, then sorted by ID",
                "identity_ids": selection_ids,
            },
            "fixed_parameters": {
                "model": "InsightFace buffalo_l w600k_r50",
                "frame_interval_seconds": 1.0,
                "nms_threshold": 0.3,
                "gallery_quality_filter": False,
                "gallery_feature_aggregation": "none",
                "faces": "all valid detections",
                "query_flow": "existing Query images; all detected faces; no annotation-box selection",
                "pair_score": "maximum cosine across Query/Gallery frame features",
            },
            "identity_count": len(selection_ids),
            "gallery_count": layout["gallery_count"],
            "query_count": layout["query_count"],
            "manifest_path": str(MANIFEST),
            "manifest_sha256": manifest_sha256,
            "identity_split_path": str(SPLIT_PATH),
            "identity_split_sha256": split_sha256,
            "identity_split_seed": split_json["seed"],
            "baseline": {
                "path": str(BASELINE.relative_to(ROOT)),
                "config_sha256": sha256(baseline_config_path),
                "metrics_sha256": sha256(baseline_metrics_path),
                "validation_sha256": sha256(baseline_validation_path),
                "pairs_sha256": sha256(baseline_pairs_path),
                "feature_summary_sha256": sha256(baseline_features_path),
                "identity_split_sha256": baseline_identities["identity_split_sha256"],
            },
            "models": {
                "arcface_path": str(ARCFACE), "arcface_sha256": model_hashes["arcface"],
                "yunet_path": str(YUNET), "yunet_sha256": model_hashes["yunet"],
                "sface_reference_path": str(SFACE), "sface_reference_sha256": model_hashes["sface_reference"],
            },
            "runtime_environment": {
                "platform": platform.platform(), "machine": platform.machine(),
                "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
            },
            "source_sha256": source_hashes,
            "input_sha256": run_input_sha256,
            "gallery_sampled_frame_count": sum(int(row.get("ytf_frame_count", 0))
                                                for row in selection_records if row["split"] == "gallery"),
            "statistical_limit": "With 20 queries and at most 380 impostor pairs, this smoke run cannot estimate TAR at FAR=1e-3. Do not interpret Top-1 or Recall@5 as a population estimate.",
            "comparison_scope_note": "This pairs a deterministic 20-identity subset of the existing threshold_selection cohort with the same rows from the recorded 0.90 run. It is a smoke check only, not an independent estimate; test media files are not opened or scored.",
        })
        write_json(output / "source_sha256.json", source_hashes)

        galleries = [row for row in selection_records if row["split"] == "gallery"]
        queries = [row for row in selection_records if row["split"] == "query"]
        backend = OpenCVFeatureBackend(
            FaceDetectionConfig(YUNET, score_threshold=threshold,
                                frame_interval_seconds=1.0),
            SFaceConfig(SFACE), recognition_model="arcface",
            arcface_model_path=ARCFACE,
        )
        gallery_cache: dict[str, CachedMediaFeatures] = {}
        query_cache: dict[str, CachedMediaFeatures] = {}
        feature_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        feature_started = time.perf_counter()

        def extract_batch(batch: list[dict], role: str,
                          cache: dict[str, CachedMediaFeatures]) -> None:
            for index, record in enumerate(batch, start=1):
                extracted = backend.extract({
                    **record,
                    "_experiment_mode": "plaintext_selection_candidate",
                    "_gallery_representation_enabled": role == "gallery",
                    "_gallery_quality_filter": False,
                    "_gallery_feature_aggregation": "none",
                })
                media_id = str(record["video_id"])
                cache[media_id] = CachedMediaFeatures(
                    str(record["identity_id"]), media_id, THRESHOLD_SELECTION,
                    "video" if role == "gallery" else "image",
                    extracted.features, extracted.sampled_frame_count,
                )
                feature_rows.append({
                    "media_id": media_id, "identity_id": record["identity_id"],
                    "role": role,
                    "sampled_frame_count": extracted.sampled_frame_count,
                    "detected_face_count": extracted.detected_face_count,
                    "feature_count": len(extracted.features),
                    "feature_extraction_failure_count": extracted.feature_extraction_failure_count,
                    "invalid_detection_record_count": extracted.invalid_detection_record_count,
                    "detection_failure_count": extracted.detection_failure_count,
                    "video_read_decode_extract_seconds": extracted.timings.get("video_read_decode_extract_seconds"),
                    "face_detection_seconds": extracted.timings.get("face_detection_seconds"),
                    "feature_extraction_seconds": extracted.timings.get("feature_extraction_seconds"),
                })
                failure_rows.extend(extracted.failures)
                if index % 5 == 0 or index == len(batch):
                    print(f"{role} {index}/{len(batch)} elapsed={time.perf_counter() - feature_started:.1f}s",
                          flush=True)

        extract_batch(galleries, "gallery", gallery_cache)
        extract_batch(queries, "query", query_cache)
        feature_seconds = time.perf_counter() - feature_started

        scoring_started = time.perf_counter()
        scores, _, matrix_seconds, _ = _score_matrix(
            queries, galleries, query_cache, gallery_cache, 16,
            expected_dimension=512,
            mode="plaintext_selection_candidate",
            split=THRESHOLD_SELECTION,
            score_aggregation="max",
        )
        detected_queries = {media_id for media_id, cache in query_cache.items()
                            if cache.features}
        candidate_pairs = build_similarity_pairs(
            selection_records, identity_split, scores, detected_queries
        )
        candidate_pair_audit = validate_pair_matrix(selection_records, candidate_pairs)
        candidate_retrieval = retrieval_metrics(
            selection_records, identity_split, candidate_pairs, detected_queries
        )
        candidate_decisions = query_decisions(
            selection_records, candidate_pairs, detected_queries
        )
        paired = compare_decisions(baseline_query_decisions, candidate_decisions)
        scoring_seconds = time.perf_counter() - scoring_started
        if baseline_pair_audit["total"] != candidate_pair_audit["total"]:
            raise RuntimeError("baseline and candidate pair matrices differ in size")
        expected_media_ids = {str(row["video_id"]) for row in selection_records}
        if (len(feature_rows) != len(selection_records) or
                {str(row["media_id"]) for row in feature_rows} != expected_media_ids):
            raise RuntimeError("feature extraction records do not cover the 20 smoke media exactly once")
        if len(candidate_decisions) != SMOKE_IDENTITY_COUNT:
            raise RuntimeError("query comparison does not cover all 20 smoke identities")
        changed_inputs = [name for name, path in hash_paths.items()
                          if sha256(path) != run_input_sha256[name]]
        if changed_inputs:
            raise RuntimeError(f"experiment inputs changed during execution: {changed_inputs}")

        feature_counts = Counter()
        for row in feature_rows:
            feature_counts[f"{row['role']}_with_features"] += row["feature_count"] > 0
            feature_counts[f"{row['role']}_without_features"] += row["feature_count"] == 0
            feature_counts[f"{row['role']}_features"] += row["feature_count"]
            feature_counts[f"{row['role']}_feature_extraction_failures"] += row["feature_extraction_failure_count"]
        metrics = {
            "status": "complete",
            "evaluation_scope": THRESHOLD_SELECTION,
            "formal_test_run": False,
            "test_media_loaded_or_scored": False,
            "smoke_test_only": True,
            "identity_count": len(selection_ids),
            "gallery_count": layout["gallery_count"],
            "query_count": layout["query_count"],
            "candidate": {
                "score_threshold": threshold,
                "retrieval": candidate_retrieval,
                "pair_count": len(candidate_pairs),
                "pair_audit": candidate_pair_audit,
                "unscored_pair_count": sum(pair["similarity"] is None for pair in candidate_pairs),
                "feature_availability": dict(feature_counts),
                "feature_failure_record_count": len(failure_rows),
            },
            "baseline_090": {
                "score_threshold": 0.9,
                "retrieval": baseline_retrieval,
                "pair_count": len(baseline_pairs),
                "pair_audit": baseline_pair_audit,
                "unscored_pair_count": sum(pair["similarity"] is None for pair in baseline_pairs),
                "feature_availability": {
                    "queries_with_features": len(baseline_detected),
                    "queries_without_features": layout["query_count"] - len(baseline_detected),
                },
            },
            "paired_query_changes": paired,
            "deltas": {
                "top1_percentage_points": 100 * (
                    candidate_retrieval["top1_accuracy"]["value"] -
                    baseline_retrieval["top1_accuracy"]["value"]
                ),
                "recall_at_5_percentage_points": 100 * (
                    candidate_retrieval["recall_at_5"]["value"] -
                    baseline_retrieval["recall_at_5"]["value"]
                ),
            },
            "timing_seconds": {
                "feature_extraction": feature_seconds,
                "similarity_calculation": matrix_seconds,
                "scoring_and_metrics": scoring_seconds,
                "total_wall": time.perf_counter() - started,
            },
            "uncertainty_note": "This is a 20-identity smoke comparison of previously used selection data. With at most 380 impostor pairs, FAR=1e-3 has insufficient trial resolution. The result is not a population accuracy or TAR/FAR estimate and cannot justify changing defaults.",
        }

        validation = {
            "status": "passed",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": [
                "20 deterministic identities sampled only from threshold_selection",
                "20 Query and 20 Gallery records extracted exactly once",
                "400 candidate pairs form a unique within-sample Cartesian product",
                "400 baseline pairs filtered from the recorded complete selection matrix",
                "Top-1 and Recall@5 agree with the per-Query comparison rows",
                "manifest, split, models, baseline, and source hashes remained unchanged",
                "test media files were not opened or scored",
                "no TAR@FAR=1e-3 or target-pass claim was produced",
            ],
            "identity_count": len(selection_ids),
            "query_count": layout["query_count"],
            "gallery_count": layout["gallery_count"],
            "candidate_pair_audit": candidate_pair_audit,
            "baseline_pair_audit": baseline_pair_audit,
            "test_media_opened_or_scored": False,
        }
        if (candidate_retrieval["top1_accuracy"]["hits"] !=
                sum(row["top1_correct"] for row in candidate_decisions.values()) or
                candidate_retrieval["recall_at_5"]["hits"] !=
                sum(row["recall_at_5"] for row in candidate_decisions.values())):
            raise RuntimeError("retrieval summary differs from independently ranked Query rows")

        write_json(output / "metrics.json", metrics)
        write_json(output / "validation.json", validation)
        with (output / "feature_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
        with (output / "feature_failures.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ("mode", "media_id", "identity_id", "frame_index", "face_index",
                      "stage", "exception_type", "exception_message")
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(failure_rows)
        with (output / "candidate_pairs.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ("query_id", "gallery_id", "similarity", "pair_type", "split", "score_status")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({
                **pair,
                "score_status": "scored" if pair["similarity"] is not None else "unscored",
            } for pair in candidate_pairs)
        with (output / "query_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            fields = (
                "query_id", "identity_id", "baseline_detected", "candidate_detected",
                "baseline_top1_gallery_id", "candidate_top1_gallery_id",
                "baseline_top1_correct", "candidate_top1_correct",
                "baseline_correct_identity_rank", "candidate_correct_identity_rank",
                "baseline_recall_at_5", "candidate_recall_at_5",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for query_id, before in baseline_query_decisions.items():
                after = candidate_decisions[query_id]
                writer.writerow({
                    "query_id": query_id,
                    "identity_id": before["identity_id"],
                    "baseline_detected": before["detected"],
                    "candidate_detected": after["detected"],
                    "baseline_top1_gallery_id": before["top1_gallery_id"],
                    "candidate_top1_gallery_id": after["top1_gallery_id"],
                    "baseline_top1_correct": before["top1_correct"],
                    "candidate_top1_correct": after["top1_correct"],
                    "baseline_correct_identity_rank": before["correct_identity_rank"],
                    "candidate_correct_identity_rank": after["correct_identity_rank"],
                    "baseline_recall_at_5": before["recall_at_5"],
                    "candidate_recall_at_5": after["recall_at_5"],
                })

        comparison = {
            "variable": "YuNet score_threshold",
            "smoke_test_only": True,
            "identity_count": len(selection_ids),
            "pair_count": candidate_pair_audit["total"],
            "baseline": {"score_threshold": 0.9,
                         "top1": baseline_retrieval["top1_accuracy"],
                         "recall_at_5": baseline_retrieval["recall_at_5"]},
            "candidate": {"score_threshold": threshold,
                          "top1": candidate_retrieval["top1_accuracy"],
                          "recall_at_5": candidate_retrieval["recall_at_5"]},
            "deltas": metrics["deltas"],
            "paired_query_changes": paired,
            "test_evaluation_run": False,
            "promotion_gate": "smoke only; no default change. Any promising candidate must be frozen and checked on a new selection run before an independent test.",
        }
        write_json(output / "comparison.json", comparison)
        report = [
            "# ArcFace YuNet Threshold Smoke Comparison",
            "",
            f"One-variable smoke comparison on {len(selection_ids)} deterministic identities sampled from the 502-identity threshold_selection cohort. The recorded ArcFace 0.90 run is the paired baseline; the candidate changes only YuNet score threshold to {threshold:.2f}. Both use all 1 fps samples, all valid detections, no quality filter, no feature aggregation, and maximum cosine scoring.",
            "This is a smoke check only. Test media files were not opened or scored. The identities come from a previously used selection cohort, and the sample is too small for a population accuracy claim or FAR=1e-3 estimate.",
            "",
            "| Metric | YuNet 0.90 baseline | Candidate | Change |",
            "|---|---:|---:|---:|",
            f"| Top-1 | {baseline_retrieval['top1_accuracy']['hits']}/{len(selection_ids)} ({baseline_retrieval['top1_accuracy']['value']:.2%}) | {candidate_retrieval['top1_accuracy']['hits']}/{len(selection_ids)} ({candidate_retrieval['top1_accuracy']['value']:.2%}) | {metrics['deltas']['top1_percentage_points']:+.2f} pp |",
            f"| Recall@5 | {baseline_retrieval['recall_at_5']['hits']}/{len(selection_ids)} ({baseline_retrieval['recall_at_5']['value']:.2%}) | {candidate_retrieval['recall_at_5']['hits']}/{len(selection_ids)} ({candidate_retrieval['recall_at_5']['value']:.2%}) | {metrics['deltas']['recall_at_5_percentage_points']:+.2f} pp |",
            f"| Detected queries | {baseline_retrieval['detected_query_count']}/{len(selection_ids)} | {candidate_retrieval['detected_query_count']}/{len(selection_ids)} | {paired['newly_detected'] - paired['lost_detection']:+d} |",
            "",
            f"Paired Query changes: {paired}.",
            f"Candidate feature failure rows: {len(failure_rows)}; pair rows: {len(candidate_pairs)} ({candidate_pair_audit['genuine']} genuine, {candidate_pair_audit['impostor']} impostor); unscored pairs: {metrics['candidate']['unscored_pair_count']}.",
            f"Runtime: feature extraction {feature_seconds:.1f}s; similarity calculation {matrix_seconds:.1f}s; total {metrics['timing_seconds']['total_wall']:.1f}s.",
            "",
            "Do not use these percentages to claim the 95% Top-1 or 90% TAR targets are met. No TAR@FAR=1e-3 was calculated because the sample cannot resolve that FAR. Do not update recognition defaults from this smoke run.",
        ]
        with (output / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(report) + "\n")
        write_status(status_path, "complete", started=baseline_started,
                     detail=f"published {len(candidate_pairs)} candidate selection pairs; test not accessed")
        print(json.dumps({
            "output": str(output),
            "identity_count": len(selection_ids),
            "pair_count": candidate_pair_audit["total"],
            "baseline_top1": baseline_retrieval["top1_accuracy"],
            "candidate_top1": candidate_retrieval["top1_accuracy"],
            "baseline_recall_at_5": baseline_retrieval["recall_at_5"],
            "candidate_recall_at_5": candidate_retrieval["recall_at_5"],
            "paired_query_changes": paired,
            "timing_seconds": metrics["timing_seconds"],
        }, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_status(status_path, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     started=baseline_started, detail=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
