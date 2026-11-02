"""Exploratory SFace/ArcFace comparison on one fixed selection identity sample."""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


SAMPLE_SIZE = 100
SAMPLE_SEED = 20261101
OUTPUT = ROOT / "data/ytf/experiments/ytf_models_sample_100_20261101"
MANIFEST = ROOT / "data/ytf/experiment_full.json"
SPLIT = ROOT / "data/ytf/query_gallery_identity_split.json"
YUNET = ROOT / "models/face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models/face_recognition_sface_2021dec.onnx"
ARCFACE = ROOT / "models/arcface_w600k_r50.onnx"


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def score_distribution(pairs: list[dict]) -> dict:
    result = {}
    for pair_type in ("genuine", "impostor"):
        values = sorted(float(row["similarity"]) for row in pairs
                        if row["pair_type"] == pair_type and row["similarity"] is not None)
        result[pair_type] = {
            "count": len(values),
            "missing_count": sum(row["pair_type"] == pair_type and row["similarity"] is None
                                  for row in pairs),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p05": values[int((len(values) - 1) * 0.05)] if values else None,
            "p95": values[int((len(values) - 1) * 0.95)] if values else None,
            "minimum": values[0] if values else None,
            "maximum": values[-1] if values else None,
        }
    return result


def evaluate_model(name: str, model_path: Path, records: list[dict], identities: list[str]) -> dict:
    started = time.perf_counter()
    backend = OpenCVFeatureBackend(
        FaceDetectionConfig(YUNET, score_threshold=0.9, frame_interval_seconds=1.0),
        SFaceConfig(SFACE),
        recognition_model=name,
        arcface_model_path=model_path if name == "arcface" else None,
    )
    gallery_records = [r for r in records if r["split"] == "gallery"]
    query_records = [r for r in records if r["split"] == "query"]
    gallery_cache: dict[str, CachedMediaFeatures] = {}
    query_cache: dict[str, CachedMediaFeatures] = {}
    extraction_counts = Counter()
    feature_started = time.perf_counter()

    for index, record in enumerate(gallery_records, start=1):
        extracted = backend.extract({
            **record,
            "_experiment_mode": "plaintext_exploratory",
            "_gallery_representation_enabled": True,
            "_gallery_quality_filter": False,
            "_gallery_feature_aggregation": "none",
        })
        video_id = str(record["video_id"])
        gallery_cache[video_id] = CachedMediaFeatures(
            str(record["identity_id"]), video_id, THRESHOLD_SELECTION, "video",
            extracted.features, extracted.sampled_frame_count,
        )
        extraction_counts["gallery_features"] += len(extracted.features)
        extraction_counts["gallery_failures"] += extracted.feature_extraction_failure_count
        extraction_counts["gallery_detected_faces"] += extracted.detected_face_count
        if index % 10 == 0 or index == len(gallery_records):
            print(f"{name} gallery {index}/{len(gallery_records)}; elapsed={time.perf_counter() - feature_started:.1f}s",
                  flush=True)

    for index, record in enumerate(query_records, start=1):
        extracted = backend.extract({**record, "_experiment_mode": "plaintext_exploratory"})
        video_id = str(record["video_id"])
        query_cache[video_id] = CachedMediaFeatures(
            str(record["identity_id"]), video_id, THRESHOLD_SELECTION, "image",
            extracted.features, extracted.sampled_frame_count,
        )
        extraction_counts["query_features"] += len(extracted.features)
        extraction_counts["query_failures"] += extracted.feature_extraction_failure_count
        extraction_counts["query_detected_faces"] += extracted.detected_face_count
        if index % 25 == 0 or index == len(query_records):
            print(f"{name} query {index}/{len(query_records)}; elapsed={time.perf_counter() - feature_started:.1f}s",
                  flush=True)
    feature_seconds = time.perf_counter() - feature_started

    identity_split = {identity: THRESHOLD_SELECTION for identity in identities}
    scores, provenance, scoring_seconds, _ = _score_matrix(
        query_records, gallery_records, query_cache, gallery_cache, 16,
        expected_dimension=512 if name == "arcface" else 128,
        mode="plaintext_exploratory", split=THRESHOLD_SELECTION,
        score_aggregation="max",
    )
    pairs = build_similarity_pairs(
        records, identity_split, scores,
        detected_query_ids={qid for qid, cache in query_cache.items() if cache.features},
    )
    retrieval = retrieval_metrics(
        records, identity_split, pairs,
        detected_query_ids={qid for qid, cache in query_cache.items() if cache.features},
    )
    output = OUTPUT / name
    output.mkdir(mode=0o700)
    pair_path = output / "genuine_impostor_pairs.csv"
    with pair_path.open("x", encoding="utf-8", newline="") as stream:
        fields = ("query_id", "gallery_id", "similarity", "pair_type", "score_status")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in pairs)

    elapsed = time.perf_counter() - started
    return {
        "model": name,
        "retrieval": retrieval,
        "feature_availability": {
            "query_total": len(query_records),
            "query_with_features": sum(bool(c.features) for c in query_cache.values()),
            "query_without_features": sum(not c.features for c in query_cache.values()),
            "gallery_total": len(gallery_records),
            "gallery_with_features": sum(bool(c.features) for c in gallery_cache.values()),
            "gallery_without_features": sum(not c.features for c in gallery_cache.values()),
            **dict(extraction_counts),
        },
        "score_distribution": score_distribution(pairs),
        "timing_seconds": {
            "feature_extraction": feature_seconds,
            "similarity_scoring": scoring_seconds,
            "total_model_run": elapsed,
        },
        "pair_count": len(pairs),
        "scored_pair_count": sum(row["similarity"] is not None for row in pairs),
    }


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing exploratory output: {OUTPUT}")
    records, identity_split, manifest_json, split_json = _load_inputs(MANIFEST, SPLIT)
    eligible_ids = sorted(identity for identity, split in identity_split.items()
                          if split == THRESHOLD_SELECTION)
    if len(eligible_ids) < SAMPLE_SIZE:
        raise SystemExit("selection split contains fewer than 100 identities")
    identities = sorted(random.Random(SAMPLE_SEED).sample(eligible_ids, SAMPLE_SIZE))
    selected = set(identities)
    sample_records = [record for record in records if record["identity_id"] in selected]
    sample_split = {identity: THRESHOLD_SELECTION for identity in identities}
    validate_layout(sample_records, sample_split)
    OUTPUT.mkdir(parents=True, mode=0o700)
    write_json(OUTPUT / "identities.json", {
        "source_split": THRESHOLD_SELECTION,
        "sample_size": SAMPLE_SIZE,
        "seed": SAMPLE_SEED,
        "sampling": "random.Random(seed).sample(sorted(selection_identity_ids), 100); IDs sorted for storage",
        "identity_ids": identities,
    })
    write_json(OUTPUT / "config.json", {
        "purpose": "exploratory SFace vs ArcFace model comparison; non-formal, selection-only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "source_split": THRESHOLD_SELECTION,
        "identity_split_seed": split_json["seed"],
        "manifest_path": str(MANIFEST),
        "manifest_sha256": __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest(),
        "models": {
            "sface": {"name": "OpenCV SFace", "path": str(SFACE), "dimension": 128,
                      "preprocessing": "SFace alignCrop and feature API; L2 normalized"},
            "arcface": {"name": "InsightFace buffalo_l w600k_r50", "path": str(ARCFACE),
                        "dimension": 512, "input": "112x112; five-point ArcFace alignment; RGB (x-127.5)/127.5; L2 normalized"},
        },
        "gallery": {"frame_interval_seconds": 1.0, "quality_filter": False,
                    "feature_aggregation": "none", "face_policy": "all valid detections",
                    "pair_score": "maximum cosine similarity across Query and Gallery frame features"},
        "query": {"flow": "existing YuNet detection and model-specific embedding; no annotation-box selection"},
        "modes": ["plaintext"],
        "feature_reuse": "extract each model's Query/Gallery features once in memory and reuse for all pair scores; vectors are not exported",
        "formal_metrics": False,
        "source_manifest_json_sha256": __import__("hashlib").sha256(json.dumps(manifest_json, sort_keys=True).encode()).hexdigest(),
    })

    print(f"fixed sample saved: {OUTPUT} ({SAMPLE_SIZE} selection identities)", flush=True)
    results = {}
    for name, model_path in (("sface", SFACE), ("arcface", ARCFACE)):
        print(f"starting {name}", flush=True)
        results[name] = evaluate_model(name, model_path, sample_records, identities)
        print(f"finished {name}: {results[name]['timing_seconds']['total_model_run']:.1f}s", flush=True)

    sface = results["sface"]["retrieval"]
    arcface = results["arcface"]["retrieval"]
    comparison = {
        "top1_percentage_point_delta_arcface_minus_sface": 100 * (
            arcface["top1_accuracy"]["value"] - sface["top1_accuracy"]["value"]),
        "recall_at_5_percentage_point_delta_arcface_minus_sface": 100 * (
            arcface["recall_at_5"]["value"] - sface["recall_at_5"]["value"]),
        "both_metrics_improved": (
            arcface["top1_accuracy"]["value"] > sface["top1_accuracy"]["value"] and
            arcface["recall_at_5"]["value"] > sface["recall_at_5"]["value"]),
        "interpretation": "exploratory only; not a formal selection/test result",
    }
    write_json(OUTPUT / "metrics.json", {"models": results, "comparison": comparison})
    report = [
        "# YTF 100-Identity Model Exploration",
        "",
        f"Sample: {SAMPLE_SIZE} fixed threshold-selection identities; seed `{SAMPLE_SEED}`.",
        "Gallery quality filtering and feature aggregation were disabled. Only plaintext was evaluated.",
        "All scores and metrics are exploratory and non-formal.",
        "",
        "| Model | Top-1 | Recall@5 | Queries with features | Galleries with features | Feature time (s) | Total (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("sface", "arcface"):
        result = results[name]
        retrieval = result["retrieval"]
        availability = result["feature_availability"]
        timing = result["timing_seconds"]
        report.append(
            f"| {name} | {retrieval['top1_accuracy']['hits']}/100 ({retrieval['top1_accuracy']['value']:.2%}) "
            f"| {retrieval['recall_at_5']['hits']}/100 ({retrieval['recall_at_5']['value']:.2%}) "
            f"| {availability['query_with_features']}/100 | {availability['gallery_with_features']}/100 "
            f"| {timing['feature_extraction']:.1f} | {timing['total_model_run']:.1f} |"
        )
    report.extend([
        "",
        f"ArcFace minus SFace: Top-1 {comparison['top1_percentage_point_delta_arcface_minus_sface']:+.2f} pp; "
        f"Recall@5 {comparison['recall_at_5_percentage_point_delta_arcface_minus_sface']:+.2f} pp.",
        "",
        "| Model | Genuine scored (missing) | Genuine mean / median | Impostor scored (missing) | Impostor mean / median |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("sface", "arcface"):
        result = results[name]
        genuine = result["score_distribution"]["genuine"]
        impostor = result["score_distribution"]["impostor"]
        report.append(
            f"| {name} | {genuine['count']} ({genuine['missing_count']}) "
            f"| {genuine['mean']:.3f} / {genuine['median']:.3f} "
            f"| {impostor['count']} ({impostor['missing_count']}) "
            f"| {impostor['mean']:.3f} / {impostor['median']:.3f} |"
        )
    report.extend([
        "",
        "FAR/TAR are intentionally not estimated from this small exploratory sample.",
        "The sample can motivate a complete selection run only; it cannot establish formal performance or course-target attainment.",
    ])
    with (OUTPUT / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    print(json.dumps(comparison, ensure_ascii=False), flush=True)
    print(f"results: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
