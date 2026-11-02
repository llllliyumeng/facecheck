"""Offline, reproducible video face retrieval experiments."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .face_detection import FaceDetection, FaceDetectionConfig, load_yunet_detector
from .face_search import FaceSearchConfig, search_video
from .recognition import (
    DEFAULT_RECOGNITION_MODEL, DEFAULT_ARCFACE_PATH, DEFAULT_SFACE_PATH,
    RecognitionConfig, recognition_config, model_name, extract_recognition_embedding, DEFAULT_ARCFACE_THRESHOLD,
)
from .storage import restore_encrypted_video, store_encrypted_video
from .video import extract_frames


class ExperimentError(ValueError):
    """Invalid manifest, configuration, or experiment data."""


@dataclass(frozen=True)
class MediaRecord:
    identity_id: str
    video_id: str
    video_path: Path
    split: str
    source_group_id: str
    label: str
    media_type: str = "video"


@dataclass(frozen=True)
class ExperimentManifest:
    records: tuple[MediaRecord, ...]
    path: Path


def load_manifest(path: Path) -> ExperimentManifest:
    """Load and strictly validate {"records": [...]} JSON manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw["records"] if isinstance(raw, dict) else raw
        if not isinstance(rows, list) or not rows:
            raise ValueError("records must be a non-empty array")
        records = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each record must be an object")
            item = MediaRecord(
                identity_id=row["identity_id"], video_id=row["video_id"],
                video_path=(path.parent / row["video_path"]).resolve(),
                split=row["split"], source_group_id=row["source_group_id"],
                label=row["label"], media_type=row.get("media_type", "video"),
            )
            if not all(isinstance(x, str) and x.strip() for x in
                       (item.identity_id, item.video_id, item.source_group_id, item.label)):
                raise ValueError("identity/video/group/label values must be non-empty strings")
            if not re.fullmatch(r"id_[A-Za-z0-9_-]+", item.identity_id):
                raise ValueError("identity_id must use the de-identified id_... format")
            if item.split not in {"gallery", "query"}:
                raise ValueError("split must be gallery or query")
            if item.media_type not in {"video", "image"}:
                raise ValueError("media_type must be video or image")
            if not item.video_path.is_file():
                raise ValueError(f"media file does not exist: {row['video_path']}")
            records.append(item)
        validate_manifest(records)
        return ExperimentManifest(tuple(records), path.resolve())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"invalid experiment manifest: {exc}") from exc


def validate_manifest(records: list[MediaRecord] | tuple[MediaRecord, ...]) -> None:
    ids = [r.video_id for r in records]
    if len(ids) != len(set(ids)):
        raise ExperimentError("video_id values must be unique")
    labels: dict[str, str] = {}
    for r in records:
        if r.identity_id in labels and labels[r.identity_id] != r.label:
            raise ExperimentError(f"identity {r.identity_id} has inconsistent labels")
        labels[r.identity_id] = r.label
    gallery, queries = [r for r in records if r.split == "gallery"], [r for r in records if r.split == "query"]
    if not gallery or not queries:
        raise ExperimentError("manifest requires both gallery and query records")
    gallery_groups = {r.source_group_id for r in gallery}
    overlap = gallery_groups.intersection(r.source_group_id for r in queries)
    if overlap:
        raise ExperimentError(f"source_group_id leakage across gallery/query: {sorted(overlap)}")
    if not {r.identity_id for r in gallery}.intersection(r.identity_id for r in queries):
        raise ExperimentError("gallery and query must share at least one identity")


def aggregate_identity_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest frame for each identity/video pair, then rank identities."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (str(row["candidate_identity_id"]), str(row["candidate_video_id"]))
        old = best.get(key)
        if old is None or float(row["similarity"]) > float(old["similarity"]):
            best[key] = dict(row)
    result = sorted(best.values(), key=lambda x: (-float(x["similarity"]), x["candidate_identity_id"], x["candidate_video_id"]))
    for rank, row in enumerate(result, 1):
        row["identity_rank"] = rank
    return result


def _not_estimable(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "not_estimable", "reason": reason}


def compute_metrics(queries: list[dict[str, Any]], identity_rows: list[dict[str, Any]],
                    pair_rows: list[dict[str, Any]], *, threshold: Optional[float] = None,
                    threshold_selection: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    total = len(queries)
    if not total:
        return {k: _not_estimable("no query records") for k in ("top1_hit_rate", "recall_at_1", "recall_at_5", "precision_at_1", "precision_at_5", "miss_rate", "false_match_rate", "tar_at_far_1e-3")}
    hit1 = sum(any(x["query_id"] == q["query_id"] and x["identity_rank"] == 1 and x["is_correct_match"] for x in identity_rows) for q in queries)
    hits = {k: sum(any(x["query_id"] == q["query_id"] and x["identity_rank"] <= k and x["is_correct_match"] for x in identity_rows) for q in queries) for k in (1, 5)}
    p_at = {k: sum(sum(1 for x in identity_rows if x["query_id"] == q["query_id"] and x["identity_rank"] <= k and x["is_correct_match"]) / k for q in queries) / total for k in (1, 5)}
    misses = sum(not q["query_detected"] or not any(x["query_id"] == q["query_id"] for x in identity_rows) for q in queries)
    false = sum(any(x["query_id"] == q["query_id"] and x["identity_rank"] == 1 and x["is_false_match"] for x in identity_rows) for q in queries)
    metrics: dict[str, Any] = {
        "query_count": total,
        "top1_hit_rate": {"value": hit1 / total, "numerator": hit1, "denominator": total, "status": "estimated"},
        "recall_at_1": {"value": hits[1] / total, "numerator": hits[1], "denominator": total, "status": "estimated"},
        "recall_at_5": {"value": hits[5] / total, "numerator": hits[5], "denominator": total, "status": "estimated"},
        "precision_at_1": {"value": p_at[1], "denominator": total, "status": "estimated", "note": "per-query precision averaged; missing slots count as non-relevant"},
        "precision_at_5": {"value": p_at[5], "denominator": total, "status": "estimated", "note": "per-query precision averaged; missing slots count as non-relevant"},
        "miss_rate": {"value": misses / total, "numerator": misses, "denominator": total, "status": "estimated"},
        "false_match_rate": {"value": false / total, "numerator": false, "denominator": total, "status": "estimated", "note": "queries whose top identity is wrong divided by all queries"},
    }
    positives = [x for x in pair_rows if x["pair_type"] == "genuine"]
    negatives = [x for x in pair_rows if x["pair_type"] == "impostor"]
    if threshold is None:
        metrics["tar_at_far_1e-3"] = _not_estimable("no independently selected threshold supplied")
    elif len(positives) < 1 or len(negatives) < 1000:
        metrics["tar_at_far_1e-3"] = _not_estimable(f"requires >=1 genuine and >=1000 independent impostor test pairs; got {len(positives)} and {len(negatives)}")
    elif not threshold_selection or threshold_selection.get("split") != "threshold_selection":
        metrics["tar_at_far_1e-3"] = _not_estimable("threshold provenance must identify a separate threshold_selection split")
    else:
        fp = sum(float(x["similarity"]) >= threshold for x in negatives)
        tp = sum(float(x["similarity"]) >= threshold for x in positives)
        far, tar = fp / len(negatives), tp / len(positives)
        metrics["tar_at_far_1e-3"] = ({"value": tar, "far": far, "threshold": threshold, "genuine_count": len(positives), "impostor_count": len(negatives), "status": "estimated"} if far <= 1e-3 else _not_estimable(f"supplied threshold yields FAR={far}, above 1e-3"))
    return metrics


def build_pair_scores(queries: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"query_id": q["query_id"], "query_identity_id": q["query_identity_id"],
             "candidate_identity_id": c["candidate_identity_id"], "candidate_video_id": c["candidate_video_id"],
             "similarity": c["similarity"], "pair_type": "genuine" if q["query_identity_id"] == c["candidate_identity_id"] else "impostor",
             "split": "test"} for q in queries for c in candidates if c.get("query_id") == q["query_id"]]


def select_threshold_at_far(selection_pairs: list[dict[str, Any]], far_limit: float = 1e-3) -> dict[str, Any]:
    """Select the lowest observed score threshold whose empirical calibration FAR is <= limit."""
    genuine = [float(x["similarity"]) for x in selection_pairs if x.get("pair_type") == "genuine"]
    impostor = [float(x["similarity"]) for x in selection_pairs if x.get("pair_type") == "impostor"]
    if not genuine or len(impostor) < math.ceil(1 / far_limit):
        return {"status": "not_estimable", "reason": f"threshold selection requires genuine pairs and at least {math.ceil(1 / far_limit)} impostor pairs", "genuine_count": len(genuine), "impostor_count": len(impostor)}
    scores = sorted(set(genuine + impostor), reverse=True)
    valid = [t for t in scores if sum(s >= t for s in impostor) / len(impostor) <= far_limit]
    if not valid:
        return {"status": "not_estimable", "reason": "no observed threshold meets requested FAR", "genuine_count": len(genuine), "impostor_count": len(impostor)}
    threshold = min(valid)
    return {"status": "estimated", "threshold": threshold, "calibration_far": sum(s >= threshold for s in impostor) / len(impostor),
            "calibration_tar": sum(s >= threshold for s in genuine) / len(genuine),
            "genuine_count": len(genuine), "impostor_count": len(impostor), "split": "threshold_selection"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _query_vectors(record: MediaRecord, detection: FaceDetectionConfig, embedding: RecognitionConfig) -> tuple[list[tuple[np.ndarray, int, float]], int, float, float]:
    started = time.perf_counter()
    if record.media_type == "image":
        image = cv2.imread(str(record.video_path))
        if image is None:
            raise ExperimentError(f"cannot decode query image {record.video_id}")
        frames = [(0, 0.0, image)]
    else:
        frames = [(f.index, f.timestamp_seconds, f.image) for f in extract_frames(record.video_path, detection.frame_interval_seconds)]
    decode = time.perf_counter() - started
    detector = load_yunet_detector(detection)
    found: list[tuple[np.ndarray, int, float]] = []
    detect_time = embed_time = 0.0
    for index, timestamp, image in frames:
        h, w = image.shape[:2]
        detector.setInputSize((w, h))
        t = time.perf_counter(); _, faces = detector.detect(image); detect_time += time.perf_counter() - t
        if faces is None:
            continue
        for raw in faces:
            a = np.asarray(raw, dtype=np.float32)
            if a.size < 15:
                continue
            d = FaceDetection(index, timestamp, w, h, float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[14]), tuple((float(a[i]), float(a[i+1])) for i in range(4, 14, 2)))
            t = time.perf_counter(); vector = extract_recognition_embedding(image, d, embedding).vector; embed_time += time.perf_counter() - t
            found.append((vector, index, timestamp))
    return found, decode, detect_time, embed_time


def run_experiment(manifest: ExperimentManifest, output_dir: Path, *, mode: str,
                   detection: FaceDetectionConfig, embedding: RecognitionConfig, top_k: int = 10,
                   min_similarity: Optional[float] = None, encryption_key: Optional[bytes] = None,
                   mac_key: Optional[bytes] = None, threshold: Optional[float] = None,
                   threshold_selection: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Run image/video queries against each gallery video and export raw records."""
    if mode not in {"plaintext", "encrypted_restore"}:
        raise ExperimentError("mode must be plaintext or encrypted_restore")
    if mode == "encrypted_restore" and (not encryption_key or not mac_key):
        raise ExperimentError("encrypted_restore requires encryption_key and mac_key")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExperimentError(f"output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery = [r for r in manifest.records if r.split == "gallery"]
    queries = [r for r in manifest.records if r.split == "query"]
    cfg = FaceSearchConfig(detection, embedding, top_k=top_k, min_similarity=min_similarity)
    query_rows: list[dict[str, Any]] = []; candidate_rows: list[dict[str, Any]] = []; timings: list[dict[str, Any]] = []
    whole_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="facecheck-experiment-") as temp:
        root = Path(temp)
        for q in queries:
            q_start = time.perf_counter()
            vectors, qdecode, qdetect, qembed = _query_vectors(q, detection, embedding)
            qrow = {"query_id": q.video_id, "query_identity_id": q.identity_id, "query_label": q.label,
                    "query_source_group_id": q.source_group_id, "query_detected": bool(vectors), "query_face_count": len(vectors),
                    "is_miss": not bool(vectors), "candidate_face_count": 0,
                    "mode": mode, "status": "ok" if vectors else "query_face_not_detected"}
            query_rows.append(qrow)
            for g in gallery:
                source_hash = _sha256(g.video_path)
                target = g.video_path
                enc_time = restore_time = None
                if mode == "encrypted_restore":
                    stats: dict[str, float] = {}; enc_started = time.perf_counter()
                    stored = store_encrypted_video(g.video_path, root / "encrypted", encryption_key, mac_key,
                                                   chunk_size=1024 * 1024, key_version="experiment-v1",
                                                   video_id=f"v{len(timings):08d}", stats=stats)
                    enc_time = time.perf_counter() - enc_started
                    restore_stats: dict[str, float] = {}; restore_started = time.perf_counter()
                    target = root / f"restored-{len(timings):08d}.video"
                    restore_encrypted_video(stored.directory, target, encryption_key, mac_key, stats=restore_stats)
                    restore_time = time.perf_counter() - restore_started
                    restored_hash = _sha256(target)
                    if restored_hash != source_hash:
                        raise ExperimentError(f"restored SHA-256 mismatch for {g.video_id}")
                else:
                    restored_hash = None
                target_vectors: dict[str, np.ndarray] = {}
                per_target_times: dict[str, float] = {}
                search_totals: dict[str, float] = {}
                candidate_face_count = 0
                for qidx, (query_vector, qframe, qts) in enumerate(vectors):
                    stats_search: dict[str, float] = {}; st = time.perf_counter()
                    matches = search_video(target, query_vector, cfg, stats=stats_search)
                    per_target_times[str(qidx)] = time.perf_counter() - st
                    candidate_face_count = max(candidate_face_count, int(stats_search.get("feature_count", 0)))
                    for key, value in stats_search.items():
                        if key != "feature_count":
                            search_totals[key] = search_totals.get(key, 0.0) + float(value)
                    for match in matches:
                        candidate_rows.append({"query_id": q.video_id, "query_identity_id": q.identity_id,
                            "candidate_identity_id": g.identity_id, "candidate_video_id": g.video_id,
                            "candidate_label": g.label, "frame_index": match.frame_index,
                            "timestamp_seconds": match.timestamp_seconds, "similarity": match.similarity,
                            "candidate_face_count": candidate_face_count,
                            "query_frame_index": qframe, "query_timestamp_seconds": qts,
                            "is_correct_match": g.identity_id == q.identity_id,
                            "is_false_match": g.identity_id != q.identity_id, "is_miss": False,
                            "mode": mode, "source_sha256": source_hash, "restored_sha256": restored_hash})
                    timings.append({"query_id": q.video_id, "candidate_video_id": g.video_id, "mode": mode,
                        "query_decode_seconds": qdecode, "query_detection_seconds": qdetect,
                        "query_embedding_seconds": qembed,
                        "video_read_decode_extract_seconds": search_totals.get("video_decode_extract_seconds"),
                        "face_detection_seconds": search_totals.get("face_detection_seconds"),
                        "feature_extraction_seconds": search_totals.get("feature_extraction_seconds", search_totals.get("sface_embedding_seconds")),
                        "similarity_calculation_and_sort_seconds": search_totals.get("similarity_sort_seconds"),
                        "encryption_seconds": enc_time, "restore_integrity_seconds": restore_time,
                        "query_total_seconds": time.perf_counter() - q_start,
                        "query_video_candidate_search_seconds": sum(per_target_times.values())})
                qrow["candidate_face_count"] += candidate_face_count
        duration = time.perf_counter() - whole_started
    # Raw frame ranks are global per query. Identity aggregation de-duplicates video frames.
    for q in queries:
        rows = [x for x in candidate_rows if x["query_id"] == q.video_id]
        rows.sort(key=lambda x: (-x["similarity"], x["candidate_video_id"], x["frame_index"]))
        for rank, row in enumerate(rows, 1): row["frame_rank"] = rank
        aggregate = aggregate_identity_candidates(rows)
        for row in aggregate:
            row["query_id"] = q.video_id
            row["is_correct_match"] = row["candidate_identity_id"] == q.identity_id
            row["is_false_match"] = not row["is_correct_match"]
            row["is_miss"] = False
    # Add explicit query-level miss marker; no fabricated candidate row.
    identity_rows = []
    for q in queries:
        identity_rows.extend(aggregate_identity_candidates([x for x in candidate_rows if x["query_id"] == q.video_id]))
        for row in identity_rows:
            if row.get("query_id") == q.video_id:
                row["is_correct_match"] = row["candidate_identity_id"] == q.identity_id
                row["is_false_match"] = not row["is_correct_match"]
    pair_rows = build_pair_scores(query_rows, identity_rows)
    selection_rows: list[dict[str, Any]] = []
    if threshold_selection and isinstance(threshold_selection.get("pairs"), list):
        selection_rows = [dict(x, split="threshold_selection") for x in threshold_selection["pairs"]]
        selected = select_threshold_at_far(selection_rows)
        if selected["status"] == "estimated":
            threshold = float(selected["threshold"])
            threshold_selection = selected
    metrics = compute_metrics(query_rows, identity_rows, pair_rows, threshold=threshold, threshold_selection=threshold_selection)
    metrics["protocol"] = {"mode": mode, "frame_level_rank_is_diagnostic": True,
        "identity_level_aggregation": "maximum frame similarity per identity/video, globally ranked",
        "top_k_scope": "metric-level identity ranks", "total_experiment_seconds": duration}
    config = {"mode": mode, "manifest": str(manifest.path), "top_k": top_k,
        "min_similarity": min_similarity, "frame_interval_seconds": detection.frame_interval_seconds,
        "yunet_model": str(detection.model_path), "yunet_score_threshold": detection.score_threshold,
        "yunet_nms_threshold": detection.nms_threshold, "recognition_model": model_name(embedding),
        "recognition_model_path": str(embedding.model_path),
        "embedding_dimension": embedding.expected_dimension, "threshold": threshold,
        "threshold_selection": threshold_selection}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "query_results.csv", query_rows)
    _write_csv(output_dir / "candidate_results.csv", candidate_rows)
    _write_csv(output_dir / "identity_results.csv", identity_rows)
    _write_csv(output_dir / "genuine_impostor_pairs.csv", pair_rows)
    _write_csv(output_dir / "threshold_selection_pairs.csv", selection_rows)
    _write_csv(output_dir / "timing.csv", timings)
    (output_dir / "report.txt").write_text(
        f"Facecheck offline experiment\nMode: {mode}\nQueries: {len(query_rows)}\nGallery videos: {len(gallery)}\n"
        f"Result: {json.dumps(metrics, ensure_ascii=False, sort_keys=True)}\n"
        "encrypted_restore means decrypt and verify, then search restored plaintext; it is not ciphertext-domain retrieval.\n",
        encoding="utf-8")
    return metrics


def run_comparison(manifest: ExperimentManifest, output_dir: Path, *,
                   detection: FaceDetectionConfig, embedding: RecognitionConfig,
                   top_k: int = 10, min_similarity: Optional[float] = None,
                   encryption_key: Optional[bytes] = None, mac_key: Optional[bytes] = None,
                   threshold: Optional[float] = None,
                   threshold_selection: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Run paired modes with one shared immutable parameter set and input manifest."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExperimentError(f"output directory is not empty; refusing to overwrite: {output_dir}")
    common = dict(detection=detection, embedding=embedding, top_k=top_k,
                  min_similarity=min_similarity, threshold=threshold,
                  threshold_selection=threshold_selection)
    plaintext = run_experiment(manifest, output_dir / "plaintext", mode="plaintext", **common)
    encrypted = run_experiment(manifest, output_dir / "encrypted_restore",
                               mode="encrypted_restore", encryption_key=encryption_key,
                               mac_key=mac_key, **common)
    result = {"shared_manifest": str(manifest.path), "shared_parameters": {
        "frame_interval_seconds": detection.frame_interval_seconds,
        "yunet_model": str(detection.model_path), "yunet_score_threshold": detection.score_threshold,
        "yunet_nms_threshold": detection.nms_threshold, "recognition_model": model_name(embedding),
        "recognition_model_path": str(embedding.model_path),
        "top_k": top_k, "min_similarity": min_similarity, "threshold": threshold},
        "plaintext": plaintext, "encrypted_restore": encrypted}
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Facecheck offline video retrieval experiment")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("plaintext", "encrypted_restore"), required=True)
    parser.add_argument("--yunet-model", type=Path, required=True)
    parser.add_argument("--recognition-model", choices=("arcface", "sface"), default=DEFAULT_RECOGNITION_MODEL)
    parser.add_argument("--arcface-model", type=Path, default=DEFAULT_ARCFACE_PATH)
    parser.add_argument("--sface-model", type=Path, default=DEFAULT_SFACE_PATH)
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.threshold is None and args.recognition_model == "arcface":
        args.threshold = DEFAULT_ARCFACE_THRESHOLD
    import os
    manifest = load_manifest(args.manifest)
    metrics = run_experiment(
        manifest, args.output, mode=args.mode,
        detection=FaceDetectionConfig(args.yunet_model, frame_interval_seconds=args.frame_interval),
        embedding=recognition_config(args.recognition_model, arcface_path=args.arcface_model,
                                     sface_path=args.sface_model), top_k=args.top_k,
        min_similarity=args.min_similarity,
        encryption_key=os.getenv("FACECHECK_SM4_KEY", "").encode() or None,
        mac_key=os.getenv("FACECHECK_MAC_KEY", "").encode() or None,
        threshold=args.threshold,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
