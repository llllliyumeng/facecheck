"""Identity-disjoint Query-Gallery protocol and pair-level threshold evaluation.

This layer consumes already-computed query/gallery similarities. It does not run
YuNet, SFace, video decoding, or encryption and is separate from the business API.
"""
from __future__ import annotations

import hashlib
import math
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


THRESHOLD_SELECTION = "threshold_selection"
TEST = "test"
DEFAULT_SPLIT_SEED = "facecheck-ytf-qg-v1"


class QueryGalleryProtocolError(ValueError):
    """The identity split, media layout, or similarity matrix is invalid."""


def _get(record: Any, key: str) -> Any:
    return record[key] if isinstance(record, Mapping) else getattr(record, key)


def split_identities(identity_ids: Iterable[str], *, threshold_count: int = 502,
                     seed: str = DEFAULT_SPLIT_SEED) -> dict[str, str]:
    """Stable SHA-256 seeded identity partition; the same ID always gets one side."""
    ids = list(identity_ids)
    if not ids or any(not isinstance(x, str) or not x for x in ids):
        raise QueryGalleryProtocolError("identity_ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise QueryGalleryProtocolError("identity_ids contain duplicates")
    if not 0 < threshold_count < len(ids):
        raise QueryGalleryProtocolError("threshold_count must be between zero and identity count")
    ranked = sorted(ids, key=lambda value: (hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(), value))
    threshold_ids = set(ranked[:threshold_count])
    return {identity_id: THRESHOLD_SELECTION if identity_id in threshold_ids else TEST
            for identity_id in ids}


def validate_layout(records: Sequence[Any], identity_split: Mapping[str, str]) -> dict[str, Any]:
    """Require exactly one gallery and one query record per identity, with no group crossing."""
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, Any] = {}
    videos: set[str] = set()
    groups: dict[str, set[str]] = defaultdict(set)
    identities = set()
    for record in records:
        identity = _get(record, "identity_id")
        side = _get(record, "split")
        video = _get(record, "video_id")
        source_group = _get(record, "source_group_id")
        label = _get(record, "label")
        if side not in {"gallery", "query"}:
            raise QueryGalleryProtocolError(f"invalid media split: {side}")
        if not all(isinstance(x, str) and x for x in (identity, video, source_group, label)):
            raise QueryGalleryProtocolError("identity, video, source group and label are required")
        if video in videos:
            raise QueryGalleryProtocolError(f"duplicate video_id: {video}")
        videos.add(video); identities.add(identity); grouped[identity][side] += 1
        if identity in labels and labels[identity] != label:
            raise QueryGalleryProtocolError(f"inconsistent label for identity {identity}")
        labels[identity] = label
        if identity not in identity_split or identity_split[identity] not in {THRESHOLD_SELECTION, TEST}:
            raise QueryGalleryProtocolError(f"missing/invalid identity split for {identity}")
        groups[identity_split[identity]].add(source_group)
    if identities != set(identity_split):
        raise QueryGalleryProtocolError("identity split must cover exactly the manifest identities")
    if any(counts != Counter({"gallery": 1, "query": 1}) for counts in grouped.values()):
        raise QueryGalleryProtocolError("each identity must have exactly one gallery and one query record")
    if groups[THRESHOLD_SELECTION] & groups[TEST]:
        raise QueryGalleryProtocolError("source_group_id crosses threshold_selection and test")
    split_identity_counts = Counter(identity_split.values())
    return {"identity_count": len(identities), "video_count": len(videos),
            "gallery_count": len(identities), "query_count": len(identities),
            "threshold_selection_identity_count": split_identity_counts[THRESHOLD_SELECTION],
            "test_identity_count": split_identity_counts[TEST],
            "threshold_selection_source_group_count": len(groups[THRESHOLD_SELECTION]),
            "test_source_group_count": len(groups[TEST]),
            "identity_overlap": 0, "source_group_overlap": 0}


def build_similarity_pairs(records: Sequence[Any], identity_split: Mapping[str, str],
                           similarity_by_pair: Mapping[tuple[str, str], float | None],
                           detected_query_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Expand a complete within-partition score matrix into genuine/impostor rows.

    Matrix keys are (query video_id, gallery video_id). A complete row is retained
    for every within-partition query/gallery combination. Missing detections use a
    null score with an explicit status; no numeric score is fabricated.
    """
    validate_layout(records, identity_split)
    queries = {str(_get(r, "video_id")): r for r in records if _get(r, "split") == "query"}
    galleries = {str(_get(r, "video_id")): r for r in records if _get(r, "split") == "gallery"}
    detected = set(detected_query_ids)
    if not detected <= set(queries):
        raise QueryGalleryProtocolError("detected query IDs are not present in the manifest")
    expected: set[tuple[str, str]] = set()
    for qid, q in queries.items():
        qsplit = identity_split[_get(q, "identity_id")]
        for gid, g in galleries.items():
            if identity_split[_get(g, "identity_id")] == qsplit:
                expected.add((qid, gid))
    provided = set(similarity_by_pair)
    if provided != expected:
        cross = [(q, g) for q, g in provided if q in queries and g in galleries and
                 identity_split[_get(queries[q], "identity_id")] != identity_split[_get(galleries[g], "identity_id")]]
        if cross:
            raise QueryGalleryProtocolError("similarity matrix contains cross-partition query/gallery comparisons")
        missing = len(expected - provided); extra = len(provided - expected)
        raise QueryGalleryProtocolError(f"similarity matrix incomplete or unexpected: missing={missing}, extra={extra}")
    result = []
    for qid, gid in sorted(expected):
        raw_score = similarity_by_pair[(qid, gid)]
        if qid not in detected and raw_score is not None:
            raise QueryGalleryProtocolError("undetected query must not have numeric similarity scores")
        score = float(raw_score) if raw_score is not None else None
        if score is not None and (not math.isfinite(score) or score < -1.000001 or score > 1.000001):
            raise QueryGalleryProtocolError(f"invalid cosine similarity for {qid}/{gid}")
        q, g = queries[qid], galleries[gid]
        same = _get(q, "identity_id") == _get(g, "identity_id")
        result.append({"query_id": qid, "gallery_id": gid, "similarity": score,
                       "pair_type": "genuine" if same else "impostor",
                       "split": identity_split[_get(q, "identity_id")],
                       "score_status": ("query_face_not_detected" if qid not in detected else
                                        "scored" if score is not None else "gallery_face_not_detected")})
    return result


def _validate_pair_rows(records: Sequence[Any], identity_split: Mapping[str, str],
                        pairs: Sequence[Mapping[str, Any]], detected_query_ids: Iterable[str]) -> None:
    queries = {str(_get(r, "video_id")): r for r in records if _get(r, "split") == "query"}
    galleries = {str(_get(r, "video_id")): r for r in records if _get(r, "split") == "gallery"}
    detected = set(detected_query_ids)
    expected = {(qid, gid) for qid, q in queries.items() for gid, g in galleries.items()
                if identity_split[_get(q, "identity_id")] == identity_split[_get(g, "identity_id")]}
    keys = [(str(row.get("query_id")), str(row.get("gallery_id"))) for row in pairs]
    if len(keys) != len(set(keys)):
        raise QueryGalleryProtocolError("duplicate query/gallery pair rows")
    if set(keys) != expected:
        raise QueryGalleryProtocolError("pair rows are incomplete or include cross-partition comparisons")
    for row, (qid, gid) in zip(pairs, keys):
        q, g = queries[qid], galleries[gid]
        same = _get(q, "identity_id") == _get(g, "identity_id")
        expected_type = "genuine" if same else "impostor"
        split = identity_split[_get(q, "identity_id")]
        if row.get("split") != split or row.get("pair_type") != expected_type:
            raise QueryGalleryProtocolError("pair split/type does not match manifest identity labels")
        score = row.get("similarity")
        if score is not None and (not math.isfinite(float(score)) or not -1.000001 <= float(score) <= 1.000001):
            raise QueryGalleryProtocolError("pair similarity must be a finite cosine score or null")
        if qid not in detected and score is not None:
            raise QueryGalleryProtocolError("undetected query must not have numeric similarity scores")


def expected_pair_counts(identity_split: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Expected complete matrix/pair dimensions, assuming every query is detected."""
    result = {}
    for split in (THRESHOLD_SELECTION, TEST):
        ids = {identity for identity, side in identity_split.items() if side == split}
        q = len(ids)
        galleries = len(ids)
        genuine = len(ids)
        total = q * galleries
        result[split] = {"query_count": q, "gallery_count": galleries,
                         "matrix_score_count": total, "genuine_count": genuine,
                         "impostor_count": total - genuine}
    return result


def select_threshold_at_far(selection_pairs: Sequence[Mapping[str, Any]], *, target_far: float = 1e-3) -> dict[str, Any]:
    """Maximize selection TAR subject to empirical FAR <= target; score ties stay together.

    Acceptance is similarity >= threshold. Candidate thresholds are observed score
    values (all ties accepted together) plus a reject-all point above the maximum.
    Among equal TARs choose lower FAR, then the higher threshold.
    """
    if not 0 < target_far < 1:
        raise QueryGalleryProtocolError("target_far must be between zero and one")
    if any(row.get("split") != THRESHOLD_SELECTION for row in selection_pairs):
        raise QueryGalleryProtocolError("threshold selection may consume threshold_selection rows only")
    if any(row.get("pair_type") not in {"genuine", "impostor"} for row in selection_pairs):
        raise QueryGalleryProtocolError("pair_type must be genuine or impostor")
    genuine_rows = [r for r in selection_pairs if r.get("pair_type") == "genuine"]
    impostor_rows = [r for r in selection_pairs if r.get("pair_type") == "impostor"]
    genuine = [float(r["similarity"]) for r in genuine_rows if r.get("similarity") is not None]
    impostor = [float(r["similarity"]) for r in impostor_rows if r.get("similarity") is not None]
    counts = _pair_score_counts(genuine_rows, impostor_rows, target_far)
    if not genuine_rows or len(impostor) < math.ceil(1 / target_far) or not genuine:
        return {"status": "not_estimable", "reason": "insufficient effective genuine/impostor scores to resolve target FAR",
                **counts, "target_far": target_far,
                "missing_score_policy": "unscored pairs are rejected: they cannot be accepted, remain in trial denominators, and are excluded from score-resolution counts"}
    if any(not math.isfinite(x) for x in genuine + impostor):
        raise QueryGalleryProtocolError("pair scores must be finite")
    ordered = [(score, 1, 0) for score in genuine]
    ordered.extend((score, 0, 1) for score in impostor)
    ordered.sort(key=lambda item: item[0], reverse=True)
    candidates = [(0.0, 0.0, math.nextafter(ordered[0][0], math.inf), 0, 0)]
    ta = fa = index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        tied_genuine = tied_impostor = 0
        while index < len(ordered) and ordered[index][0] == threshold:
            tied_genuine += ordered[index][1]
            tied_impostor += ordered[index][2]
            index += 1
        ta += tied_genuine
        fa += tied_impostor
        tar, far = ta / len(genuine_rows), fa / len(impostor_rows)
        if far <= target_far:
            candidates.append((tar, far, threshold, ta, fa))
    # Highest TAR; among ties, lower FAR and then higher threshold.
    tar, far, threshold, ta, fa = max(candidates, key=lambda x: (x[0], -x[1], x[2]))
    return {"status": "estimated", "threshold": threshold, "comparison": ">=",
            "tie_policy": "all pairs equal to threshold are accepted together",
            "selection_rule": "maximize TAR subject to empirical FAR <= target; ties choose lower FAR, then higher threshold",
            "target_far": target_far, "split": THRESHOLD_SELECTION,
            **counts,
            "selection_tar": tar, "selection_far": far,
            "false_accepts": fa, "true_accepts": ta,
            "empirical_far_all_impostor_trials": far,
            "missing_score_policy": "unscored pairs are rejected: they cannot be accepted, remain in trial denominators, and are excluded from score-resolution counts",
            "uncertainty_note": "pair scores share query/gallery identities; pair-level binomial confidence intervals are not reported; use identity-cluster resampling for uncertainty estimates"}


def _pair_score_counts(genuine_rows: Sequence[Mapping[str, Any]],
                       impostor_rows: Sequence[Mapping[str, Any]], target_far: float) -> dict[str, Any]:
    genuine_total = len(genuine_rows)
    impostor_total = len(impostor_rows)
    genuine_scored = sum(row.get("similarity") is not None for row in genuine_rows)
    impostor_scored = sum(row.get("similarity") is not None for row in impostor_rows)
    genuine_unscored = genuine_total - genuine_scored
    impostor_unscored = impostor_total - impostor_scored
    total = genuine_total + impostor_total
    return {
        "genuine_count": genuine_total,
        "impostor_count": impostor_total,
        "scored_genuine_count": genuine_scored,
        "scored_impostor_count": impostor_scored,
        "unscored_genuine_count": genuine_unscored,
        "unscored_impostor_count": impostor_unscored,
        "unscored_pair_count": genuine_unscored + impostor_unscored,
        "unscored_pair_rate": ((genuine_unscored + impostor_unscored) / total) if total else None,
        "genuine_unscored_rate": (genuine_unscored / genuine_total) if genuine_total else None,
        "impostor_unscored_rate": (impostor_unscored / impostor_total) if impostor_total else None,
        "far_resolution": (1 / impostor_scored) if impostor_scored else None,
        "target_far_resolution_met": impostor_scored >= math.ceil(1 / target_far),
    }


def evaluate_test_threshold(test_pairs: Sequence[Mapping[str, Any]], threshold_result: Mapping[str, Any], *,
                            target_far: float = 1e-3) -> dict[str, Any]:
    """Evaluate the frozen selection threshold on test rows only; never recalibrates."""
    if not 0 < target_far < 1:
        raise QueryGalleryProtocolError("target_far must be between zero and one")
    if any(row.get("split") != TEST for row in test_pairs):
        raise QueryGalleryProtocolError("test evaluation may consume test rows only")
    if any(row.get("pair_type") not in {"genuine", "impostor"} for row in test_pairs):
        raise QueryGalleryProtocolError("pair_type must be genuine or impostor")
    genuine_rows = [r for r in test_pairs if r.get("pair_type") == "genuine"]
    impostor_rows = [r for r in test_pairs if r.get("pair_type") == "impostor"]
    genuine = [float(r["similarity"]) for r in genuine_rows if r.get("similarity") is not None]
    impostor = [float(r["similarity"]) for r in impostor_rows if r.get("similarity") is not None]
    counts = _pair_score_counts(genuine_rows, impostor_rows, target_far)
    if threshold_result.get("status") != "estimated" or threshold_result.get("split") != THRESHOLD_SELECTION:
        return {"status": "not_estimable", "reason": "a frozen threshold from threshold_selection is required",
                **counts, "target_far": target_far, "threshold": None,
                "tar": None, "far": None, "empirical_far_all_impostor_trials": None,
                "missing_score_policy": "unscored pairs are rejected: they cannot be accepted, remain in trial denominators, and are excluded from score-resolution counts"}
    threshold = float(threshold_result["threshold"])
    ta = sum(x >= threshold for x in genuine); fa = sum(x >= threshold for x in impostor)
    far = fa / len(impostor_rows) if impostor_rows else None
    if not genuine_rows or len(impostor) < math.ceil(1 / target_far) or not genuine:
        return {"status": "not_estimable", "reason": "test sample has insufficient effective impostor scores to resolve target FAR",
                "threshold": threshold, **counts,
                "true_accepts": ta, "false_accepts": fa,
                "empirical_far_all_impostor_trials": far,
                "tar": None, "far": far,
                "missing_score_policy": "unscored pairs are rejected: they cannot be accepted, remain in trial denominators, and are excluded from score-resolution counts"}
    tar = ta / len(genuine_rows)
    return {"status": "estimated", "threshold": threshold, "comparison": ">=",
            "target_far": target_far, "tar": tar, "far": far,
            "operating_point_met": far <= target_far,
            **counts,
            "true_accepts": ta, "false_accepts": fa,
            "empirical_far_all_impostor_trials": far,
            "missing_score_policy": "unscored pairs are rejected: they cannot be accepted, remain in trial denominators, and are excluded from score-resolution counts",
            "uncertainty_note": "pair scores share query/gallery identities; pair-level binomial confidence intervals are not reported; use identity-cluster resampling for uncertainty estimates",
            "threshold_source": THRESHOLD_SELECTION}


def retrieval_metrics(records: Sequence[Any], identity_split: Mapping[str, str],
                      pairs: Sequence[Mapping[str, Any]], detected_query_ids: Iterable[str]) -> dict[str, Any]:
    """Top-1/Recall/Precision and miss/error rates over all queries in their own split."""
    validate_layout(records, identity_split)
    queries = {str(_get(r, "video_id")): r for r in records if _get(r, "split") == "query"}
    detected = set(detected_query_ids)
    if not detected <= set(queries):
        raise QueryGalleryProtocolError("detected query IDs are not present in the manifest")
    _validate_pair_rows(records, identity_split, pairs, detected)
    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        if row.get("query_id") not in queries:
            raise QueryGalleryProtocolError("pair row references an unknown query")
        if row.get("split") != identity_split[_get(queries[row["query_id"]], "identity_id")]:
            raise QueryGalleryProtocolError("pair row split does not match query identity partition")
        by_query[str(row["query_id"])].append(row)
    total = len(queries)
    if not total:
        raise QueryGalleryProtocolError("no query records")
    hits = {k: 0 for k in (1, 5)}
    precision = {k: 0.0 for k in (1, 5)}
    misses = false_top1 = 0
    for qid, query in queries.items():
        all_rows = by_query.get(qid, [])
        rows = sorted((r for r in all_rows if r.get("similarity") is not None),
                      key=lambda r: (-float(r["similarity"]), str(r["gallery_id"])))
        if len(all_rows) != sum(1 for r in records if _get(r, "split") == "gallery" and
                                identity_split[_get(r, "identity_id")] == identity_split[_get(query, "identity_id")]):
            raise QueryGalleryProtocolError(f"query {qid} does not have a complete within-split gallery score list")
        if qid not in detected or not rows:
            misses += 1
            continue
        for k in (1, 5):
            ncorrect = sum(r.get("pair_type") == "genuine" for r in rows[:k])
            if ncorrect:
                hits[k] += 1
            precision[k] += ncorrect / k
        if rows[0].get("pair_type") != "genuine":
            false_top1 += 1
    return {"query_count": total, "detected_query_count": len(detected),
            "top1_accuracy": {"value": hits[1] / total, "hits": hits[1], "denominator": total},
            "recall_at_1": {"value": hits[1] / total, "hits": hits[1], "denominator": total},
            "recall_at_5": {"value": hits[5] / total, "hits": hits[5], "denominator": total},
            "precision_at_1": {"value": precision[1] / total, "denominator": total},
            "precision_at_5": {"value": precision[5] / total, "denominator": total},
            "miss_rate": {"value": misses / total, "misses": misses, "denominator": total},
            "false_match_rate": {"value": false_top1 / total, "false_top1": false_top1, "denominator": total}}


def compare_mode_scores(plaintext: Mapping[tuple[str, str], float | None],
                        encrypted_restore: Mapping[tuple[str, str], float | None], *,
                        tolerance: float = 1e-6) -> dict[str, Any]:
    """Check paired scores from plaintext and decrypt/verify/restore runs."""
    if tolerance < 0:
        raise QueryGalleryProtocolError("tolerance must be non-negative")
    if set(plaintext) != set(encrypted_restore):
        return {"consistent": False, "reason": "pair key sets differ",
                "plaintext_pair_count": len(plaintext), "encrypted_restore_pair_count": len(encrypted_restore)}
    if any((plaintext[key] is None) != (encrypted_restore[key] is None) for key in plaintext):
        return {"consistent": False, "reason": "score availability differs between modes",
                "plaintext_pair_count": len(plaintext), "encrypted_restore_pair_count": len(encrypted_restore)}
    values = [float(x) for x in list(plaintext.values()) + list(encrypted_restore.values()) if x is not None]
    if any(not math.isfinite(x) for x in values):
        raise QueryGalleryProtocolError("mode comparison scores must be finite")
    deltas = [abs(float(plaintext[key]) - float(encrypted_restore[key])) for key in plaintext if plaintext[key] is not None]
    maximum = max(deltas, default=0.0)
    return {"consistent": maximum <= tolerance, "pair_count": len(plaintext),
            "scored_pair_count": len(deltas),
            "tolerance": tolerance, "max_absolute_similarity_delta": maximum}


def evaluate_protocol(records: Sequence[Any], identity_split: Mapping[str, str],
                      plaintext_pairs: Sequence[Mapping[str, Any]],
                      encrypted_restore_pairs: Sequence[Mapping[str, Any]],
                      plaintext_detected_query_ids: Iterable[str],
                      encrypted_detected_query_ids: Iterable[str], *,
                      target_far: float = 1e-3,
                      stage_timings: Mapping[str, Mapping[str, float | None]] | None = None) -> dict[str, Any]:
    """Evaluate both storage modes under the same frozen identity-disjoint protocol.

    Threshold is selected only from plaintext threshold_selection genuine/impostor
    scores and is then applied unchanged to test scores from both modes.
    """
    validate_layout(records, identity_split)
    plain_detected = set(plaintext_detected_query_ids)
    encrypted_detected = set(encrypted_detected_query_ids)
    _validate_pair_rows(records, identity_split, plaintext_pairs, plain_detected)
    _validate_pair_rows(records, identity_split, encrypted_restore_pairs, encrypted_detected)
    plain_scores = {(str(r["query_id"]), str(r["gallery_id"])): (float(r["similarity"]) if r.get("similarity") is not None else None) for r in plaintext_pairs}
    encrypted_scores = {(str(r["query_id"]), str(r["gallery_id"])): (float(r["similarity"]) if r.get("similarity") is not None else None) for r in encrypted_restore_pairs}
    if len(plain_scores) != len(plaintext_pairs) or len(encrypted_scores) != len(encrypted_restore_pairs):
        raise QueryGalleryProtocolError("duplicate query/gallery pair rows")
    consistency = compare_mode_scores(plain_scores, encrypted_scores)
    threshold_rows = [r for r in plaintext_pairs if r.get("split") == THRESHOLD_SELECTION]
    plain_test = [r for r in plaintext_pairs if r.get("split") == TEST]
    encrypted_test = [r for r in encrypted_restore_pairs if r.get("split") == TEST]
    threshold = select_threshold_at_far(threshold_rows, target_far=target_far)
    plain_tar = evaluate_test_threshold(plain_test, threshold, target_far=target_far)
    encrypted_tar = evaluate_test_threshold(encrypted_test, threshold, target_far=target_far)
    timings_out = {}
    timing_fields = ("video_read_decode_extract_seconds", "frame_extraction_seconds",
                     "face_detection_seconds", "feature_extraction_seconds",
                     "similarity_calculation_seconds", "ranking_seconds",
                     "encryption_seconds", "restore_integrity_seconds",
                     "query_total_seconds", "experiment_total_seconds")
    for mode in ("plaintext", "encrypted_restore"):
        supplied = (stage_timings or {}).get(mode, {})
        cleaned = {}
        for field in timing_fields:
            value = supplied.get(field)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise QueryGalleryProtocolError(f"invalid {mode} timing for {field}")
            cleaned[field] = float(value) if value is not None else None
        timings_out[mode] = cleaned
    return {"protocol": "identity-disjoint-query-gallery", "target_far": target_far,
            "threshold_source": "plaintext threshold_selection pairs only",
            "threshold_selection": threshold,
            "plaintext": {"retrieval": retrieval_metrics(records, identity_split, plaintext_pairs,
                                                           plain_detected),
                          "tar_far_test": plain_tar},
            "encrypted_restore": {"retrieval": retrieval_metrics(records, identity_split, encrypted_restore_pairs,
                                                                   encrypted_detected),
                                  "tar_far_test": encrypted_tar},
            "mode_consistency": consistency, "stage_timings": timings_out,
            "cross_partition_comparisons": 0}


def export_similarity_pairs(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    """Write raw pair scores without replacing an existing file."""
    if path.exists():
        raise QueryGalleryProtocolError(f"refusing to overwrite pair export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("query_id", "gallery_id", "similarity", "pair_type", "split", "score_status")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in pairs:
            writer.writerow({field: row.get(field) for field in fields})
