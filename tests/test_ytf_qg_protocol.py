import csv
from collections import Counter

import pytest

from facecheck.ytf_qg_protocol import (
    QueryGalleryProtocolError, THRESHOLD_SELECTION, TEST,
    build_similarity_pairs, compare_mode_scores, evaluate_protocol,
    evaluate_test_threshold, expected_pair_counts, export_similarity_pairs,
    retrieval_metrics, select_threshold_at_far, split_identities, validate_layout,
)


def _fixture():
    records = []
    for identity in ("id_a", "id_b", "id_c", "id_d"):
        records.extend([
            {"identity_id": identity, "video_id": f"g_{identity}", "split": "gallery",
             "source_group_id": f"src_g_{identity}", "label": identity},
            {"identity_id": identity, "video_id": f"q_{identity}", "split": "query",
             "source_group_id": f"src_q_{identity}", "label": identity},
        ])
    assignment = {"id_a": THRESHOLD_SELECTION, "id_b": THRESHOLD_SELECTION,
                  "id_c": TEST, "id_d": TEST}
    scores = {
        ("q_id_a", "g_id_a"): .90, ("q_id_a", "g_id_b"): .60,
        ("q_id_b", "g_id_a"): .50, ("q_id_b", "g_id_b"): .85,
        ("q_id_c", "g_id_c"): .95, ("q_id_c", "g_id_d"): .20,
        ("q_id_d", "g_id_c"): .70, ("q_id_d", "g_id_d"): .90,
    }
    return records, assignment, scores


def test_reproducible_identity_split_and_expected_502_matrix_counts():
    identities = [f"id_{i:04d}" for i in range(1004)]
    one = split_identities(identities)
    two = split_identities(identities)
    assert one == two
    assert Counter(one.values()) == {THRESHOLD_SELECTION: 502, TEST: 502}
    counts = expected_pair_counts(one)
    for side in (THRESHOLD_SELECTION, TEST):
        assert counts[side] == {"query_count": 502, "gallery_count": 502,
                                "matrix_score_count": 252004, "genuine_count": 502,
                                "impostor_count": 251502}


def test_pairs_are_complete_labeled_and_partition_limited():
    records, assignment, scores = _fixture()
    pairs = build_similarity_pairs(records, assignment, scores, {f"q_{x}" for x in assignment})
    assert len(pairs) == 8
    assert sum(p["pair_type"] == "genuine" for p in pairs) == 4
    assert sum(p["pair_type"] == "impostor" for p in pairs) == 4
    assert all(p["split"] == assignment[p["query_id"][2:]] for p in pairs)
    assert validate_layout(records, assignment)["identity_overlap"] == 0
    cross = dict(scores); cross[("q_id_a", "g_id_c")] = .1
    with pytest.raises(QueryGalleryProtocolError, match="cross-partition"):
        build_similarity_pairs(records, assignment, cross, {f"q_{x}" for x in assignment})
    with pytest.raises(QueryGalleryProtocolError, match="incomplete"):
        build_similarity_pairs(records, assignment, dict(list(scores.items())[:-1]), {f"q_{x}" for x in assignment})


def test_unscored_gallery_detection_is_null_and_counts_as_rejected_pair():
    records, assignment, scores = _fixture()
    scores[("q_id_a", "g_id_b")] = None
    pairs = build_similarity_pairs(records, assignment, scores, {f"q_{x}" for x in assignment})
    row = next(p for p in pairs if p["query_id"] == "q_id_a" and p["gallery_id"] == "g_id_b")
    assert row["similarity"] is None
    assert row["score_status"] == "gallery_face_not_detected"
    assert len([p for p in pairs if p["split"] == THRESHOLD_SELECTION]) == 4
    assert compare_mode_scores({("q", "g"): None}, {("q", "g"): None})["consistent"] is True


def test_undetected_query_is_miss_and_has_no_pairs():
    records, assignment, scores = _fixture()
    detected = {"q_id_b", "q_id_c", "q_id_d"}
    unavailable = {k: (v if k[0] in detected else None) for k, v in scores.items()}
    pairs = build_similarity_pairs(records, assignment, unavailable, detected)
    metrics = retrieval_metrics(records, assignment, pairs, detected)
    assert metrics["query_count"] == 4
    assert metrics["miss_rate"]["value"] == .25
    assert metrics["miss_rate"]["misses"] == 1
    assert len(pairs) == 8
    assert all(p["similarity"] is None for p in pairs if p["query_id"] == "q_id_a")
    with pytest.raises(QueryGalleryProtocolError, match="undetected query"):
        build_similarity_pairs(records, assignment, scores, detected)


def test_threshold_uses_selection_only_ties_are_inclusive_and_small_sample_is_not_estimable():
    selection = ([{"split": THRESHOLD_SELECTION, "pair_type": "genuine", "similarity": .8}] +
                 [{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": s}
                  for s in (.8, .7, .6, .5, .4)])
    chosen = select_threshold_at_far(selection, target_far=.2)
    assert chosen["status"] == "estimated"
    assert chosen["threshold"] == .8
    assert chosen["comparison"] == ">="
    assert chosen["selection_far"] == pytest.approx(.2)
    assert chosen["far_resolution"] == pytest.approx(.2)
    with pytest.raises(QueryGalleryProtocolError, match="threshold_selection rows only"):
        select_threshold_at_far(selection + [{"split": TEST, "pair_type": "impostor", "similarity": .1}], target_far=.2)
    insufficient = selection[:1] + [{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": .1} for _ in range(999)]
    result = select_threshold_at_far(insufficient)
    assert result["status"] == "not_estimable"
    assert result["scored_impostor_count"] == 999
    assert result["far_resolution"] == pytest.approx(1 / 999)


def test_fixed_test_threshold_reports_far_tar_without_recalibration():
    records, assignment, scores = _fixture()
    pairs = build_similarity_pairs(records, assignment, scores, {f"q_{x}" for x in assignment})
    threshold_rows = [p for p in pairs if p["split"] == THRESHOLD_SELECTION]
    test_rows = [p for p in pairs if p["split"] == TEST]
    # Relax the target only for this tiny synthetic fixture. Test scores do not change the threshold.
    calibrated = select_threshold_at_far(threshold_rows, target_far=.5)
    before = calibrated["threshold"]
    test = evaluate_test_threshold(test_rows, calibrated, target_far=.5)
    assert calibrated["threshold"] == before
    assert test["threshold"] == before
    assert test["tar"] == 1.0 and test["far"] == 0.0
    assert test["operating_point_met"] is True
    with pytest.raises(QueryGalleryProtocolError, match="test rows only"):
        evaluate_test_threshold(threshold_rows, calibrated, target_far=.5)
    limited_test = ([{"split": TEST, "pair_type": "genuine", "similarity": .9}] +
                    [{"split": TEST, "pair_type": "impostor", "similarity": .1} for _ in range(999)])
    assert evaluate_test_threshold(limited_test, {"status": "estimated", "threshold": .8,
                                                   "split": THRESHOLD_SELECTION})["status"] == "not_estimable"


def test_tar_far_counts_scored_and_unscored_trials_consistently():
    selection = [
        {"split": THRESHOLD_SELECTION, "pair_type": "genuine", "similarity": .9},
        {"split": THRESHOLD_SELECTION, "pair_type": "genuine", "similarity": None},
        *[{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": .1}
          for _ in range(1000)],
        *[{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": None}
          for _ in range(2)],
    ]
    chosen = select_threshold_at_far(selection)
    assert chosen["status"] == "estimated"
    assert chosen["genuine_count"] == 2
    assert chosen["scored_genuine_count"] == 1
    assert chosen["unscored_genuine_count"] == 1
    assert chosen["impostor_count"] == 1002
    assert chosen["scored_impostor_count"] == 1000
    assert chosen["unscored_impostor_count"] == 2
    assert chosen["unscored_pair_count"] == 3
    assert chosen["unscored_pair_rate"] == pytest.approx(3 / 1004)
    assert chosen["far_resolution"] == pytest.approx(1 / 1000)
    assert chosen["selection_tar"] == pytest.approx(.5)
    assert chosen["selection_far"] == pytest.approx(0)
    assert "unscored pairs are rejected" in chosen["missing_score_policy"]

    test = [
        {"split": TEST, "pair_type": "genuine", "similarity": .95},
        {"split": TEST, "pair_type": "genuine", "similarity": None},
        {"split": TEST, "pair_type": "impostor", "similarity": .95},
        *[{"split": TEST, "pair_type": "impostor", "similarity": .1} for _ in range(999)],
        *[{"split": TEST, "pair_type": "impostor", "similarity": None} for _ in range(2)],
    ]
    evaluated = evaluate_test_threshold(test, chosen)
    assert evaluated["status"] == "estimated"
    assert evaluated["tar"] == pytest.approx(.5)
    assert evaluated["far"] == pytest.approx(1 / 1002)
    assert evaluated["empirical_far_all_impostor_trials"] == pytest.approx(1 / 1002)
    assert evaluated["far_resolution"] == pytest.approx(1 / 1000)
    assert evaluated["unscored_pair_count"] == 3


def test_low_effective_impostor_count_is_not_estimable_even_when_total_trials_are_large():
    rows = ([{"split": THRESHOLD_SELECTION, "pair_type": "genuine", "similarity": .9}] +
            [{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": .1} for _ in range(999)] +
            [{"split": THRESHOLD_SELECTION, "pair_type": "impostor", "similarity": None} for _ in range(1001)])
    selection = select_threshold_at_far(rows)
    assert selection["status"] == "not_estimable"
    assert selection["impostor_count"] == 2000
    assert selection["scored_impostor_count"] == 999
    assert selection["far_resolution"] == pytest.approx(1 / 999)
    frozen = {"status": "estimated", "split": THRESHOLD_SELECTION, "threshold": .8}
    test_rows = [dict(row, split=TEST) for row in rows]
    evaluated = evaluate_test_threshold(test_rows, frozen)
    assert evaluated["status"] == "not_estimable"
    assert evaluated["tar"] is None
    assert evaluated["impostor_count"] == 2000
    assert evaluated["scored_impostor_count"] == 999
    assert evaluated["far_resolution"] == pytest.approx(1 / 999)


def test_test_group_counts_are_preserved_when_selection_threshold_is_not_estimable():
    rows = ([{"split": TEST, "pair_type": "genuine", "similarity": .8}] +
            [{"split": TEST, "pair_type": "impostor", "similarity": .2} for _ in range(9)] +
            [{"split": TEST, "pair_type": "impostor", "similarity": None}])
    result = evaluate_test_threshold(rows, {"status": "not_estimable"})
    assert result["status"] == "not_estimable"
    assert result["genuine_count"] == 1
    assert result["impostor_count"] == 10
    assert result["scored_impostor_count"] == 9
    assert result["unscored_impostor_count"] == 1
    assert result["far_resolution"] == pytest.approx(1 / 9)
    assert result["tar"] is None


def test_combined_protocol_mode_consistency_and_null_unmeasured_timings():
    records, assignment, scores = _fixture()
    detected = {f"q_{x}" for x in assignment}
    pairs = build_similarity_pairs(records, assignment, scores, detected)
    result = evaluate_protocol(records, assignment, pairs, pairs, detected, detected,
                               target_far=.5,
                               stage_timings={"plaintext": {"detection_seconds": .2}})
    assert result["threshold_source"] == "plaintext threshold_selection pairs only"
    assert result["mode_consistency"]["consistent"] is True
    assert result["plaintext"]["retrieval"]["top1_accuracy"]["value"] == 1.0
    assert result["encrypted_restore"]["tar_far_test"]["threshold"] == result["threshold_selection"]["threshold"]
    assert result["stage_timings"]["plaintext"]["face_detection_seconds"] is None
    assert result["stage_timings"]["encrypted_restore"]["encryption_seconds"] is None
    altered = dict(scores); altered[("q_id_a", "g_id_a")] += .01
    assert compare_mode_scores(scores, altered)["consistent"] is False


def test_pair_csv_export_preserves_raw_fields_and_refuses_overwrite(tmp_path):
    path = tmp_path / "pairs.csv"
    pair = {"query_id": "q1", "gallery_id": "g1", "similarity": .91,
            "pair_type": "genuine", "split": TEST, "score_status": "scored"}
    export_similarity_pairs(path, [pair])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == {"query_id": "q1", "gallery_id": "g1", "similarity": "0.91",
                       "pair_type": "genuine", "split": TEST, "score_status": "scored"}
    with pytest.raises(QueryGalleryProtocolError, match="refusing to overwrite"):
        export_similarity_pairs(path, [pair])
