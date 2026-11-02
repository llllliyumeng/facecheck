import json

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from facecheck.experiment import load_manifest
from facecheck.ytf_adapter import (
    YTFAdapterError, build_query_gallery_records, load_ytf_metadata,
    annotated_frame_paths, select_disjoint_positive_pairs, validate_archive_members,
    video_frame_paths, select_two_videos_per_identity, audit_official_fold_partition,
)


def _make_mat(tmp_path, *, bad_match=False):
    names = ["PersonA/0", "PersonA/1", "PersonB/0", "PersonB/1", "PersonC/0", "PersonC/1"]
    labels = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.uint16)
    video_names = np.empty((6, 1), dtype=object)
    mat_names = np.empty((6, 1), dtype=object)
    for i, name in enumerate(names):
        video_names[i, 0] = name
        subject, number = name.split("/")
        mat_names[i, 0] = f"{subject}/video_{number}.mat"
    split = np.zeros((4, 3, 10), dtype=np.uint16)
    split[:, :, :] = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [1, 3, 0]], dtype=np.uint16)[:, :, None]
    if bad_match:
        split[0, 2, :] = 0
    path = tmp_path / "meta_and_splits.mat"
    scipy_io.savemat(path, {"video_labels": labels, "video_names": video_names,
                            "mat_names": mat_names, "Splits": split,
                            "extra_field": np.array([[7]])})
    return path


def test_mat_parser_labels_names_and_official_fold_structure(tmp_path):
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    assert "extra_field" in metadata.fields
    assert len(metadata.videos) == 6
    assert metadata.videos[0].index == 1
    assert metadata.videos[0].video_name == "PersonA/0"
    assert metadata.videos[0].identity_label == 1
    assert metadata.videos[0].identity_id == "id_0001"
    assert len(metadata.folds) == 10 and len(metadata.folds[0]) == 4
    assert [p.is_match for p in metadata.folds[0]] == [True, True, True, False]
    assert metadata.folds[0][0].video_index_1 == 1


def test_mat_parser_rejects_pair_flag_that_disagrees_with_identity_labels(tmp_path):
    with pytest.raises(YTFAdapterError, match="disagrees"):
        load_ytf_metadata(_make_mat(tmp_path, bad_match=True))


def test_two_video_identity_selection_and_disjoint_official_fold_partition(tmp_path):
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    selected = select_two_videos_per_identity(metadata)
    assert len(selected) == 3
    assert all(a.identity_label == b.identity_label and a.index != b.index for a, b in selected)
    audit = audit_official_fold_partition(metadata, [1], list(range(2, 11)))
    assert audit["threshold_selection"]["genuine_count"] == 3
    assert audit["threshold_selection"]["impostor_count"] == 1
    assert audit["threshold_test_video_overlap"] > 0
    assert audit["independent_by_pair_video_identity"] is False


def test_exact_archive_path_mapping_and_missing_frames(tmp_path):
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    paths = [
        "frame_images_DB/PersonA/0/0.1.jpg",
        "frame_images_DB/PersonA/0/0.2.jpg",
        "frame_images_DB/PersonA.labeled_faces.txt",
        "frame_images_DB/PersonA/1/1.3.jpg",
        "frame_images_DB/PersonB/0/0.1.jpg",
        "frame_images_DB/PersonB/1/1.2.jpg",
        "frame_images_DB/PersonB.labeled_faces.txt",
        "frame_images_DB/PersonC/0/0.4.jpg",
        "frame_images_DB/PersonC/1/1.5.jpg",
        "frame_images_DB/PersonC.labeled_faces.txt",
    ]
    report = validate_archive_members(metadata, paths)
    assert report["valid"] is True
    assert report["frame_counts"]["PersonA/0"] == 2
    broken = validate_archive_members(metadata, paths[:8] + paths[9:])
    assert broken["valid"] is False
    assert "PersonC/1" in broken["missing_video_names"]


def test_disjoint_positive_pair_plan_and_experiment_manifest_compatibility(tmp_path):
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    pairs = select_disjoint_positive_pairs(metadata, fold=1, count=3)
    assert len({p.video_index_1 for p in pairs} | {p.video_index_2 for p in pairs}) == 6
    assert all(p.is_match for p in pairs)
    gallery, queries = {}, {}
    for pair in pairs:
        g = tmp_path / f"gallery-{pair.video_index_1}.mp4"
        q = tmp_path / f"query-{pair.video_index_2}.jpg"
        g.write_bytes(b"reconstructed-video-placeholder")
        q.write_bytes(b"image-placeholder")
        gallery[pair.video_index_1] = g
        queries[pair.video_index_2] = q
    rows = build_query_gallery_records(metadata, pairs, gallery, queries)
    assert len(rows) == 6
    assert len({r["source_group_id"] for r in rows}) == 6
    path = tmp_path / "experiment_small.json"
    path.write_text(json.dumps({"records": rows}))
    manifest = load_manifest(path)
    assert len(manifest.records) == 6
    assert sum(r.split == "gallery" for r in manifest.records) == 3
    assert sum(r.split == "query" for r in manifest.records) == 3


def test_adapter_rejects_missing_frame_or_reconstructed_paths(tmp_path):
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    pairs = select_disjoint_positive_pairs(metadata, 1, 1)
    with pytest.raises(YTFAdapterError, match="mapping"):
        build_query_gallery_records(metadata, pairs, {}, {})
    idx1, idx2 = pairs[0].video_index_1, pairs[0].video_index_2
    gallery = tmp_path / "made.mp4"
    gallery.write_bytes(b"placeholder")
    with pytest.raises(YTFAdapterError, match="query frame does not exist"):
        build_query_gallery_records(metadata, pairs, {idx1: gallery}, {idx2: tmp_path / "missing.jpg"})


def test_frame_sequence_and_labeled_faces_path_mapping(tmp_path):
    import cv2
    metadata = load_ytf_metadata(_make_mat(tmp_path))
    video = metadata.videos[0]
    folder = tmp_path / video.subject_name / video.video_number
    folder.mkdir(parents=True)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    for frame_no in (12, 3):
        assert cv2.imwrite(str(folder / f"{video.video_number}.{frame_no}.jpg"), image)
    paths = video_frame_paths(tmp_path, video)
    assert [p.name for p in paths] == ["0.3.jpg", "0.12.jpg"]
    annotation = tmp_path / f"{video.subject_name}.labeled_faces.txt"
    annotation.write_text(f"{video.subject_name}\\0\\0.12.jpg,0,1,2,3,4,0,1\n")
    assert annotated_frame_paths(annotation, video, tmp_path) == (paths[1],)
    annotation.write_text(f"{video.subject_name}\\0\\0.99.jpg,0,1,2,3,4,0,1\n")
    with pytest.raises(YTFAdapterError, match="missing/invalid frame"):
        annotated_frame_paths(annotation, video, tmp_path)
