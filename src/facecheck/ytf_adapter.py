"""Small-sample adapter for the YouTube Faces (YTF) frame archive.

This module parses official pair folds and builds a *development* query/gallery
manifest. Reconstructed gallery videos are derivative inputs, not original YTF MP4s.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import cv2


class YTFAdapterError(ValueError):
    """YTF metadata, archive mapping, or sample plan is invalid."""


@dataclass(frozen=True)
class YTFVideo:
    index: int  # Official MATLAB index, validated as one-based.
    video_name: str
    subject_name: str
    video_number: str
    identity_label: int
    identity_id: str
    mat_name: str


@dataclass(frozen=True)
class YTFPair:
    fold: int
    pair_number: int
    video_index_1: int
    video_index_2: int
    is_match: bool


@dataclass(frozen=True)
class YTFMetadata:
    videos: tuple[YTFVideo, ...]
    folds: tuple[tuple[YTFPair, ...], ...]
    fields: tuple[str, ...]


def select_two_videos_per_identity(metadata: YTFMetadata) -> tuple[tuple[YTFVideo, YTFVideo], ...]:
    """Choose the first two distinct MATLAB-indexed sequences for every eligible identity."""
    grouped: dict[int, list[YTFVideo]] = {}
    for video in metadata.videos:
        grouped.setdefault(video.identity_label, []).append(video)
    selected = []
    for label in sorted(grouped):
        videos = sorted(grouped[label], key=lambda item: item.index)
        if len(videos) >= 2:
            first, second = videos[:2]
            if first.index == second.index or first.video_name == second.video_name:
                raise YTFAdapterError(f"identity {label} does not have two distinct videos")
            selected.append((first, second))
    return tuple(selected)


def audit_official_fold_partition(metadata: YTFMetadata, threshold_folds: Iterable[int],
                                  test_folds: Iterable[int]) -> dict[str, Any]:
    """Audit official fold partitions without altering pair membership or labels."""
    threshold = set(threshold_folds); test = set(test_folds)
    all_folds = set(range(1, len(metadata.folds) + 1))
    if not threshold or not test or threshold & test or not (threshold | test) <= all_folds:
        raise YTFAdapterError("threshold/test folds must be valid, non-empty, and disjoint")
    def rows(folds: set[int]) -> list[YTFPair]:
        return [pair for fold in folds for pair in metadata.folds[fold - 1]]
    train_rows, test_rows = rows(threshold), rows(test)
    def pair_key(pair: YTFPair) -> tuple[int, int, bool]:
        return min(pair.video_index_1, pair.video_index_2), max(pair.video_index_1, pair.video_index_2), pair.is_match
    train_pairs = {pair_key(p) for p in train_rows}; test_pairs = {pair_key(p) for p in test_rows}
    train_videos = {v for p in train_rows for v in (p.video_index_1, p.video_index_2)}
    test_videos = {v for p in test_rows for v in (p.video_index_1, p.video_index_2)}
    train_ids = {metadata.videos[v - 1].identity_label for v in train_videos}
    test_ids = {metadata.videos[v - 1].identity_label for v in test_videos}
    fold_pair_keys: dict[tuple[int, int, bool], set[int]] = {}
    for fold_no, fold_rows in enumerate(metadata.folds, 1):
        for pair in fold_rows:
            fold_pair_keys.setdefault(pair_key(pair), set()).add(fold_no)
    duplicate_pairs = [key for key, folds in fold_pair_keys.items() if len(folds) > 1]
    def summary(selected: list[YTFPair]) -> dict[str, Any]:
        return {"folds": sorted(threshold if selected is train_rows else test),
                "pair_count": len(selected), "genuine_count": sum(p.is_match for p in selected),
                "impostor_count": sum(not p.is_match for p in selected),
                "unique_video_count": len({v for p in selected for v in (p.video_index_1, p.video_index_2)}),
                "unique_identity_count": len({metadata.videos[v - 1].identity_label for p in selected for v in (p.video_index_1, p.video_index_2)})}
    return {"threshold_selection": summary(train_rows), "test": summary(test_rows),
            "threshold_test_pair_overlap": len(train_pairs & test_pairs),
            "threshold_test_video_overlap": len(train_videos & test_videos),
            "threshold_test_identity_overlap": len(train_ids & test_ids),
            "duplicate_pair_keys_across_all_folds": len(duplicate_pairs),
            "duplicate_pair_extra_appearances": sum(len(folds) - 1 for folds in fold_pair_keys.values() if len(folds) > 1),
            "independent_by_pair_video_identity": not (train_pairs & test_pairs or train_videos & test_videos or train_ids & test_ids)}


def _mat_string(value: Any) -> str:
    """Decode MATLAB char arrays and nested cell values without guessing IDs."""
    while isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            parts = value.reshape(-1).tolist()
            if len(parts) == 1:
                return str(parts[0]).rstrip("\x00 ")
            return "".join(str(part) for part in parts).rstrip("\x00 ")
        if value.size != 1:
            raise YTFAdapterError("MAT string cell contains more than one value")
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00 ")
    if isinstance(value, str):
        return value.rstrip("\x00 ")
    raise YTFAdapterError(f"unsupported MAT string value: {type(value).__name__}")


def load_ytf_metadata(mat_path: Path) -> YTFMetadata:
    """Parse YTF meta_and_splits.mat and validate its 1-based official pair indices."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # scipy remains an optional, local data-prep dependency
        raise YTFAdapterError("SciPy is required to parse YTF MAT metadata") from exc
    try:
        raw = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
    except Exception as exc:
        raise YTFAdapterError(f"could not read MAT file: {mat_path}") from exc
    fields = tuple(k for k in raw if not k.startswith("__"))
    for required in ("video_labels", "video_names", "mat_names", "Splits"):
        if required not in raw:
            raise YTFAdapterError(f"MAT file is missing {required}")
    labels = np.asarray(raw["video_labels"]).reshape(-1)
    name_cells = np.asarray(raw["video_names"]).reshape(-1)
    mat_cells = np.asarray(raw["mat_names"]).reshape(-1)
    split_array = np.asarray(raw["Splits"])
    if not (len(labels) == len(name_cells) == len(mat_cells)) or not len(labels):
        raise YTFAdapterError("video_labels, video_names and mat_names lengths differ")
    if not np.issubdtype(labels.dtype, np.number) or not np.all(np.isfinite(labels)):
        raise YTFAdapterError("video_labels must contain finite numeric labels")
    if np.any(labels <= 0) or np.any(labels != np.floor(labels)):
        raise YTFAdapterError("video_labels must be positive integer identity labels")
    videos: list[YTFVideo] = []
    seen: set[str] = set()
    subject_labels: dict[str, int] = {}
    label_subjects: dict[int, str] = {}
    for i, (label, name_cell, mat_cell) in enumerate(zip(labels, name_cells, mat_cells), 1):
        name, mat_name = _mat_string(name_cell), _mat_string(mat_cell).replace("\\", "/")
        parts = name.split("/")
        if len(parts) != 2 or not parts[0] or not re.fullmatch(r"\d+", parts[1]):
            raise YTFAdapterError(f"invalid video_names entry at MATLAB index {i}: {name!r}")
        expected_mat = f"{parts[0]}/video_{parts[1]}.mat"
        if mat_name != expected_mat:
            raise YTFAdapterError(f"mat_names/video_names mismatch at MATLAB index {i}: {mat_name!r} != {expected_mat!r}")
        if name in seen:
            raise YTFAdapterError(f"duplicate video name: {name}")
        seen.add(name)
        identity_label = int(label)
        previous_label = subject_labels.setdefault(parts[0], identity_label)
        previous_subject = label_subjects.setdefault(identity_label, parts[0])
        if previous_label != identity_label or previous_subject != parts[0]:
            raise YTFAdapterError("video_names subject directories and video_labels do not map one-to-one")
        videos.append(YTFVideo(i, name, parts[0], parts[1], identity_label,
                               f"id_{identity_label:04d}", mat_name))
    # YTF stores (500 pairs, 3 fields, 10 folds); synthetic tests may use fewer pairs.
    if split_array.ndim != 3 or split_array.shape[1] != 3 or split_array.shape[2] != 10:
        raise YTFAdapterError(f"Splits must have shape (pairs, 3, 10), got {split_array.shape}")
    if not np.issubdtype(split_array.dtype, np.number) or not np.all(np.isfinite(split_array)):
        raise YTFAdapterError("Splits must be finite numeric values")
    if np.any(split_array[:, :2, :] != np.floor(split_array[:, :2, :])):
        raise YTFAdapterError("Splits video indices must be integers")
    folds: list[tuple[YTFPair, ...]] = []
    for fold_idx in range(10):
        pairs = []
        for row_idx, row in enumerate(split_array[:, :, fold_idx], 1):
            first, second, flag = (int(row[0]), int(row[1]), int(row[2]))
            # Values range through N and the official labels agree after subtracting 1.
            if not (1 <= first <= len(videos) and 1 <= second <= len(videos)):
                raise YTFAdapterError(f"invalid 1-based video index in fold {fold_idx + 1}, pair {row_idx}")
            if flag not in (0, 1):
                raise YTFAdapterError(f"invalid match flag in fold {fold_idx + 1}, pair {row_idx}")
            actual_match = videos[first - 1].identity_label == videos[second - 1].identity_label
            if actual_match != bool(flag):
                raise YTFAdapterError(f"official match flag disagrees with video_labels in fold {fold_idx + 1}, pair {row_idx}")
            pairs.append(YTFPair(fold_idx + 1, row_idx, first, second, bool(flag)))
        folds.append(tuple(pairs))
    return YTFMetadata(tuple(videos), tuple(folds), fields)


def validate_archive_members(metadata: YTFMetadata, member_names: Iterable[str]) -> dict[str, Any]:
    """Validate exact metadata video directories against tar member names."""
    expected = {v.video_name for v in metadata.videos}
    found: set[str] = set()
    frame_counts: dict[str, int] = {}
    label_files: set[str] = set()
    malformed: list[str] = []
    frame_re = re.compile(r"^frame_images_DB/([^/]+)/([^/]+)/([^/]+)\.([0-9]+)\.jpg$")
    for raw_name in member_names:
        name = raw_name.lstrip("./")
        if not name.startswith("frame_images_DB/"):
            continue
        if name.endswith(".labeled_faces.txt"):
            label_files.add(name[len("frame_images_DB/"):])
        match = frame_re.fullmatch(name)
        if match:
            subject, video, stem, _frame = match.groups()
            if stem != video:
                malformed.append(name)
            rel = f"{subject}/{video}"
            found.add(rel)
            frame_counts[rel] = frame_counts.get(rel, 0) + 1
    missing, extra = sorted(expected - found), sorted(found - expected)
    no_frames = sorted(name for name in expected if not frame_counts.get(name))
    missing_labels = sorted({v.subject_name for v in metadata.videos} -
                            {name.removesuffix(".labeled_faces.txt") for name in label_files})
    return {"expected_video_count": len(expected), "found_video_count": len(found),
            "missing_video_names": missing, "extra_video_names": extra,
            "videos_without_frames": no_frames, "malformed_frame_paths": malformed,
            "missing_labeled_faces_subjects": missing_labels, "frame_counts": frame_counts,
            "valid": not (missing or extra or no_frames or malformed or missing_labels)}


def select_disjoint_positive_pairs(metadata: YTFMetadata, fold: int, count: int) -> tuple[YTFPair, ...]:
    """Greedily take official genuine pairs with no repeated videos or identities."""
    if not 1 <= fold <= len(metadata.folds) or count <= 0:
        raise YTFAdapterError("fold must be valid and count must be positive")
    selected: list[YTFPair] = []
    used_videos: set[int] = set()
    used_identities: set[int] = set()
    for pair in metadata.folds[fold - 1]:
        if not pair.is_match:
            continue
        first, second = metadata.videos[pair.video_index_1 - 1], metadata.videos[pair.video_index_2 - 1]
        if first.identity_label != second.identity_label:
            raise YTFAdapterError("positive official pair has inconsistent identities")
        if pair.video_index_1 in used_videos or pair.video_index_2 in used_videos:
            continue
        if first.identity_label in used_identities:
            continue
        selected.append(pair)
        used_videos.update((pair.video_index_1, pair.video_index_2))
        used_identities.add(first.identity_label)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise YTFAdapterError(f"only {len(selected)} disjoint positive pairs available in fold {fold}; requested {count}")
    return tuple(selected)


def video_frame_paths(frame_root: Path, video: YTFVideo) -> tuple[Path, ...]:
    """Return exact numbered frame paths for one metadata-mapped video directory."""
    directory = frame_root / video.subject_name / video.video_number
    if not directory.is_dir():
        raise YTFAdapterError(f"frame directory does not exist: {directory}")
    pattern = re.compile(rf"^{re.escape(video.video_number)}\.(\d+)\.jpg$")
    indexed = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        raise YTFAdapterError(f"no numbered frames found in {directory}")
    if len({index for index, _ in indexed}) != len(indexed):
        raise YTFAdapterError(f"duplicate frame numbers in {directory}")
    return tuple(path for _, path in indexed)


def annotated_frame_paths(label_file: Path, video: YTFVideo, frame_root: Path) -> tuple[Path, ...]:
    """Return frames listed in the subject labeled_faces file for the exact video."""
    if not label_file.is_file():
        raise YTFAdapterError(f"labeled_faces file does not exist: {label_file}")
    all_frames = {p.name: p for p in video_frame_paths(frame_root, video)}
    prefix = f"{video.subject_name}/{video.video_number}/"
    selected: list[Path] = []
    for line_no, line in enumerate(label_file.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        first = line.split(",", 1)[0].strip().replace("\\", "/").lstrip("./")
        if not first.startswith(prefix):
            continue
        relative = first[len(prefix):]
        if "/" in relative or relative not in all_frames:
            raise YTFAdapterError(f"annotation references missing/invalid frame at {label_file}:{line_no}: {first}")
        selected.append(all_frames[relative])
    # Preserve annotation ordering while removing duplicate references.
    unique = tuple(dict.fromkeys(selected))
    if not unique:
        raise YTFAdapterError(f"no annotated frames found for {video.video_name}")
    return unique


def reconstruct_video(frame_paths: Iterable[Path], output_path: Path, *, fps: float = 1.0) -> int:
    """Build a small derived MP4 from ordered YTF frames; never overwrites output."""
    if fps <= 0 or not np.isfinite(fps):
        raise YTFAdapterError("reconstruction fps must be positive and finite")
    frames = tuple(frame_paths)
    if not frames:
        raise YTFAdapterError("cannot reconstruct video from an empty frame sequence")
    if output_path.exists() or output_path.is_symlink():
        raise YTFAdapterError(f"refusing to overwrite reconstructed video: {output_path}")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise YTFAdapterError(f"cannot decode frame: {frames[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise YTFAdapterError(f"video writer could not open: {output_path}")
    try:
        for index, path in enumerate(frames):
            image = first if index == 0 else cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise YTFAdapterError(f"cannot decode frame: {path}")
            if image.shape[:2] != (height, width):
                raise YTFAdapterError(f"frame dimensions differ in sequence: {path}")
            writer.write(image)
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise YTFAdapterError(f"reconstructed video was not written: {output_path}")
    return len(frames)


def build_query_gallery_records(metadata: YTFMetadata, pairs: Iterable[YTFPair],
                                gallery_videos: Mapping[int, Path],
                                query_images: Mapping[int, Path], *,
                                frame_directories: Optional[Mapping[int, Path]] = None) -> list[dict[str, Any]]:
    """Build current experiment.py records; gallery videos must already be reconstructed."""
    records: list[dict[str, Any]] = []
    used: set[int] = set()
    for pair in pairs:
        gallery_idx, query_idx = pair.video_index_1, pair.video_index_2
        if gallery_idx in used or query_idx in used or gallery_idx == query_idx:
            raise YTFAdapterError("selected pair set reuses a video across query/gallery")
        used.update((gallery_idx, query_idx))
        gallery, query = metadata.videos[gallery_idx - 1], metadata.videos[query_idx - 1]
        if gallery.identity_label != query.identity_label:
            raise YTFAdapterError("query/gallery sample pair is not genuine")
        if gallery_idx not in gallery_videos or query_idx not in query_images:
            raise YTFAdapterError("missing reconstructed gallery video or query frame mapping")
        gp, qp = Path(gallery_videos[gallery_idx]), Path(query_images[query_idx])
        if not gp.is_file():
            raise YTFAdapterError(f"reconstructed gallery video does not exist: {gp}")
        if not qp.is_file():
            raise YTFAdapterError(f"query frame does not exist: {qp}")
        identity_id = gallery.identity_id
        label = f"subject_{gallery.identity_label:04d}"
        for item, split, path, media_type in ((gallery, "gallery", gp, "video"),
                                               (query, "query", qp, "image")):
            row: dict[str, Any] = {
                "identity_id": identity_id, "video_id": f"ytf_video_{item.index:04d}",
                "video_path": str(path), "split": split,
                "source_group_id": f"ytf_source_video_{item.index:04d}",
                "label": label, "media_type": media_type,
                "ytf_video_index_1based": item.index,
                "ytf_video_name": item.video_name, "ytf_mat_name": item.mat_name,
                "ytf_identity_label": item.identity_label, "ytf_fold": pair.fold,
                "ytf_pair_number": pair.pair_number,
            }
            if frame_directories and item.index in frame_directories:
                row["ytf_frame_directory"] = str(frame_directories[item.index])
            records.append(row)
    return records
