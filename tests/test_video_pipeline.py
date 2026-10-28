import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from facecheck.chunks import count_file_chunks, iter_file_chunks
from facecheck.storage import StorageError, restore_encrypted_video, store_encrypted_video
from facecheck.video import VideoProcessingError, extract_frames, read_video_info


def _write_test_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        6.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v writer is unavailable in this environment")
    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_info_and_frame_extraction(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_test_video(video)

    info = read_video_info(video)
    frames = extract_frames(video, interval_seconds=0.25)

    assert info.frame_count == 6
    assert info.fps == pytest.approx(6.0, abs=0.2)
    assert (info.width, info.height) == (32, 24)
    assert info.duration_seconds == pytest.approx(1.0, abs=0.2)
    assert len(frames) >= 3
    assert frames[0].index == 0
    assert frames[0].timestamp_seconds >= 0
    assert all(frame.image is not None for frame in frames)


def test_video_errors_and_invalid_interval(tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="unable to open"):
        read_video_info(tmp_path / "missing.mp4")
    with pytest.raises(ValueError, match="greater than zero"):
        extract_frames(tmp_path / "missing.mp4", interval_seconds=0)


def test_chunk_boundaries_and_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")

    chunks = list(iter_file_chunks(source, 4))
    assert [(chunk.index, chunk.size, chunk.data) for chunk in chunks] == [
        (0, 4, b"0123"),
        (1, 4, b"4567"),
        (2, 2, b"89"),
    ]
    assert count_file_chunks(source, 4) == 3
    empty = tmp_path / "empty.bin"
    empty.touch()
    assert list(iter_file_chunks(empty, 4)) == []


def test_encrypted_storage_restores_identical_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_test_video(source)
    encryption_key = b"0123456789abcdef"
    mac_key = b"independent-mac-key"

    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        encryption_key,
        mac_key,
        chunk_size=100,
        key_version="video-key-v1",
        video_id="video-1",
    )
    restored = restore_encrypted_video(
        stored.directory,
        tmp_path / "restored.mp4",
        encryption_key,
        mac_key,
    )

    assert hashlib.sha256(restored.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    manifest = json.loads((stored.directory / "manifest.json").read_text())
    assert manifest["chunk_count"] > 1
    assert "encryption_key" not in manifest
    assert "mac_key" not in manifest


def test_storage_stats_are_optional_and_do_not_change_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789" * 100)
    stats = {}
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=32,
        key_version="video-key-v1",
        video_id="stats-video",
        stats=stats,
    )
    restore_stats = {}
    restored = restore_encrypted_video(
        stored.directory,
        tmp_path / "restored.bin",
        b"0123456789abcdef",
        b"independent-mac-key",
        stats=restore_stats,
    )

    assert restored.read_bytes() == source.read_bytes()
    assert stats["source_sha256_seconds"] >= 0
    assert stats["sm4_seconds"] > 0
    assert stats["hmac_sm3_seconds"] > 0
    assert stats["ciphertext_write_seconds"] >= 0
    assert restore_stats["sm4_seconds"] > 0
    assert restore_stats["hmac_sm3_seconds"] > 0
    assert restore_stats["sha256_seconds"] >= 0
    assert restore_stats["plaintext_write_seconds"] >= 0
    assert all(
        "0123456789abcdef" not in str(value)
        and "independent-mac-key" not in str(value)
        for value in [*stats.values(), *restore_stats.values()]
    )


def test_empty_file_round_trip_and_existing_destination_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.bin"
    source.touch()
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=100,
        key_version="video-key-v1",
        video_id="empty-video",
    )
    destination = tmp_path / "restored.bin"
    restore_encrypted_video(stored.directory, destination, b"0123456789abcdef", b"independent-mac-key")
    assert destination.read_bytes() == b""
    with pytest.raises(StorageError, match="destination already exists"):
        restore_encrypted_video(
            stored.directory,
            destination,
            b"0123456789abcdef",
            b"independent-mac-key",
        )


def test_storage_rejects_chunk_symlink_and_existing_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=100,
        key_version="video-key-v1",
        video_id="symlink-video",
    )
    chunk = stored.directory / "chunks" / "00000000.bin"
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    chunk.unlink()
    chunk.symlink_to(target)
    with pytest.raises(StorageError, match="symbolic links"):
        restore_encrypted_video(
            stored.directory,
            tmp_path / "symlink-restored.bin",
            b"0123456789abcdef",
            b"independent-mac-key",
        )

    stored = store_encrypted_video(
        source,
        tmp_path / "storage-partial",
        b"0123456789abcdef",
        b"independent-mac-key",
        chunk_size=100,
        key_version="video-key-v1",
        video_id="partial-video",
    )
    partial = tmp_path / "partial-restored.bin.partial"
    partial.write_bytes(b"stale")
    with pytest.raises(StorageError, match="temporary destination already exists"):
        restore_encrypted_video(
            stored.directory,
            tmp_path / "partial-restored.bin",
            b"0123456789abcdef",
            b"independent-mac-key",
        )


def test_storage_rejects_manifest_sequence_and_size_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(201)))
    key = b"0123456789abcdef"
    mac = b"independent-mac-key"

    for name, mutate in (
        ("version", lambda manifest: manifest.update(version=99)),
        ("duplicate", lambda manifest: manifest["chunks"][1].update(index=0)),
        ("size", lambda manifest: manifest["chunks"][0].update(size=99)),
        ("source-size", lambda manifest: manifest.update(source_size=0)),
    ):
        stored = store_encrypted_video(
            source,
            tmp_path / f"storage-{name}",
            key,
            mac,
            chunk_size=100,
            key_version="video-key-v1",
            video_id=f"manifest-{name}",
        )
        manifest_path = stored.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(StorageError):
            restore_encrypted_video(
                stored.directory,
                tmp_path / f"{name}.bin",
                key,
                mac,
            )


def test_storage_rejects_missing_swapped_tampered_and_metadata_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(251)))
    encryption_key = b"0123456789abcdef"
    mac_key = b"independent-mac-key"
    stored = store_encrypted_video(
        source,
        tmp_path / "storage",
        encryption_key,
        mac_key,
        chunk_size=100,
        key_version="video-key-v1",
        video_id="video-2",
    )
    chunk_paths = sorted((stored.directory / "chunks").glob("*.bin"))

    chunk_paths[0].unlink()
    with pytest.raises(StorageError, match="chunk is missing"):
        restore_encrypted_video(stored.directory, tmp_path / "missing.bin", encryption_key, mac_key)

    stored = store_encrypted_video(
        source,
        tmp_path / "storage-swapped",
        encryption_key,
        mac_key,
        chunk_size=100,
        key_version="video-key-v1",
        video_id="video-swapped",
    )
    paths = sorted((stored.directory / "chunks").glob("*.bin"))
    first_data = paths[0].read_bytes()
    second_data = paths[1].read_bytes()
    paths[0].write_bytes(second_data)
    paths[1].write_bytes(first_data)
    with pytest.raises(ValueError, match="authentication failed"):
        restore_encrypted_video(stored.directory, tmp_path / "swapped.bin", encryption_key, mac_key)

    stored = store_encrypted_video(
        source,
        tmp_path / "storage-2",
        encryption_key,
        mac_key,
        chunk_size=100,
        key_version="video-key-v1",
        video_id="video-3",
    )
    paths = sorted((stored.directory / "chunks").glob("*.bin"))
    paths[0].write_bytes(paths[0].read_bytes()[:-1] + bytes([paths[0].read_bytes()[-1] ^ 1]))
    with pytest.raises(ValueError, match="authentication failed"):
        restore_encrypted_video(stored.directory, tmp_path / "tampered.bin", encryption_key, mac_key)

    manifest_path = stored.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["key_version"] = "different-key-version"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="authentication failed"):
        restore_encrypted_video(stored.directory, tmp_path / "metadata.bin", encryption_key, mac_key)

    manifest["key_version"] = "video-key-v1"
    manifest_path.write_text(json.dumps(manifest))
    (stored.directory / "chunks" / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(StorageError, match="unexpected chunk file"):
        restore_encrypted_video(stored.directory, tmp_path / "extra.bin", encryption_key, mac_key)
