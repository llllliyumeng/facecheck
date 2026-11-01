from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from gmssl import sm2

from facecheck.crypto import SM2_PRIVATE_KEY, SM2_PUBLIC_KEY
from facecheck.key_management import Sm2MemoryKeyProvider
from facecheck.key_rotation import EnvelopedRotationItem, rotate_enveloped_batch
from facecheck.storage import (
    StorageError,
    rewrap_enveloped_video,
    restore_encrypted_video,
    restore_enveloped_video,
    store_encrypted_video,
    store_enveloped_video,
)


def _second_sm2_keypair() -> tuple[str, str]:
    private_key = f"{2:064x}"
    calculator = sm2.CryptSM2(public_key="", private_key=private_key)
    public_key = calculator._kg(int(private_key, 16), sm2.default_ecc_table["g"])
    return private_key, public_key


def _provider(key_id: str = "recipient-v1") -> Sm2MemoryKeyProvider:
    return Sm2MemoryKeyProvider(
        key_id=key_id,
        public_key=SM2_PUBLIC_KEY,
        private_key=SM2_PRIVATE_KEY,
    )


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "small-source.bin"
    path.write_bytes(b"small synthetic video payload for the v2 envelope prototype")
    return path


def test_v2_envelope_round_trip_and_authenticated_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    provider = _provider()
    stats = {}
    stored = store_enveloped_video(
        source,
        tmp_path / "storage-v2",
        provider,
        chunk_size=17,
        key_version="video-dek-v1",
        video_id="v2-roundtrip",
        stats=stats,
    )
    manifest = json.loads((stored.directory / "manifest.json").read_text())
    assert manifest["version"] == 2
    assert manifest["key_envelope"]["key_id"] == provider.key_id
    assert manifest["manifest_authentication"]["algorithm"] == "HMAC-SM3"
    assert stored.manifest["version"] == 2
    assert stats["sm2_key_wrap_seconds"] >= 0

    destination = restore_enveloped_video(
        stored.directory,
        tmp_path / "restored.bin",
        provider,
        expected_video_id="v2-roundtrip",
    )
    assert destination.read_bytes() == source.read_bytes()


def test_v2_manifest_identity_tampering_and_wrong_provider_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    provider = _provider()
    stored = store_enveloped_video(
        source, tmp_path / "storage", provider,
        chunk_size=64, key_version="video-dek-v1", video_id="v2-auth",
    )

    with pytest.raises(StorageError, match="identity"):
        restore_enveloped_video(
            stored.directory, tmp_path / "wrong-id.bin", provider,
            expected_video_id="other-video",
        )
    assert not (tmp_path / "wrong-id.bin").exists()

    wrong_recipient = _provider(key_id="recipient-v2")
    with pytest.raises(StorageError, match="envelope"):
        restore_enveloped_video(
            stored.directory, tmp_path / "wrong-key.bin", wrong_recipient,
            expected_video_id="v2-auth",
        )
    assert not (tmp_path / "wrong-key.bin").exists()

    manifest_path = stored.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunk_size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageError, match="manifest authentication failed"):
        restore_enveloped_video(
            stored.directory, tmp_path / "tampered-manifest.bin", provider,
            expected_video_id="v2-auth",
        )
    assert not (tmp_path / "tampered-manifest.bin").exists()
    assert not (tmp_path / "tampered-manifest.bin.partial").exists()


def test_v2_chunk_tampering_is_rejected_without_publishing_plaintext(tmp_path: Path) -> None:
    source = _source(tmp_path)
    provider = _provider()
    stored = store_enveloped_video(
        source, tmp_path / "storage", provider,
        chunk_size=19, key_version="video-dek-v1", video_id="v2-chunk",
    )
    chunk = stored.directory / "chunks" / "00000000.bin"
    damaged = bytearray(chunk.read_bytes())
    damaged[-1] ^= 1
    chunk.write_bytes(damaged)

    destination = tmp_path / "damaged-restore.bin"
    with pytest.raises(ValueError, match="authentication failed"):
        restore_enveloped_video(
            stored.directory, destination, provider,
            expected_video_id="v2-chunk",
        )
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_v1_remains_readable_and_rewrap_keeps_source_usable(tmp_path: Path) -> None:
    source = _source(tmp_path)
    legacy = store_encrypted_video(
        source, tmp_path / "storage-v1", b"0123456789abcdef", b"legacy-mac-key",
        chunk_size=23, key_version="legacy-v1", video_id="legacy-video",
    )
    legacy_manifest = json.loads((legacy.directory / "manifest.json").read_text())
    assert legacy_manifest["version"] == 1
    restored_legacy = restore_encrypted_video(
        legacy.directory, tmp_path / "legacy-restored.bin",
        b"0123456789abcdef", b"legacy-mac-key",
    )
    assert restored_legacy.read_bytes() == source.read_bytes()

    old_provider = _provider("recipient-old")
    old_object = store_enveloped_video(
        source, tmp_path / "storage-old", old_provider,
        chunk_size=23, key_version="video-dek-v1", video_id="rewrap-video",
    )
    new_private, new_public = _second_sm2_keypair()
    new_provider = Sm2MemoryKeyProvider(
        key_id="recipient-new", public_key=new_public, private_key=new_private,
    )
    old_manifest = (old_object.directory / "manifest.json").read_bytes()
    source_chunk = (old_object.directory / "chunks" / "00000000.bin").read_bytes()
    rotated = rewrap_enveloped_video(
        old_object.directory,
        tmp_path / "storage-new",
        old_provider,
        new_provider,
        expected_video_id="rewrap-video",
        expected_manifest_sha256=hashlib.sha256(old_manifest).hexdigest(),
    )
    new_manifest = json.loads((rotated.directory / "manifest.json").read_text())
    assert new_manifest["key_envelope"]["key_id"] == "recipient-new"
    assert (rotated.directory / "chunks" / "00000000.bin").read_bytes() == source_chunk
    assert (old_object.directory / "manifest.json").read_bytes() == old_manifest

    restored_new = restore_enveloped_video(
        rotated.directory, tmp_path / "new-key-restored.bin", new_provider,
        expected_video_id="rewrap-video",
    )
    restored_old = restore_enveloped_video(
        old_object.directory, tmp_path / "old-key-still-works.bin", old_provider,
        expected_video_id="rewrap-video",
    )
    assert restored_new.read_bytes() == source.read_bytes()
    assert restored_old.read_bytes() == source.read_bytes()


def test_batch_rewrap_journal_resumes_failed_item_without_replacing_sources(
    tmp_path: Path,
) -> None:
    old_provider = _provider("batch-old")
    new_private, new_public = _second_sm2_keypair()
    new_provider = Sm2MemoryKeyProvider(
        key_id="batch-new", public_key=new_public, private_key=new_private,
    )
    source_root = tmp_path / "source-objects"
    source_root.mkdir()
    items = []
    original_chunks = {}
    for index, video_id in enumerate(("batch-a", "batch-b")):
        source = tmp_path / f"source-{index}.bin"
        source.write_bytes((f"small batch video {index}" * 5).encode())
        stored = store_enveloped_video(
            source, source_root, old_provider,
            chunk_size=12, key_version="video-dek-v1", video_id=video_id,
        )
        manifest = stored.directory / "manifest.json"
        items.append(EnvelopedRotationItem(
            video_id=video_id,
            source_directory=stored.directory,
            source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        ))
        chunk = stored.directory / "chunks" / "00000000.bin"
        original_chunks[video_id] = chunk.read_bytes()

    damaged_chunk = items[1].source_directory / "chunks" / "00000000.bin"
    damaged = bytearray(damaged_chunk.read_bytes())
    damaged[-1] ^= 1
    damaged_chunk.write_bytes(damaged)

    target_root = tmp_path / "rotated-objects"
    journal_path = tmp_path / "rotation-journal.json"
    partial = rotate_enveloped_batch(
        items, target_root, old_provider, new_provider, journal_path=journal_path,
    )
    assert partial["status"] == "partial"
    assert partial["items"]["batch-a"]["state"] == "complete"
    assert partial["items"]["batch-b"]["state"] == "failed"
    assert stat_mode(journal_path) == 0o600

    damaged_chunk.write_bytes(original_chunks["batch-b"])
    resumed = rotate_enveloped_batch(
        items, target_root, old_provider, new_provider, journal_path=journal_path,
    )
    assert resumed["status"] == "complete"
    assert resumed["items"]["batch-a"]["attempts"] == 1
    assert resumed["items"]["batch-b"]["attempts"] == 2
    for item in items:
        destination = tmp_path / f"{item.video_id}-restored.bin"
        restore_enveloped_video(
            target_root / item.video_id, destination, new_provider,
            expected_video_id=item.video_id,
        )
        assert destination.is_file()
        assert (item.source_directory / "manifest.json").is_file()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
