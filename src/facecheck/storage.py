from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .chunks import iter_file_chunks
from .crypto import sm4_decrypt, sm4_encrypt

MANIFEST_NAME = "manifest.json"
CHUNKS_DIR_NAME = "chunks"
MANIFEST_VERSION = 1
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class StorageError(ValueError):
    """Raised when encrypted video storage is invalid or incomplete."""


@dataclass(frozen=True)
class StoredVideo:
    video_id: str
    directory: Path
    manifest: dict[str, object]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context(video_id: str, chunk_index: int, key_version: str) -> bytes:
    return json.dumps(
        {
            "key_version": key_version,
            "purpose": "facecheck-video-chunk",
            "video_id": video_id,
            "chunk_index": chunk_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _chunk_path(directory: Path, index: int) -> Path:
    return directory / CHUNKS_DIR_NAME / f"{index:08d}.bin"


def store_encrypted_video(
    source: Path,
    storage_root: Path,
    encryption_key: bytes,
    mac_key: bytes,
    *,
    chunk_size: int,
    key_version: str,
    video_id: Optional[str] = None,
    stats: Optional[Dict[str, float]] = None,
) -> StoredVideo:
    if not source.is_file():
        raise StorageError(f"source video does not exist: {source}")
    if not key_version:
        raise StorageError("key_version must not be empty")
    video_id = video_id or uuid.uuid4().hex
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise StorageError("video_id contains invalid characters")
    directory = storage_root / video_id
    chunks_directory = directory / CHUNKS_DIR_NAME
    if directory.exists():
        raise StorageError(f"storage directory already exists: {directory}")
    chunks_directory.mkdir(parents=True)
    chunk_records: List[Dict[str, object]] = []
    try:
        iterator = iter_file_chunks(source, chunk_size)
        while True:
            read_started = time.perf_counter()
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            if stats is not None:
                stats["file_read_seconds"] = stats.get("file_read_seconds", 0.0) + (
                    time.perf_counter() - read_started
                )
            crypto_stats: Dict[str, float] = {}
            encrypted = sm4_encrypt(
                chunk.data,
                encryption_key,
                mac_key,
                associated_data=_context(video_id, chunk.index, key_version),
                stats=crypto_stats,
            )
            if stats is not None:
                for name, value in crypto_stats.items():
                    stats[name] = stats.get(name, 0.0) + value
            write_started = time.perf_counter()
            path = _chunk_path(directory, chunk.index)
            path.write_bytes(encrypted)
            if stats is not None:
                stats["ciphertext_write_seconds"] = stats.get(
                    "ciphertext_write_seconds", 0.0
                ) + (time.perf_counter() - write_started)
            chunk_records.append({"index": chunk.index, "size": chunk.size, "file": path.name})
        digest_started = time.perf_counter()
        source_digest = _digest(source)
        if stats is not None:
            stats["source_sha256_seconds"] = stats.get(
                "source_sha256_seconds", 0.0
            ) + (time.perf_counter() - digest_started)
        manifest = {
            "version": MANIFEST_VERSION,
            "video_id": video_id,
            "key_version": key_version,
            "chunk_size": chunk_size,
            "chunk_count": len(chunk_records),
            "source_size": source.stat().st_size,
            "source_sha256": source_digest,
            "chunks": chunk_records,
        }
        (directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return StoredVideo(video_id=video_id, directory=directory, manifest=manifest)
    except Exception:
        for child in sorted(directory.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
        for child in sorted(directory.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        directory.rmdir()
        raise


def restore_encrypted_video(
    stored_directory: Path,
    destination: Path,
    encryption_key: bytes,
    mac_key: bytes,
    stats: Optional[Dict[str, float]] = None,
) -> Path:
    manifest_path = stored_directory / MANIFEST_NAME
    chunks_directory = stored_directory / CHUNKS_DIR_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StorageError(f"manifest is missing: {manifest_path}")
    if chunks_directory.is_symlink() or not chunks_directory.is_dir():
        raise StorageError(f"chunks directory is missing: {chunks_directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_id = manifest["video_id"]
        key_version = manifest["key_version"]
        records = manifest["chunks"]
        expected_count = manifest["chunk_count"]
        manifest_version = manifest["version"]
        chunk_size = manifest["chunk_size"]
        expected_size = manifest["source_size"]
        expected_digest = manifest["source_sha256"]
        if (
            manifest_version != MANIFEST_VERSION
            or not isinstance(video_id, str)
            or not isinstance(key_version, str)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_digest, str)
        ):
            raise TypeError
        if not isinstance(records, list) or not isinstance(expected_count, int) or expected_count < 0:
            raise TypeError
        if len(records) != expected_count or [
            r["index"] for r in records
        ] != list(range(expected_count)):
            raise StorageError("manifest chunk sequence is invalid")
        sizes = [r["size"] for r in records]
        if (
            any(not isinstance(size, int) or size < 0 for size in sizes)
            or sum(sizes) != expected_size
            or any(size != chunk_size for size in sizes[:-1])
            or (sizes and sizes[-1] > chunk_size)
            or (not sizes and expected_size != 0)
        ):
            raise StorageError("manifest chunk sizes are invalid")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise StorageError("manifest is invalid") from exc
    expected_files = {_chunk_path(stored_directory, index).name for index in range(expected_count)}
    actual_paths = list(chunks_directory.glob("*.bin"))
    if any(path.is_symlink() for path in actual_paths):
        raise StorageError("symbolic links are not allowed for chunks")
    actual_files = {path.name for path in actual_paths}
    missing_files = expected_files - actual_files
    if missing_files:
        raise StorageError(f"chunk is missing: {sorted(missing_files)[0]}")
    extra_files = actual_files - expected_files
    if extra_files:
        raise StorageError(f"unexpected chunk file: {sorted(extra_files)[0]}")
    if destination.exists() or destination.is_symlink():
        raise StorageError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise StorageError(f"temporary destination already exists: {temporary}")
    try:
        with temporary.open("wb") as output:
            for record in records:
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("index"), int)
                    or record.get("file") != _chunk_path(stored_directory, record["index"]).name
                ):
                    raise StorageError("manifest chunk filename is invalid")
                index = record["index"]
                chunk_path = _chunk_path(stored_directory, index)
                if not chunk_path.is_file():
                    raise StorageError(f"chunk is missing: {chunk_path}")
                read_started = time.perf_counter()
                encrypted = chunk_path.read_bytes()
                if stats is not None:
                    stats["ciphertext_read_seconds"] = stats.get(
                        "ciphertext_read_seconds", 0.0
                    ) + (time.perf_counter() - read_started)
                crypto_stats: Dict[str, float] = {}
                plaintext = sm4_decrypt(
                    encrypted,
                    encryption_key,
                    mac_key,
                    associated_data=_context(video_id, index, key_version),
                    stats=crypto_stats,
                )
                if stats is not None:
                    for name, value in crypto_stats.items():
                        stats[name] = stats.get(name, 0.0) + value
                if len(plaintext) != record["size"]:
                    raise StorageError(f"chunk size mismatch: {index}")
                write_started = time.perf_counter()
                output.write(plaintext)
                if stats is not None:
                    stats["plaintext_write_seconds"] = stats.get(
                        "plaintext_write_seconds", 0.0
                    ) + (time.perf_counter() - write_started)
        digest_started = time.perf_counter()
        restored_digest = _digest(temporary)
        if stats is not None:
            stats["sha256_seconds"] = stats.get("sha256_seconds", 0.0) + (
                time.perf_counter() - digest_started
            )
        if temporary.stat().st_size != expected_size or restored_digest != expected_digest:
            raise StorageError("restored video integrity check failed")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination
