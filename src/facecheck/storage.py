from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .chunks import iter_file_chunks
from .crypto import derive_storage_keys, sm3_hmac, sm3_kdf, sm4_decrypt, sm4_encrypt
from .key_management import (
    VIDEO_KEY_MATERIAL_SIZE,
    VideoKeyProvider,
    WrappedVideoKey,
)

MANIFEST_NAME = "manifest.json"
CHUNKS_DIR_NAME = "chunks"
MANIFEST_VERSION = 1
ENVELOPED_MANIFEST_VERSION = 2
KEY_DERIVATION = "sm3-kdf-v1"
MANIFEST_AUTHENTICATION = "HMAC-SM3"
LEGACY_KEY_DERIVATION = "legacy-raw"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_CRYPTO_WORKERS = 32


class StorageError(ValueError):
    """Raised when encrypted video storage is invalid or incomplete."""


@dataclass(frozen=True)
class StoredVideo:
    video_id: str
    directory: Path
    manifest: dict[str, object]


def _canonical_manifest(manifest: dict[str, object]) -> bytes:
    unsigned = {name: value for name, value in manifest.items()
                if name != "manifest_authentication"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _validate_v2_key_version(key_version: str) -> bytes:
    if not isinstance(key_version, str) or not key_version:
        raise StorageError("v2 key version is invalid")
    try:
        encoded = key_version.encode("utf-8")
    except UnicodeEncodeError:
        raise StorageError("v2 key version is invalid") from None
    if len(encoded) > 128:
        raise StorageError("v2 key version is invalid")
    return encoded


def _manifest_auth_key(mac_master_key: bytes, video_id: str, key_version: str) -> bytes:
    _validate_v2_key_version(key_version)
    context = (
        b"facecheck-video-manifest-auth-v2\x00"
        + len(video_id.encode("utf-8")).to_bytes(2, "big")
        + video_id.encode("utf-8")
        + len(key_version.encode("utf-8")).to_bytes(2, "big")
        + key_version.encode("utf-8")
    )
    return sm3_kdf(mac_master_key, context, 32)


def _authenticate_manifest(
    manifest: dict[str, object], mac_master_key: bytes, video_id: str, key_version: str,
) -> None:
    manifest["manifest_authentication"] = {
        "algorithm": MANIFEST_AUTHENTICATION,
        "key_derivation": KEY_DERIVATION,
        "tag": sm3_hmac(
            _manifest_auth_key(mac_master_key, video_id, key_version),
            _canonical_manifest(manifest),
        ).hex(),
    }


def _verify_manifest_authentication(
    manifest: dict[str, object], mac_master_key: bytes, video_id: str, key_version: str,
) -> None:
    authentication = manifest.get("manifest_authentication")
    if (not isinstance(authentication, dict)
            or set(authentication) != {"algorithm", "key_derivation", "tag"}
            or authentication.get("algorithm") != MANIFEST_AUTHENTICATION
            or authentication.get("key_derivation") != KEY_DERIVATION
            or not isinstance(authentication.get("tag"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", authentication["tag"])):
        raise StorageError("manifest authentication failed")
    expected = sm3_hmac(
        _manifest_auth_key(mac_master_key, video_id, key_version),
        _canonical_manifest(manifest),
    ).hex()
    if not hmac.compare_digest(authentication["tag"], expected):
        raise StorageError("manifest authentication failed")


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


def _validate_workers(workers: int) -> int:
    if (isinstance(workers, bool) or not isinstance(workers, int)
            or workers <= 0 or workers > MAX_CRYPTO_WORKERS):
        raise StorageError(f"workers must be an integer between 1 and {MAX_CRYPTO_WORKERS}")
    return workers


def _encrypt_chunk(
    chunk: object,
    encryption_key: bytes,
    mac_key: bytes,
    video_id: str,
    key_version: str,
) -> Tuple[object, bytes, Dict[str, float]]:
    crypto_stats: Dict[str, float] = {}
    encrypted = sm4_encrypt(
        chunk.data,
        encryption_key,
        mac_key,
        associated_data=_context(video_id, chunk.index, key_version),
        stats=crypto_stats,
    )
    return chunk, encrypted, crypto_stats


def _iter_parallel_encrypted(
    chunks: Iterator[object],
    encryption_key: bytes,
    mac_key: bytes,
    video_id: str,
    key_version: str,
    workers: int,
) -> Iterator[Tuple[object, bytes, Dict[str, float], float]]:
    """Encrypt a bounded number of chunks concurrently and yield in order."""
    if workers == 1:
        while True:
            started = time.perf_counter()
            try:
                chunk = next(chunks)
            except StopIteration:
                break
            read_seconds = time.perf_counter() - started
            yield (*_encrypt_chunk(chunk, encryption_key, mac_key, video_id, key_version), read_seconds)
        return
    pending: Dict[int, Future[Tuple[object, bytes, Dict[str, float]]]] = {}
    read_seconds: Dict[int, float] = {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="facecheck-sm4") as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                started = time.perf_counter()
                try:
                    chunk = next(chunks)
                except StopIteration:
                    exhausted = True
                    break
                read_seconds[chunk.index] = time.perf_counter() - started
                pending[chunk.index] = pool.submit(
                    _encrypt_chunk, chunk, encryption_key, mac_key, video_id, key_version
                )
            if pending:
                index = min(pending)
                future = pending.pop(index)
                yield (*future.result(), read_seconds.pop(index, 0.0))


def _decrypt_chunk(
    record: Dict[str, object],
    stored_directory: Path,
    encryption_key: bytes,
    mac_key: bytes,
    video_id: str,
    key_version: str,
) -> Tuple[Dict[str, object], bytes, float, Dict[str, float]]:
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
    read_seconds = time.perf_counter() - read_started
    crypto_stats: Dict[str, float] = {}
    plaintext = sm4_decrypt(
        encrypted,
        encryption_key,
        mac_key,
        associated_data=_context(video_id, index, key_version),
        stats=crypto_stats,
    )
    return record, plaintext, read_seconds, crypto_stats


def _iter_parallel_decrypted(
    records: List[Dict[str, object]],
    stored_directory: Path,
    encryption_key: bytes,
    mac_key: bytes,
    video_id: str,
    key_version: str,
    workers: int,
) -> Iterator[Tuple[Dict[str, object], bytes, float, Dict[str, float]]]:
    if workers == 1:
        for record in records:
            yield _decrypt_chunk(record, stored_directory, encryption_key, mac_key, video_id, key_version)
        return
    pending: Dict[int, Future[Tuple[Dict[str, object], bytes, float, Dict[str, float]]]] = {}
    next_submit = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="facecheck-sm4") as pool:
        while pending or next_submit < len(records):
            while next_submit < len(records) and len(pending) < workers * 2:
                record = records[next_submit]
                index = int(record.get("index", next_submit)) if isinstance(record, dict) else next_submit
                pending[index] = pool.submit(
                    _decrypt_chunk, record, stored_directory, encryption_key, mac_key, video_id, key_version
                )
                next_submit += 1
            index = min(pending)
            future = pending.pop(index)
            yield future.result()


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
    workers: int = 1,
) -> StoredVideo:
    if not source.is_file():
        raise StorageError(f"source video does not exist: {source}")
    if not key_version:
        raise StorageError("key_version must not be empty")
    workers = _validate_workers(workers)
    encryption_key, mac_key = derive_storage_keys(
        encryption_key, mac_key, key_version
    )
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
        for chunk, encrypted, crypto_stats, read_seconds in _iter_parallel_encrypted(
            iterator, encryption_key, mac_key, video_id, key_version, workers
        ):
            if stats is not None:
                stats["file_read_seconds"] = stats.get("file_read_seconds", 0.0) + (
                    read_seconds
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
            "key_derivation": KEY_DERIVATION,
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


def store_enveloped_video(
    source: Path,
    storage_root: Path,
    key_provider: VideoKeyProvider,
    *,
    chunk_size: int,
    key_version: str,
    video_id: Optional[str] = None,
    stats: Optional[Dict[str, float]] = None,
    workers: int = 1,
) -> StoredVideo:
    """Store one v2 video using random per-video key material wrapped by SM2.

    The v1 chunk writer is reused unchanged in a private staging directory.
    Only the authenticated v2 manifest is atomically published at the final
    object path; an interrupted write cannot expose a partial v1 object.
    """
    if not source.is_file():
        raise StorageError(f"source video does not exist: {source}")
    if not isinstance(key_version, str) or not key_version:
        raise StorageError("key_version must not be empty")
    _validate_v2_key_version(key_version)
    workers = _validate_workers(workers)
    selected_video_id = video_id or uuid.uuid4().hex
    if not VIDEO_ID_PATTERN.fullmatch(selected_video_id):
        raise StorageError("video_id contains invalid characters")
    if storage_root.is_symlink():
        raise StorageError("storage root must not be a symbolic link")
    storage_root.mkdir(parents=True, exist_ok=True)
    final_directory = storage_root / selected_video_id
    if final_directory.exists() or final_directory.is_symlink():
        raise StorageError(f"storage directory already exists: {final_directory}")

    key_material = secrets.token_bytes(VIDEO_KEY_MATERIAL_SIZE)
    encryption_master_key = key_material[:16]
    mac_master_key = key_material[16:]
    published = False
    try:
        with tempfile.TemporaryDirectory(prefix=".facecheck-v2-stage-", dir=storage_root) as stage:
            staged_root = Path(stage)
            staged = store_encrypted_video(
                source,
                staged_root,
                encryption_master_key,
                mac_master_key,
                chunk_size=chunk_size,
                key_version=key_version,
                video_id=selected_video_id,
                stats=stats,
                workers=workers,
            )
            key_wrap_started = time.perf_counter()
            try:
                wrapped = key_provider.wrap_video_key(
                    key_material,
                    video_id=selected_video_id,
                    key_version=key_version,
                )
            except Exception:
                raise StorageError("video key envelope creation failed") from None
            if (not isinstance(wrapped, WrappedVideoKey)
                    or wrapped.key_id != key_provider.key_id):
                raise StorageError("video key envelope provider returned invalid metadata")
            if stats is not None:
                stats["sm2_key_wrap_seconds"] = stats.get("sm2_key_wrap_seconds", 0.0) + (
                    time.perf_counter() - key_wrap_started
                )

            manifest = dict(staged.manifest)
            manifest["version"] = ENVELOPED_MANIFEST_VERSION
            manifest["key_envelope"] = wrapped.to_manifest()
            authentication_started = time.perf_counter()
            _authenticate_manifest(
                manifest, mac_master_key, selected_video_id, key_version,
            )
            if stats is not None:
                stats["manifest_authentication_seconds"] = stats.get(
                    "manifest_authentication_seconds", 0.0
                ) + (time.perf_counter() - authentication_started)

            manifest_path = staged.directory / MANIFEST_NAME
            staged_manifest = staged.directory / (MANIFEST_NAME + ".new")
            with staged_manifest.open("xb") as output:
                output.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(staged_manifest, manifest_path)
            if final_directory.exists() or final_directory.is_symlink():
                raise StorageError(f"storage directory already exists: {final_directory}")
            os.replace(staged.directory, final_directory)
            published = True
        return StoredVideo(selected_video_id, final_directory, manifest)
    except Exception:
        if published and final_directory.exists():
            for child in sorted(final_directory.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            final_directory.rmdir()
        raise


def restore_encrypted_video(
    stored_directory: Path,
    destination: Path,
    encryption_key: Optional[bytes],
    mac_key: Optional[bytes],
    stats: Optional[Dict[str, float]] = None,
    workers: int = 1,
    expected_manifest_sha256: Optional[str] = None,
    key_provider: Optional[VideoKeyProvider] = None,
    expected_video_id: Optional[str] = None,
) -> Path:
    manifest_path = stored_directory / MANIFEST_NAME
    chunks_directory = stored_directory / CHUNKS_DIR_NAME
    workers = _validate_workers(workers)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StorageError(f"manifest is missing: {manifest_path}")
    if chunks_directory.is_symlink() or not chunks_directory.is_dir():
        raise StorageError(f"chunks directory is missing: {chunks_directory}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        if expected_manifest_sha256 is not None:
            if (not isinstance(expected_manifest_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
                    or hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256):
                raise StorageError("manifest does not match the trusted task anchor")
        # Validate and parse the same byte snapshot, avoiding a check/read race.
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        video_id = manifest["video_id"]
        key_version = manifest["key_version"]
        manifest_version = manifest["version"]
        if (not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id)
                or not isinstance(key_version, str) or not key_version):
            raise StorageError("manifest is invalid")
        if expected_video_id is not None and video_id != expected_video_id:
            raise StorageError("stored video identity does not match the trusted task")
        if manifest_version == ENVELOPED_MANIFEST_VERSION:
            _validate_v2_key_version(key_version)
            if (key_provider is None or not isinstance(expected_video_id, str)
                    or not VIDEO_ID_PATTERN.fullmatch(expected_video_id)):
                raise StorageError("v2 restore requires a key provider and trusted video ID")
            try:
                wrapped_key = WrappedVideoKey.from_manifest(manifest.get("key_envelope"))
                key_material = key_provider.unwrap_video_key(
                    wrapped_key, video_id=video_id, key_version=key_version,
                )
            except Exception:
                raise StorageError("video key envelope could not be opened") from None
            if not isinstance(key_material, bytes) or len(key_material) != VIDEO_KEY_MATERIAL_SIZE:
                raise StorageError("video key envelope could not be opened")
            encryption_key, mac_key = key_material[:16], key_material[16:]
            _verify_manifest_authentication(manifest, mac_key, video_id, key_version)
        elif manifest_version == MANIFEST_VERSION:
            if key_provider is not None:
                raise StorageError("v1 restore does not accept a video key provider")
            if not isinstance(encryption_key, bytes) or not isinstance(mac_key, bytes):
                raise StorageError("v1 restore requires encryption and MAC keys")
        else:
            raise StorageError("unsupported manifest version")

        records = manifest["chunks"]
        expected_count = manifest["chunk_count"]
        key_derivation = manifest.get("key_derivation", LEGACY_KEY_DERIVATION)
        chunk_size = manifest["chunk_size"]
        expected_size = manifest["source_size"]
        expected_digest = manifest["source_sha256"]
        if (
            not isinstance(chunk_size, int)
            or chunk_size <= 0
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or key_derivation not in {KEY_DERIVATION, LEGACY_KEY_DERIVATION}
            or (manifest_version == ENVELOPED_MANIFEST_VERSION
                and key_derivation != KEY_DERIVATION)
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
    if key_derivation == KEY_DERIVATION:
        encryption_key, mac_key = derive_storage_keys(
            encryption_key, mac_key, key_version
        )
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
            for record, plaintext, read_seconds, crypto_stats in _iter_parallel_decrypted(
                records, stored_directory, encryption_key, mac_key, video_id, key_version, workers
            ):
                if stats is not None:
                    stats["ciphertext_read_seconds"] = stats.get(
                        "ciphertext_read_seconds", 0.0
                    ) + read_seconds
                if stats is not None:
                    for name, value in crypto_stats.items():
                        stats[name] = stats.get(name, 0.0) + value
                if len(plaintext) != record["size"]:
                    raise StorageError(f"chunk size mismatch: {record['index']}")
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


def restore_enveloped_video(
    stored_directory: Path,
    destination: Path,
    key_provider: VideoKeyProvider,
    *,
    expected_video_id: str,
    stats: Optional[Dict[str, float]] = None,
    workers: int = 1,
    expected_manifest_sha256: Optional[str] = None,
) -> Path:
    """Restore v2 storage using a provider and an external object identity."""
    return restore_encrypted_video(
        stored_directory,
        destination,
        None,
        None,
        stats=stats,
        workers=workers,
        expected_manifest_sha256=expected_manifest_sha256,
        key_provider=key_provider,
        expected_video_id=expected_video_id,
    )


def restore_video_object(
    stored_directory: Path,
    destination: Path,
    encryption_key: Optional[bytes],
    mac_key: Optional[bytes],
    *,
    expected_video_id: str,
    expected_manifest_sha256: Optional[str] = None,
    key_provider_resolver: Optional[Callable[[str], VideoKeyProvider]] = None,
    stats: Optional[Dict[str, float]] = None,
    workers: int = 1,
) -> Path:
    """Restore a registered v1 or v2 object without format downgrade fallback."""
    manifest_path = stored_directory / MANIFEST_NAME
    if (stored_directory.is_symlink() or manifest_path.is_symlink()
            or not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024):
        raise StorageError("stored manifest is unavailable or unsafe")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise StorageError("stored manifest is invalid") from None
    if not isinstance(manifest, dict) or manifest.get("video_id") != expected_video_id:
        raise StorageError("stored video identity does not match the trusted task")
    version = manifest.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise StorageError("unsupported manifest version")
    if version == MANIFEST_VERSION:
        if not isinstance(encryption_key, bytes) or not isinstance(mac_key, bytes):
            raise StorageError("v1 restore requires encryption and MAC keys")
        return restore_encrypted_video(
            stored_directory, destination, encryption_key, mac_key,
            stats=stats, workers=workers,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_video_id=expected_video_id,
        )
    if version == ENVELOPED_MANIFEST_VERSION:
        if key_provider_resolver is None:
            raise StorageError("v2 key provider directory is unavailable")
        try:
            wrapped = WrappedVideoKey.from_manifest(manifest.get("key_envelope"))
            provider = key_provider_resolver(wrapped.key_id)
        except Exception:
            raise StorageError("video key is unavailable") from None
        if provider.key_id != wrapped.key_id:
            raise StorageError("video key directory returned the wrong key")
        return restore_enveloped_video(
            stored_directory, destination, provider,
            expected_video_id=expected_video_id,
            expected_manifest_sha256=expected_manifest_sha256,
            stats=stats, workers=workers,
        )
    raise StorageError("unsupported manifest version")


def rewrap_enveloped_video(
    stored_directory: Path,
    target_storage_root: Path,
    old_key_provider: VideoKeyProvider,
    new_key_provider: VideoKeyProvider,
    *,
    expected_video_id: str,
    expected_manifest_sha256: Optional[str] = None,
    stats: Optional[Dict[str, float]] = None,
    workers: int = 1,
) -> StoredVideo:
    """Copy one v2 object and rewrap its per-video keys without re-encrypting chunks.

    The original object is left untouched. The returned copy is fully restored
    and SHA-256 checked before it is published under ``target_storage_root``.
    The caller remains responsible for switching the external object reference.
    """
    workers = _validate_workers(workers)
    if not isinstance(expected_video_id, str) or not VIDEO_ID_PATTERN.fullmatch(expected_video_id):
        raise StorageError("trusted video identifier is invalid")
    if stored_directory.is_symlink() or not stored_directory.is_dir():
        raise StorageError("stored video directory is invalid")
    if target_storage_root.is_symlink():
        raise StorageError("target storage root must not be a symbolic link")
    source_real = stored_directory.resolve()
    target_real = target_storage_root.resolve()
    try:
        target_real.relative_to(source_real)
        target_inside_source = True
    except ValueError:
        target_inside_source = False
    if source_real == target_real or target_inside_source:
        raise StorageError("rewrap target must use a separate storage root")
    manifest_path = stored_directory / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StorageError("manifest is missing")
    try:
        manifest_bytes = manifest_path.read_bytes()
        if (expected_manifest_sha256 is not None
                and (not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
                     or hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256)):
            raise StorageError("manifest does not match the trusted task anchor")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        video_id = manifest.get("video_id")
        key_version = manifest.get("key_version")
        if (manifest.get("version") != ENVELOPED_MANIFEST_VERSION
                or video_id != expected_video_id
                or not isinstance(key_version, str) or not key_version):
            raise StorageError("stored object is not the expected v2 video")
        _validate_v2_key_version(key_version)
        wrapped_key = WrappedVideoKey.from_manifest(manifest.get("key_envelope"))
        key_material = old_key_provider.unwrap_video_key(
            wrapped_key, video_id=video_id, key_version=key_version,
        )
        if not isinstance(key_material, bytes) or len(key_material) != VIDEO_KEY_MATERIAL_SIZE:
            raise StorageError("video key envelope could not be opened")
        mac_master_key = key_material[16:]
        _verify_manifest_authentication(manifest, mac_master_key, video_id, key_version)
    except StorageError:
        raise
    except Exception:
        raise StorageError("video key envelope could not be opened") from None

    if any(path.is_symlink() for path in stored_directory.rglob("*")):
        raise StorageError("symbolic links are not allowed in stored video objects")
    target_storage_root.mkdir(parents=True, exist_ok=True)
    final_directory = target_storage_root / video_id
    if final_directory.exists() or final_directory.is_symlink():
        raise StorageError(f"storage directory already exists: {final_directory}")

    started = time.perf_counter()
    try:
        new_wrapped_key = new_key_provider.wrap_video_key(
            key_material, video_id=video_id, key_version=key_version,
        )
    except Exception:
        raise StorageError("video key rewrapping failed") from None
    if (not isinstance(new_wrapped_key, WrappedVideoKey)
            or new_wrapped_key.key_id != new_key_provider.key_id):
        raise StorageError("video key envelope provider returned invalid metadata")
    if stats is not None:
        stats["sm2_key_rewrap_seconds"] = stats.get("sm2_key_rewrap_seconds", 0.0) + (
            time.perf_counter() - started
        )

    rotated_manifest = dict(manifest)
    rotated_manifest["key_envelope"] = new_wrapped_key.to_manifest()
    _authenticate_manifest(rotated_manifest, mac_master_key, video_id, key_version)
    published = False
    with tempfile.TemporaryDirectory(prefix=".facecheck-v2-rewrap-", dir=target_storage_root) as stage:
        stage_root = Path(stage)
        staged_directory = stage_root / video_id
        try:
            shutil.copytree(stored_directory, staged_directory, symlinks=False)
            staged_manifest = staged_directory / (MANIFEST_NAME + ".new")
            with staged_manifest.open("xb") as output:
                output.write(json.dumps(rotated_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(staged_manifest, staged_directory / MANIFEST_NAME)
            verification_path = stage_root / "rewrap-verify.bin"
            new_manifest_digest = hashlib.sha256(
                (staged_directory / MANIFEST_NAME).read_bytes()
            ).hexdigest()
            restore_enveloped_video(
                staged_directory,
                verification_path,
                new_key_provider,
                expected_video_id=video_id,
                expected_manifest_sha256=new_manifest_digest,
                workers=workers,
            )
            verification_path.unlink()
            if final_directory.exists() or final_directory.is_symlink():
                raise StorageError(f"storage directory already exists: {final_directory}")
            os.replace(staged_directory, final_directory)
            published = True
        except Exception:
            if published and final_directory.exists():
                shutil.rmtree(final_directory)
            raise
    return StoredVideo(video_id, final_directory, rotated_manifest)


def rotate_encrypted_video_key(
    stored_directory: Path,
    storage_root: Path,
    old_encryption_key: bytes,
    old_mac_key: bytes,
    new_encryption_key: bytes,
    new_mac_key: bytes,
    *,
    new_key_version: str,
    chunk_size: Optional[int] = None,
    video_id: Optional[str] = None,
    workers: int = 1,
    stats: Optional[Dict[str, float]] = None,
) -> StoredVideo:
    """Re-encrypt one stored video under a new versioned key.

    The original directory is never modified. A temporary authenticated
    restore is followed by a new encrypted store; failed migration removes
    only temporary/new output and leaves the original ciphertext usable.
    """
    workers = _validate_workers(workers)
    if not isinstance(new_key_version, str) or not new_key_version:
        raise StorageError("new_key_version must not be empty")
    manifest_path = stored_directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_chunk_size = int(manifest["chunk_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageError("manifest is invalid") from exc
    selected_chunk_size = chunk_size if chunk_size is not None else old_chunk_size
    if selected_chunk_size <= 0:
        raise StorageError("chunk_size must be greater than zero")
    migration_stats: Dict[str, float] = {}
    if stats is not None:
        migration_stats = stats
    with tempfile.TemporaryDirectory(prefix="faceguard-rotate-") as temporary:
        restored = Path(temporary) / "restored.video"
        restore_encrypted_video(
            stored_directory,
            restored,
            old_encryption_key,
            old_mac_key,
            stats=migration_stats,
            workers=workers,
        )
        rotated = store_encrypted_video(
            restored,
            storage_root,
            new_encryption_key,
            new_mac_key,
            chunk_size=selected_chunk_size,
            key_version=new_key_version,
            video_id=video_id,
            stats=migration_stats,
            workers=workers,
        )
        verification = Path(temporary) / "rotated-verify.video"
        try:
            restore_encrypted_video(
                rotated.directory,
                verification,
                new_encryption_key,
                new_mac_key,
                workers=workers,
            )
        except Exception:
            for child in sorted(rotated.directory.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            rotated.directory.rmdir()
            raise
        return rotated
