"""Restartable, file-backed coordinator for small v2 recipient-key rotations.

This module creates verified target objects and journals progress. It does not
switch application/database references, delete source objects, revoke old
keys, or replace a production task queue/KMS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .key_management import VideoKeyProvider
from .storage import (
    MANIFEST_NAME,
    MAX_CRYPTO_WORKERS,
    rewrap_enveloped_video,
    restore_enveloped_video,
)

JOURNAL_VERSION = 1


class KeyRotationError(ValueError):
    """Raised when a rotation plan or its persistent journal is inconsistent."""


@dataclass(frozen=True)
class EnvelopedRotationItem:
    video_id: str
    source_directory: Path
    source_manifest_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    if path.is_symlink():
        raise KeyRotationError("rotation journal must not be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(journal, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _verify_target(
    target_directory: Path,
    target_root: Path,
    new_provider: VideoKeyProvider,
    item: EnvelopedRotationItem,
    expected_manifest_sha256: str | None,
) -> str:
    manifest_path = target_directory / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise KeyRotationError("rotation target manifest is unavailable")
    manifest_hash = _file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise KeyRotationError("completed rotation target manifest changed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise KeyRotationError("rotation target manifest is invalid") from None
    source_manifest_path = item.source_directory / MANIFEST_NAME
    if (source_manifest_path.is_symlink() or not source_manifest_path.is_file()
            or _file_sha256(source_manifest_path) != item.source_manifest_sha256):
        raise KeyRotationError("rotation source manifest changed")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise KeyRotationError("rotation source manifest is invalid") from None
    if (manifest.get("video_id") != item.video_id
            or manifest.get("key_envelope", {}).get("key_id") != new_provider.key_id
            or manifest.get("source_sha256") != source_manifest.get("source_sha256")
            or manifest.get("source_size") != source_manifest.get("source_size")):
        raise KeyRotationError("rotation target metadata does not match the plan")

    with tempfile.TemporaryDirectory(prefix=".facecheck-rotation-check-", dir=target_root) as temp:
        destination = Path(temp) / "verify.bin"
        try:
            restore_enveloped_video(
                target_directory,
                destination,
                new_provider,
                expected_video_id=item.video_id,
                expected_manifest_sha256=manifest_hash,
            )
        except Exception:
            raise KeyRotationError("rotation target integrity check failed") from None
    return manifest_hash


def rotate_enveloped_batch(
    items: Sequence[EnvelopedRotationItem],
    target_storage_root: Path,
    old_key_provider: VideoKeyProvider,
    new_key_provider: VideoKeyProvider,
    *,
    journal_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Create verified rewrapped copies and resume from an atomic JSON journal.

    Sources remain untouched. A completed item is revalidated when a journal
    is resumed; an object published just before a process interruption is
    detected and verified instead of being overwritten.
    """
    if old_key_provider.key_id == new_key_provider.key_id:
        raise KeyRotationError("old and new recipient key IDs must differ")
    if (isinstance(workers, bool) or not isinstance(workers, int)
            or workers < 1 or workers > MAX_CRYPTO_WORKERS):
        raise KeyRotationError(f"workers must be between 1 and {MAX_CRYPTO_WORKERS}")
    if not items:
        raise KeyRotationError("rotation batch must contain at least one item")
    if target_storage_root.is_symlink():
        raise KeyRotationError("rotation target root must not be a symbolic link")
    if journal_path.is_symlink():
        raise KeyRotationError("rotation journal must not be a symbolic link")

    ordered_items = sorted(items, key=lambda item: item.video_id)
    item_ids = [item.video_id for item in ordered_items]
    if len(item_ids) != len(set(item_ids)):
        raise KeyRotationError("rotation batch contains duplicate video IDs")
    item_plan = []
    for item in ordered_items:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", item.video_id):
            raise KeyRotationError("rotation batch contains an invalid video ID")
        if not re.fullmatch(r"[0-9a-f]{64}", item.source_manifest_sha256):
            raise KeyRotationError("source manifest digest is invalid")
        item_plan.append({
            "video_id": item.video_id,
            "source_manifest_sha256": item.source_manifest_sha256,
        })
    target_root = target_storage_root.resolve()
    journal_real = journal_path.resolve()
    for item in ordered_items:
        source_real = item.source_directory.resolve()
        for write_path in (target_root, journal_real):
            try:
                write_path.relative_to(source_real)
            except ValueError:
                continue
            raise KeyRotationError("rotation outputs must not be written inside a source object")
    batch_fingerprint = {
        "version": JOURNAL_VERSION,
        "old_key_id": old_key_provider.key_id,
        "new_key_id": new_key_provider.key_id,
        "target_root": str(target_root),
        "items": item_plan,
    }
    batch_id = hashlib.sha256(_canonical_json(batch_fingerprint)).hexdigest()

    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise KeyRotationError("rotation journal is invalid") from None
        if (not isinstance(journal, dict)
                or journal.get("version") != JOURNAL_VERSION
                or journal.get("batch_id") != batch_id
                or journal.get("old_key_id") != old_key_provider.key_id
                or journal.get("new_key_id") != new_key_provider.key_id
                or journal.get("target_root") != str(target_root)
                or journal.get("item_plan") != item_plan
                or not isinstance(journal.get("items"), dict)
                or set(journal["items"]) != set(item_ids)):
            raise KeyRotationError("rotation journal does not match the requested plan")
    else:
        journal = {
            "version": JOURNAL_VERSION,
            "batch_id": batch_id,
            "status": "planned",
            "old_key_id": old_key_provider.key_id,
            "new_key_id": new_key_provider.key_id,
            "target_root": str(target_root),
            "item_plan": item_plan,
            "items": {
                item.video_id: {
                    "state": "pending",
                    "source_manifest_sha256": item.source_manifest_sha256,
                    "attempts": 0,
                }
                for item in ordered_items
            },
        }
        _write_journal(journal_path, journal)

    target_storage_root.mkdir(parents=True, exist_ok=True)
    journal["status"] = "running"
    _write_journal(journal_path, journal)

    for item in ordered_items:
        record = journal["items"].get(item.video_id)
        if not isinstance(record, dict):
            raise KeyRotationError("rotation journal item is invalid")
        source_manifest = item.source_directory / MANIFEST_NAME
        if (source_manifest.is_symlink() or not source_manifest.is_file()
                or _file_sha256(source_manifest) != item.source_manifest_sha256):
            record["state"] = "failed"
            record["error_type"] = "SourceManifestMismatch"
            journal["status"] = "partial"
            _write_journal(journal_path, journal)
            continue

        target_directory = target_storage_root / item.video_id
        if record.get("state") == "complete":
            _verify_target(
                target_directory, target_storage_root, new_key_provider, item,
                record.get("target_manifest_sha256"),
            )
            continue

        if target_directory.exists() or target_directory.is_symlink():
            try:
                target_hash = _verify_target(
                    target_directory, target_storage_root, new_key_provider, item, None,
                )
                record["state"] = "complete"
                record["target_manifest_sha256"] = target_hash
                record.pop("error_type", None)
                _write_journal(journal_path, journal)
                continue
            except Exception:
                record["state"] = "failed"
                record["error_type"] = "ExistingTargetVerificationFailed"
                journal["status"] = "partial"
                _write_journal(journal_path, journal)
                continue

        record["state"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record.pop("error_type", None)
        _write_journal(journal_path, journal)
        try:
            rotated = rewrap_enveloped_video(
                item.source_directory,
                target_storage_root,
                old_key_provider,
                new_key_provider,
                expected_video_id=item.video_id,
                expected_manifest_sha256=item.source_manifest_sha256,
                workers=workers,
            )
            target_manifest = rotated.directory / MANIFEST_NAME
            target_hash = _verify_target(
                rotated.directory, target_storage_root, new_key_provider, item,
                _file_sha256(target_manifest),
            )
            record["state"] = "complete"
            record["target_manifest_sha256"] = target_hash
            journal["status"] = "running"
        except Exception as exc:
            record["state"] = "failed"
            record["error_type"] = type(exc).__name__
            journal["status"] = "partial"
        _write_journal(journal_path, journal)

    journal["status"] = (
        "complete" if all(record.get("state") == "complete"
                           for record in journal["items"].values()) else "partial"
    )
    _write_journal(journal_path, journal)
    return journal
