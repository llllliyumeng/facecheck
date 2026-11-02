"""SQLite-backed rotation queue and compare-and-swap object reference switch.

This is a single-database deployment primitive. It records inventory and cache/
backup dispositions, but an external KMS/HSM and the configured backup owners
remain responsible for production key revocation and complete inventory proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .database import TaskRepository, utc_now
from .feature_store import MAX_CACHE_BYTES, FeatureStore
from .key_management import VideoKeyProvider
from .storage import (
    ENVELOPED_MANIFEST_VERSION,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    restore_encrypted_video,
    rewrap_enveloped_video,
    restore_enveloped_video,
    store_enveloped_video,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KINDS = {"feature_cache", "backup", "replica"}


class RotationControlError(ValueError):
    """Raised when a rotation plan or state transition is unsafe."""


class RotationControlRepository:
    """Durable work queue tied to task rows in the same SQLite database."""

    def __init__(self, database: Path):
        self.database = Path(database).resolve()
        self.tasks = TaskRepository(self.database)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS video_key_rotation_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_key_id TEXT NOT NULL,
                    target_key_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued','running','partial','completed','ready_to_revoke','revoking','revoked')),
                    inventory_sha256 TEXT,
                    inventory_complete INTEGER NOT NULL DEFAULT 0,
                    writers_fenced INTEGER NOT NULL DEFAULT 0,
                    readers_drained INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revocation_receipt TEXT
                );
                CREATE TABLE IF NOT EXISTS video_key_rotation_items (
                    job_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    source_storage_id TEXT NOT NULL,
                    source_manifest_sha256 TEXT NOT NULL,
                    source_key_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN
                        ('queued','running','retryable','switched','stale')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    target_storage_id TEXT,
                    target_manifest_sha256 TEXT,
                    target_key_version TEXT,
                    error_type TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, task_id),
                    FOREIGN KEY(job_id) REFERENCES video_key_rotation_jobs(job_id),
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_video_rotation_queue
                    ON video_key_rotation_items(job_id, state, updated_at);
                CREATE TABLE IF NOT EXISTS video_key_rotation_artifacts (
                    job_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    task_id TEXT,
                    kind TEXT NOT NULL CHECK(kind IN ('feature_cache','backup','replica')),
                    source_key_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_sha256 TEXT,
                    state TEXT NOT NULL CHECK(state IN
                        ('pending','migrated','deleted','failed')),
                    target_ref TEXT,
                    target_sha256 TEXT,
                    evidence TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, artifact_id),
                    FOREIGN KEY(job_id) REFERENCES video_key_rotation_jobs(job_id)
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_plan(
        self,
        source_key_id: str,
        target_key_id: str,
        items: Sequence[Mapping[str, str]],
        *,
        artifacts: Sequence[Mapping[str, str]] = (),
        inventory_sha256: str | None = None,
        inventory_complete: bool = False,
        writers_fenced: bool = False,
        readers_drained: bool = False,
        actor_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        self._validate_key_id(source_key_id)
        self._validate_key_id(target_key_id)
        if source_key_id == target_key_id:
            raise RotationControlError("source and target key IDs must differ")
        if not items and not artifacts:
            raise RotationControlError("rotation plan must include a video or key-bearing artifact")
        if inventory_sha256 is not None and not _SHA256.fullmatch(inventory_sha256):
            raise RotationControlError("inventory digest is invalid")
        if any(not isinstance(item, Mapping) for item in items):
            raise RotationControlError("rotation plan contains invalid video metadata")
        normalized_items = sorted(items, key=lambda item: str(item.get("task_id", "")))
        task_ids = [item.get("task_id") for item in normalized_items]
        if (any(not isinstance(task_id, str) or not _ID.fullmatch(task_id) for task_id in task_ids)
                or len(task_ids) != len(set(task_ids))):
            raise RotationControlError("rotation plan contains invalid or duplicate task IDs")
        normalized_artifacts = list(artifacts)
        for item in normalized_items:
            task_id = item["task_id"]
            storage_id = item.get("source_storage_id", "")
            manifest_sha = item.get("source_manifest_sha256", "")
            normalized_artifacts.append({
                "artifact_id": "source-" + hashlib.sha256(task_id.encode()).hexdigest()[:24],
                "task_id": task_id,
                "kind": "replica",
                "source_key_id": source_key_id,
                "source_ref": storage_id,
                "source_sha256": manifest_sha,
            })
        if any(not isinstance(artifact, Mapping) for artifact in normalized_artifacts):
            raise RotationControlError("rotation plan contains invalid artifact metadata")
        artifact_ids = [artifact.get("artifact_id") for artifact in normalized_artifacts]
        if (any(not isinstance(value, str) or not _ID.fullmatch(value) for value in artifact_ids)
                or len(artifact_ids) != len(set(artifact_ids))):
            raise RotationControlError("rotation plan contains invalid or duplicate artifact IDs")
        identifier = job_id or uuid.uuid4().hex
        if not _ID.fullmatch(identifier):
            raise RotationControlError("rotation job ID is invalid")

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT job_id FROM video_key_rotation_jobs WHERE source_key_id=? LIMIT 1",
                (source_key_id,),
            ).fetchone()
            if existing is not None:
                raise RotationControlError(
                    "a rotation job already exists for this source key ID; key IDs cannot be reused"
                )
            now = utc_now()
            db.execute(
                "INSERT INTO video_key_rotation_jobs(job_id,source_key_id,target_key_id,status,"
                "inventory_sha256,inventory_complete,writers_fenced,readers_drained,created_at,updated_at) "
                "VALUES(?,?,?,'queued',?,?,?,?,?,?)",
                (identifier, source_key_id, target_key_id, inventory_sha256,
                 int(inventory_complete),
                 int(writers_fenced), int(readers_drained), now, now),
            )
            for item in normalized_items:
                task_id = item["task_id"]
                storage_id = item.get("source_storage_id")
                manifest_sha = item.get("source_manifest_sha256")
                item_key_id = item.get("source_key_id", source_key_id)
                if (not isinstance(storage_id, str) or not _ID.fullmatch(storage_id)
                        or not isinstance(manifest_sha, str) or not _SHA256.fullmatch(manifest_sha)
                        or item_key_id != source_key_id):
                    raise RotationControlError("rotation item metadata is invalid")
                task = db.execute(
                    "SELECT status,storage_id,processing_metrics FROM tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if task is None or task["status"] != "completed" or task["storage_id"] != storage_id:
                    raise RotationControlError("rotation source no longer matches the task reference")
                metrics = self._metrics(task["processing_metrics"])
                if metrics.get("manifest_sha256") != manifest_sha:
                    raise RotationControlError("rotation source manifest anchor changed")
                recorded_key_id = metrics.get("key_id")
                if recorded_key_id is not None and recorded_key_id != source_key_id:
                    raise RotationControlError("rotation source key ID does not match the plan")
                db.execute(
                    "INSERT INTO video_key_rotation_items(job_id,task_id,source_storage_id,"
                    "source_manifest_sha256,source_key_id,state,updated_at) "
                    "VALUES(?,?,?,?,?,'queued',?)",
                    (identifier, task_id, storage_id, manifest_sha, source_key_id, now),
                )
            for artifact in normalized_artifacts:
                artifact_id = artifact["artifact_id"]
                kind = artifact.get("kind")
                artifact_key = artifact.get("source_key_id", source_key_id)
                source_ref = artifact.get("source_ref")
                digest = artifact.get("source_sha256")
                task_id = artifact.get("task_id")
                if (kind not in _ARTIFACT_KINDS or artifact_key != source_key_id
                        or not isinstance(source_ref, str) or not source_ref
                        or (digest is not None and not _SHA256.fullmatch(digest))
                        or (task_id is not None and (not isinstance(task_id, str)
                            or not _ID.fullmatch(task_id)
                            or db.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone() is None))):
                    raise RotationControlError("rotation artifact metadata is invalid")
                db.execute(
                    "INSERT INTO video_key_rotation_artifacts(job_id,artifact_id,task_id,kind,"
                    "source_key_id,source_ref,source_sha256,state,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,'pending',?)",
                    (identifier, artifact_id, task_id, kind, artifact_key, source_ref, digest, now),
                )
            self.tasks._audit(db, None, "video_key_rotation_planned", json.dumps({
                "job_id": identifier, "source_key_id": source_key_id,
                "target_key_id": target_key_id, "video_count": len(normalized_items),
                "artifact_count": len(normalized_artifacts), "inventory_sha256": inventory_sha256,
                "actor_id": actor_id,
            }, sort_keys=True))
        return identifier

    def create_plan_from_storage(
        self,
        storage_root: Path,
        source_key_id: str,
        target_key_id: str,
        *,
        feature_store: FeatureStore | None = None,
        backup_roots: Sequence[Path] = (),
        backup_inventory_complete: bool = False,
        writers_fenced: bool = False,
        readers_drained: bool = False,
        actor_id: str | None = None,
    ) -> str:
        """Build a hashed local inventory of task objects, cache files and backups.

        ``backup_inventory_complete`` is an explicit operator attestation that all
        backup and disaster-recovery locations were supplied and inspected.
        """
        self._validate_key_id(source_key_id)
        self._validate_key_id(target_key_id)
        root = Path(storage_root)
        if root.is_symlink() or not root.is_dir():
            raise RotationControlError("video storage root is unavailable or unsafe")
        with self.connect() as db:
            tasks = [dict(row) for row in db.execute(
                "SELECT task_id,status,storage_id,processing_metrics FROM tasks "
                "WHERE status='completed' ORDER BY task_id"
            )]

        entries: list[dict[str, Any]] = []
        items: list[dict[str, str]] = []
        artifacts: list[dict[str, str]] = []
        referenced_storage: set[str] = set()
        task_by_storage: dict[str, str] = {}
        known_task_ids = {task["task_id"] for task in tasks}
        for task in tasks:
            storage_id = task.get("storage_id")
            if not isinstance(storage_id, str) or not _ID.fullmatch(storage_id):
                raise RotationControlError("completed task has an invalid storage reference")
            directory = root / storage_id
            manifest, digest = self._read_manifest(directory)
            metrics = self._metrics(task.get("processing_metrics"))
            if (manifest.get("video_id") != task["task_id"]
                    or metrics.get("manifest_sha256") != digest):
                raise RotationControlError("task object does not match its trusted manifest anchor")
            key_id = self._manifest_key_id(manifest, metrics)
            referenced_storage.add(storage_id)
            task_by_storage[storage_id] = task["task_id"]
            entries.append({"kind": "task_object", "task_id": task["task_id"],
                            "storage_id": storage_id, "key_id": key_id,
                            "manifest_sha256": digest})
            if key_id == source_key_id:
                items.append({"task_id": task["task_id"], "source_storage_id": storage_id,
                              "source_manifest_sha256": digest,
                              "source_key_id": source_key_id})

        # A storage copy with no task reference is still a key-bearing replica.
        storage_inventory_complete = True
        for directory in sorted(root.iterdir()):
            if directory.name.startswith("."):
                # Hidden staging/temporary entries can contain an interrupted
                # migration or plaintext left by forced process termination.
                # They are deliberately not declared inventoried by this scan.
                storage_inventory_complete = False
                entries.append({"kind": "unreviewed_storage_entry", "reference": str(directory)})
                continue
            if directory.name in referenced_storage:
                continue
            if directory.is_symlink():
                storage_inventory_complete = False
                entries.append({"kind": "unsafe_storage_entry", "reference": str(directory)})
                continue
            if not directory.is_dir():
                storage_inventory_complete = False
                entries.append({"kind": "unreviewed_storage_entry", "reference": str(directory)})
                continue
            if not (directory / MANIFEST_NAME).exists():
                if any(directory.iterdir()):
                    storage_inventory_complete = False
                    entries.append({"kind": "unreviewed_storage_directory", "reference": str(directory)})
                continue
            manifest, digest = self._read_manifest(directory)
            key_id = self._manifest_key_id(manifest, {})
            entries.append({"kind": "unreferenced_object", "storage_id": directory.name,
                            "key_id": key_id, "manifest_sha256": digest})
            if key_id == source_key_id:
                artifacts.append({
                    "artifact_id": "replica-" + hashlib.sha256(str(directory).encode()).hexdigest()[:24],
                    "kind": "replica", "source_key_id": source_key_id,
                    "source_ref": str(directory), "source_sha256": digest,
                })

        if source_key_id == "legacy-api-v1":
            cache_root = (feature_store.root if feature_store is not None
                          else root.parent / "protected-features")
            cache_inventory = (
                feature_store.inventory_all() if feature_store is not None
                else self._inventory_cache_root(cache_root)
            )
            for storage_id, cache_files in cache_inventory.items():
                if not cache_files:
                    continue
                cache_digest = hashlib.sha256(json.dumps(
                    cache_files, sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest()
                task_id = task_by_storage.get(storage_id)
                artifact_id = "cache-" + hashlib.sha256(storage_id.encode()).hexdigest()[:24]
                cache_artifact = {
                    "artifact_id": artifact_id,
                    "kind": "feature_cache", "source_key_id": source_key_id,
                    "source_ref": str(cache_root / storage_id),
                    "source_sha256": cache_digest,
                }
                if task_id is not None:
                    cache_artifact["task_id"] = task_id
                artifacts.append(cache_artifact)
                entries.append({"kind": "feature_cache", "storage_id": storage_id,
                                "key_id": source_key_id, "sha256": cache_digest})

        backup_complete = bool(backup_inventory_complete) and storage_inventory_complete
        seen_backup_manifests: set[Path] = set()
        for backup_root in backup_roots:
            backup_root = Path(backup_root)
            if backup_root.is_symlink() or not backup_root.is_dir():
                backup_complete = False
                continue
            for manifest_path in backup_root.rglob(MANIFEST_NAME):
                if manifest_path in seen_backup_manifests:
                    continue
                seen_backup_manifests.add(manifest_path)
                if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
                    backup_complete = False
                    continue
                try:
                    manifest, digest = self._read_manifest(manifest_path.parent)
                except RotationControlError:
                    backup_complete = False
                    continue
                key_id = self._manifest_key_id(manifest, {})
                entries.append({"kind": "backup_manifest", "reference": str(manifest_path),
                                "key_id": key_id, "manifest_sha256": digest})
                if key_id == source_key_id:
                    backup_artifact = {
                        "artifact_id": "backup-" + hashlib.sha256(str(manifest_path).encode()).hexdigest()[:24],
                        "kind": "backup", "source_key_id": source_key_id,
                        "source_ref": str(manifest_path.parent), "source_sha256": digest,
                    }
                    if manifest.get("video_id") in known_task_ids:
                        backup_artifact["task_id"] = manifest["video_id"]
                    artifacts.append(backup_artifact)
        entries.sort(key=lambda value: json.dumps(value, sort_keys=True))
        inventory_digest = hashlib.sha256(json.dumps(
            {"version": 1, "storage_root": str(root.resolve()),
             "entries": entries, "backup_roots": sorted(str(Path(path).resolve()) for path in backup_roots),
             "storage_inventory_complete": storage_inventory_complete,
             "backup_inventory_complete": bool(backup_inventory_complete)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
        if not items:
            if not artifacts:
                raise RotationControlError("no task objects or artifacts use the requested source key")
        return self.create_plan(
            source_key_id, target_key_id, items, artifacts=artifacts,
            inventory_sha256=inventory_digest,
            inventory_complete=backup_complete,
            writers_fenced=writers_fenced,
            readers_drained=readers_drained,
            actor_id=actor_id,
        )

    def claim_next(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute(
                "SELECT status FROM video_key_rotation_jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] in {"revoked", "ready_to_revoke"}:
                return None
            row = db.execute(
                "SELECT * FROM video_key_rotation_items WHERE job_id=? "
                "AND state IN ('queued','retryable') ORDER BY task_id LIMIT 1",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            now = utc_now()
            db.execute(
                "UPDATE video_key_rotation_items SET state='running',attempts=attempts+1,"
                "error_type=NULL,updated_at=? WHERE job_id=? AND task_id=?",
                (now, job_id, row["task_id"]),
            )
            db.execute(
                "UPDATE video_key_rotation_jobs SET status='running',updated_at=? WHERE job_id=?",
                (now, job_id),
            )
            result = dict(row)
            result["state"] = "running"
            result["attempts"] += 1
            result["job"] = dict(db.execute(
                "SELECT * FROM video_key_rotation_jobs WHERE job_id=?", (job_id,),
            ).fetchone())
            return result

    def fail_item(self, job_id: str, task_id: str, error: BaseException) -> None:
        error_type = type(error).__name__[:100]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE video_key_rotation_items SET state='retryable',error_type=?,updated_at=? "
                "WHERE job_id=? AND task_id=? AND state='running'",
                (error_type, utc_now(), job_id, task_id),
            ).rowcount
            if changed != 1:
                raise RotationControlError("rotation item is not running")
            db.execute(
                "UPDATE video_key_rotation_jobs SET status='partial',updated_at=? WHERE job_id=?",
                (utc_now(), job_id),
            )
            self.tasks._audit(db, task_id, "video_key_rotation_item_failed", json.dumps({
                "job_id": job_id, "error_type": error_type,
            }, sort_keys=True))

    def recover_interrupted(self, job_id: str, *, worker_stopped_confirmed: bool) -> int:
        """Requeue abandoned claims only after an operator confirms the worker stopped."""
        if worker_stopped_confirmed is not True:
            raise RotationControlError("worker shutdown must be confirmed before recovery")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE video_key_rotation_items SET state='retryable',"
                "error_type='WorkerInterrupted',updated_at=? "
                "WHERE job_id=? AND state='running'",
                (utc_now(), job_id),
            ).rowcount
            if changed:
                db.execute(
                    "UPDATE video_key_rotation_jobs SET status='partial',updated_at=? WHERE job_id=?",
                    (utc_now(), job_id),
                )
                self.tasks._audit(db, None, "video_key_rotation_interrupted_items_requeued", json.dumps({
                    "job_id": job_id, "item_count": changed,
                }, sort_keys=True))
        return int(changed)

    def complete_verified_and_switch(
        self,
        job_id: str,
        task_id: str,
        *,
        target_storage_id: str,
        target_manifest_sha256: str,
        target_key_version: str,
        actor_id: str | None = "rotation-worker",
    ) -> None:
        """Atomically update task pointer, trusted manifest anchor and audit event.

        Caller must fully restore and authenticate the target object before calling.
        The source object is intentionally retained for rollback and backup policy.
        """
        if (not isinstance(target_storage_id, str) or not _ID.fullmatch(target_storage_id)
                or not isinstance(target_manifest_sha256, str)
                or not _SHA256.fullmatch(target_manifest_sha256)
                or not isinstance(target_key_version, str) or not target_key_version):
            raise RotationControlError("verified target metadata is invalid")
        source_changed = False
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute(
                "SELECT * FROM video_key_rotation_jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            item = db.execute(
                "SELECT * FROM video_key_rotation_items WHERE job_id=? AND task_id=?",
                (job_id, task_id),
            ).fetchone()
            task = db.execute(
                "SELECT status,storage_id,processing_metrics FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if job is None or item is None or task is None:
                raise RotationControlError("rotation job, item or task is unavailable")
            if item["state"] != "running":
                raise RotationControlError("rotation item is not running")
            metrics = self._metrics(task["processing_metrics"])
            if (task["status"] != "completed"
                    or task["storage_id"] != item["source_storage_id"]
                    or metrics.get("manifest_sha256") != item["source_manifest_sha256"]):
                db.execute(
                    "UPDATE video_key_rotation_items SET state='stale',error_type='SourceReferenceChanged',updated_at=? "
                    "WHERE job_id=? AND task_id=?",
                    (utc_now(), job_id, task_id),
                )
                db.execute(
                    "UPDATE video_key_rotation_jobs SET status='partial',updated_at=? WHERE job_id=?",
                    (utc_now(), job_id),
                )
                source_changed = True
            else:
                metrics.update({
                    "manifest_sha256": target_manifest_sha256,
                    "key_id": job["target_key_id"],
                    "key_version": target_key_version,
                    "last_rotation_job_id": job_id,
                })
                changed = db.execute(
                    "UPDATE tasks SET storage_id=?,processing_metrics=?,updated_at=? "
                    "WHERE task_id=? AND status='completed' AND storage_id=?",
                    (target_storage_id, json.dumps(metrics, sort_keys=True), utc_now(),
                     task_id, item["source_storage_id"]),
                ).rowcount
                if changed != 1:
                    raise RotationControlError("atomic task reference switch failed")
                db.execute(
                    "UPDATE video_key_rotation_items SET state='switched',target_storage_id=?,"
                    "target_manifest_sha256=?,target_key_version=?,updated_at=? "
                    "WHERE job_id=? AND task_id=?",
                    (target_storage_id, target_manifest_sha256, target_key_version,
                     utc_now(), job_id, task_id),
                )
                remaining = db.execute(
                    "SELECT COUNT(*) FROM video_key_rotation_items WHERE job_id=? AND state!='switched'",
                    (job_id,),
                ).fetchone()[0]
                state = "completed" if remaining == 0 else "running"
                db.execute(
                    "UPDATE video_key_rotation_jobs SET status=?,updated_at=? WHERE job_id=?",
                    (state, utc_now(), job_id),
                )
                self.tasks._audit(db, task_id, "video_key_rotation_reference_switched", json.dumps({
                    "job_id": job_id, "source_storage_id": item["source_storage_id"],
                    "target_storage_id": target_storage_id,
                    "source_manifest_sha256": item["source_manifest_sha256"],
                    "target_manifest_sha256": target_manifest_sha256,
                    "target_key_id": job["target_key_id"],
                    "actor_id": actor_id,
                }, sort_keys=True))
        if source_changed:
            raise RotationControlError("source task reference changed during rotation")

    def finish_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        disposition: str,
        evidence: str,
        target_ref: str | None = None,
        target_sha256: str | None = None,
        actor_id: str | None = None,
    ) -> None:
        if disposition not in {"migrated", "deleted"} or not evidence or len(evidence) > 500:
            raise RotationControlError("artifact disposition is invalid")
        if disposition == "migrated":
            if not target_ref or target_sha256 is None or not _SHA256.fullmatch(target_sha256):
                raise RotationControlError("migrated artifact requires a target and digest")
        elif target_ref is not None or target_sha256 is not None:
            raise RotationControlError("deleted artifact cannot have a migration target")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            artifact = db.execute(
                "SELECT task_id,source_key_id FROM video_key_rotation_artifacts "
                "WHERE job_id=? AND artifact_id=? AND state IN ('pending','failed')",
                (job_id, artifact_id),
            ).fetchone()
            if artifact is None:
                raise RotationControlError("rotation artifact is unavailable")
            if artifact["task_id"] is not None:
                item = db.execute(
                    "SELECT state FROM video_key_rotation_items WHERE job_id=? AND task_id=?",
                    (job_id, artifact["task_id"]),
                ).fetchone()
                task = db.execute(
                    "SELECT status,processing_metrics FROM tasks WHERE task_id=?",
                    (artifact["task_id"],),
                ).fetchone()
                current_key_id = (self._metrics(task["processing_metrics"]).get("key_id")
                                  if task is not None else None)
                if ((item is not None and item["state"] != "switched")
                        or (task is not None and task["status"] == "completed"
                            and (current_key_id or "legacy-api-v1") == artifact["source_key_id"])):
                    raise RotationControlError(
                        "task must reference the target key before its old artifact is resolved"
                    )
            changed = db.execute(
                "UPDATE video_key_rotation_artifacts SET state=?,target_ref=?,target_sha256=?,"
                "evidence=?,updated_at=? WHERE job_id=? AND artifact_id=? AND state IN ('pending','failed')",
                (disposition, target_ref, target_sha256, evidence, utc_now(), job_id, artifact_id),
            ).rowcount
            if changed != 1:
                raise RotationControlError("rotation artifact is unavailable")
            self.tasks._audit(db, None, "video_key_rotation_artifact_resolved", json.dumps({
                "job_id": job_id, "artifact_id": artifact_id,
                "disposition": disposition, "evidence": evidence, "actor_id": actor_id,
            }, sort_keys=True))

    def revocation_readiness(self, job_id: str) -> dict[str, Any]:
        with self.connect() as db:
            return self._revocation_readiness(db, job_id)

    def _revocation_readiness(self, db: sqlite3.Connection, job_id: str) -> dict[str, Any]:
        job = db.execute(
            "SELECT * FROM video_key_rotation_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        if job is None:
            raise KeyError(job_id)
        pending_items = db.execute(
            "SELECT COUNT(*) FROM video_key_rotation_items WHERE job_id=? AND state!='switched'",
            (job_id,),
        ).fetchone()[0]
        pending_artifacts = db.execute(
            "SELECT COUNT(*) FROM video_key_rotation_artifacts WHERE job_id=? AND state NOT IN ('migrated','deleted')",
            (job_id,),
        ).fetchone()[0]
        current_refs = []
        rows = db.execute(
            "SELECT task_id,processing_metrics FROM tasks WHERE status='completed'"
        ).fetchall()
        for row in rows:
            metrics = self._metrics(row["processing_metrics"])
            # Historic v1 rows may not have a key_id in processing_metrics.
            # Treat them as using the legacy key so a missing annotation cannot
            # make the revocation check pass.
            key_id = metrics.get("key_id") or "legacy-api-v1"
            if key_id == job["source_key_id"]:
                current_refs.append(row["task_id"])
        blockers = []
        if not job["inventory_sha256"]:
            blockers.append("complete inventory digest is missing")
        if not job["inventory_complete"]:
            blockers.append("inventory completeness has not been attested")
        if not job["writers_fenced"]:
            blockers.append("old-key writers are not fenced")
        if not job["readers_drained"]:
            blockers.append("in-flight readers and cache writers are not drained")
        if pending_items:
            blockers.append(f"{pending_items} video references are not switched")
        if pending_artifacts:
            blockers.append(f"{pending_artifacts} cache/backup/replica artifacts are unresolved")
        if current_refs:
            blockers.append(f"{len(current_refs)} current task rows still reference the old key")
        return {
            "ready": not blockers,
            "job_id": job_id,
            "source_key_id": job["source_key_id"],
            "target_key_id": job["target_key_id"],
            "inventory_sha256": job["inventory_sha256"],
            "writers_fenced": bool(job["writers_fenced"]),
            "readers_drained": bool(job["readers_drained"]),
            "pending_video_references": int(pending_items),
            "unresolved_artifacts": int(pending_artifacts),
            "current_database_references": current_refs,
            "blockers": blockers,
            "production_kms_verified": False,
            "external_backup_inventory_verified": False,
        }

    def revoke_old_key(
        self,
        job_id: str,
        revoke: Callable[[str], str],
        *,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            readiness = self._revocation_readiness(db, job_id)
            if not readiness["ready"]:
                raise RotationControlError("old key revocation is blocked by incomplete inventory")
            changed = db.execute(
                "UPDATE video_key_rotation_jobs SET status='revoking',updated_at=? "
                "WHERE job_id=? AND status NOT IN ('revoking','revoked')",
                (utc_now(), job_id),
            ).rowcount
            if changed != 1:
                raise RotationControlError("rotation job is unavailable for revocation")
            self.tasks._audit(db, None, "video_key_rotation_revocation_started", json.dumps({
                "job_id": job_id, "source_key_id": readiness["source_key_id"],
                "actor_id": actor_id,
            }, sort_keys=True))
        try:
            receipt = revoke(readiness["source_key_id"])
        except Exception as exc:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE video_key_rotation_jobs SET status='ready_to_revoke',updated_at=? "
                    "WHERE job_id=? AND status='revoking'", (utc_now(), job_id),
                )
                self.tasks._audit(db, None, "video_key_rotation_revocation_failed", json.dumps({
                    "job_id": job_id, "source_key_id": readiness["source_key_id"],
                    "error_type": type(exc).__name__,
                    "actor_id": actor_id,
                }, sort_keys=True))
            raise RotationControlError("external key revocation failed") from exc
        if not isinstance(receipt, str) or not receipt.strip() or len(receipt) > 500:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE video_key_rotation_jobs SET status='ready_to_revoke',updated_at=? "
                    "WHERE job_id=? AND status='revoking'", (utc_now(), job_id),
                )
                self.tasks._audit(db, None, "video_key_rotation_revocation_failed", json.dumps({
                    "job_id": job_id, "source_key_id": readiness["source_key_id"],
                    "error_type": "InvalidRevocationReceipt",
                    "actor_id": actor_id,
                }, sort_keys=True))
            raise RotationControlError("key revocation provider returned no receipt")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE video_key_rotation_jobs SET status='revoked',revocation_receipt=?,updated_at=? "
                "WHERE job_id=? AND status='revoking'",
                (receipt.strip(), utc_now(), job_id),
            ).rowcount
            if changed != 1:
                raise RotationControlError("rotation job changed during key revocation")
            self.tasks._audit(db, None, "video_key_rotation_key_revoked", json.dumps({
                "job_id": job_id, "source_key_id": readiness["source_key_id"],
                "target_key_id": readiness["target_key_id"], "receipt": receipt.strip(),
                "actor_id": actor_id,
            }, sort_keys=True))
        return {**readiness, "revoked": True, "revocation_receipt": receipt.strip()}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as db:
            job = db.execute(
                "SELECT * FROM video_key_rotation_jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            items = [dict(row) for row in db.execute(
                "SELECT * FROM video_key_rotation_items WHERE job_id=? ORDER BY task_id", (job_id,),
            )]
            artifacts = [dict(row) for row in db.execute(
                "SELECT * FROM video_key_rotation_artifacts WHERE job_id=? ORDER BY artifact_id", (job_id,),
            )]
        return {"job": dict(job), "items": items, "artifacts": artifacts}

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise RotationControlError("rotation job limit is invalid")
        with self.connect() as db:
            rows = db.execute(
                "SELECT job_id,status,source_key_id,target_key_id,inventory_complete,"
                "writers_fenced,readers_drained,created_at,updated_at FROM video_key_rotation_jobs "
                "ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _metrics(value: str | None) -> dict[str, Any]:
        try:
            decoded = json.loads(value) if value else {}
        except (TypeError, json.JSONDecodeError):
            raise RotationControlError("task processing metrics are invalid") from None
        if not isinstance(decoded, dict):
            raise RotationControlError("task processing metrics are invalid")
        return decoded

    @staticmethod
    def _validate_key_id(key_id: str) -> None:
        if not isinstance(key_id, str) or not _ID.fullmatch(key_id):
            raise RotationControlError("rotation key ID is invalid")

    @staticmethod
    def _read_manifest(directory: Path) -> tuple[dict[str, Any], str]:
        path = directory / MANIFEST_NAME
        if (directory.is_symlink() or not directory.is_dir() or path.is_symlink()
                or not path.is_file() or path.stat().st_size > 1024 * 1024):
            raise RotationControlError("video manifest is missing or unsafe")
        try:
            data = path.read_bytes()
            manifest = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise RotationControlError("video manifest is invalid") from None
        if not isinstance(manifest, dict):
            raise RotationControlError("video manifest is invalid")
        return manifest, hashlib.sha256(data).hexdigest()

    @staticmethod
    def _manifest_key_id(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
        version = manifest.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise RotationControlError("video manifest version is invalid")
        if version == ENVELOPED_MANIFEST_VERSION:
            envelope = manifest.get("key_envelope")
            key_id = envelope.get("key_id") if isinstance(envelope, dict) else None
        elif version == MANIFEST_VERSION:
            key_id = metrics.get("key_id") or "legacy-api-v1"
        else:
            raise RotationControlError("unsupported video manifest version")
        if not isinstance(key_id, str) or not _ID.fullmatch(key_id):
            raise RotationControlError("video manifest key ID is invalid")
        return key_id

    @staticmethod
    def _inventory_cache_root(cache_root: Path) -> dict[str, list[dict[str, Any]]]:
        if cache_root.is_symlink():
            raise RotationControlError("protected feature cache root is unsafe")
        if not cache_root.exists():
            return {}
        if not cache_root.is_dir():
            raise RotationControlError("protected feature cache root is unsafe")
        result = {}
        for directory in sorted(cache_root.iterdir()):
            if (directory.is_symlink() or not directory.is_dir()
                    or not _ID.fullmatch(directory.name)):
                raise RotationControlError("protected feature cache contains an unsafe entry")
            files = []
            for path in sorted(directory.iterdir()):
                if (path.is_symlink() or not path.is_file()
                        or path.stat().st_size > MAX_CACHE_BYTES):
                    raise RotationControlError("protected feature cache contains an unsafe entry")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                files.append({"reference": str(path), "sha256": digest.hexdigest(),
                              "size": path.stat().st_size})
            result[directory.name] = files
        return result


class VideoKeyRotationWorker:
    """Process verified object migrations and switch DB pointers transactionally."""

    def __init__(
        self,
        repository: RotationControlRepository,
        storage_root: Path,
        *,
        target_provider: VideoKeyProvider,
        source_provider_resolver: Callable[[str], VideoKeyProvider],
        legacy_encryption_key: bytes | None = None,
        legacy_mac_key: bytes | None = None,
        feature_store: FeatureStore | None = None,
    ):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.target_provider = target_provider
        self.source_provider_resolver = source_provider_resolver
        self.legacy_encryption_key = legacy_encryption_key
        self.legacy_mac_key = legacy_mac_key
        self.feature_store = feature_store

    def process_next(self, job_id: str) -> dict[str, Any] | None:
        item = self.repository.claim_next(job_id)
        if item is None:
            return None
        task_id = item["task_id"]
        target_storage_id = "rot-" + hashlib.sha256(
            (job_id + ":" + task_id).encode("utf-8")
        ).hexdigest()[:32]
        target_directory = self.storage_root / target_storage_id
        target_created = False
        try:
            job = item["job"]
            if self.target_provider.key_id != job["target_key_id"]:
                raise RotationControlError("worker target provider does not match the rotation plan")
            if self.storage_root.is_symlink() or not self.storage_root.is_dir():
                raise RotationControlError("video storage root is unavailable or unsafe")
            source_directory = self.storage_root / item["source_storage_id"]
            manifest, source_manifest_sha256 = self.repository._read_manifest(source_directory)
            if source_manifest_sha256 != item["source_manifest_sha256"]:
                raise RotationControlError("source manifest changed after inventory")
            if manifest.get("video_id") != task_id:
                raise RotationControlError("source manifest identity does not match the task")
            if self.repository._manifest_key_id(manifest, {}) != item["source_key_id"]:
                raise RotationControlError("source key ID does not match the rotation plan")
            if target_directory.exists() or target_directory.is_symlink():
                target_manifest, target_manifest_hash = self._verify_existing_target(
                    target_directory, task_id, manifest,
                )
            else:
                target_manifest, target_manifest_hash = self._create_target(
                    source_directory, task_id, manifest, source_manifest_sha256,
                    target_directory,
                )
                target_created = True

            # The source object remains registered as a replica until an operator
            # migrates or deletes it under the backup/rollback retention policy.
            self.repository.complete_verified_and_switch(
                job_id, task_id,
                target_storage_id=target_storage_id,
                target_manifest_sha256=target_manifest_hash,
                target_key_version=str(target_manifest["key_version"]),
            )
            try:
                cache_evidence = self.process_pending_cache_artifacts(job_id, task_id=task_id)
            except Exception as cache_error:
                cache_evidence = [{"cleanup_pending": type(cache_error).__name__}]
            return {"job_id": job_id, "task_id": task_id, "state": "switched",
                    "source_storage_id": item["source_storage_id"],
                    "target_storage_id": target_storage_id,
                    "target_manifest_sha256": target_manifest_hash,
                    "target_key_id": self.target_provider.key_id,
                    "cache_cleanup": cache_evidence}
        except Exception as exc:
            if target_created and target_directory.exists():
                task = TaskRepository(self.repository.database).get(task_id)
                if not task or task.get("storage_id") != target_storage_id:
                    shutil.rmtree(target_directory, ignore_errors=True)
            try:
                self.repository.fail_item(job_id, task_id, exc)
            except RotationControlError:
                pass
            if isinstance(exc, RotationControlError):
                raise
            raise RotationControlError("video key rotation item failed") from exc

    def _create_target(self, source_directory, task_id, manifest, source_manifest_sha256,
                       target_directory):
        with tempfile.TemporaryDirectory(prefix=".rotation-stage-", dir=self.storage_root) as stage:
            stage_root = Path(stage)
            target_root = stage_root / "target"
            target_root.mkdir()
            version = manifest.get("version")
            if isinstance(version, bool) or not isinstance(version, int):
                raise RotationControlError("source manifest version is invalid")
            if version == MANIFEST_VERSION:
                if self.legacy_encryption_key is None or self.legacy_mac_key is None:
                    raise RotationControlError("legacy v1 key material is unavailable")
                restored = stage_root / "legacy-plaintext.bin"
                restore_encrypted_video(
                    source_directory, restored, self.legacy_encryption_key,
                    self.legacy_mac_key,
                    expected_manifest_sha256=source_manifest_sha256,
                    expected_video_id=task_id,
                )
                target = store_enveloped_video(
                    restored, target_root, self.target_provider,
                    chunk_size=int(manifest["chunk_size"]),
                    key_version="enveloped-" + self.target_provider.key_id,
                    video_id=task_id,
                )
                target_manifest = dict(target.manifest)
                target_manifest_path = target.directory / MANIFEST_NAME
                target_digest = hashlib.sha256(target_manifest_path.read_bytes()).hexdigest()
                verify = stage_root / "v2-verify.bin"
                restore_enveloped_video(
                    target.directory, verify, self.target_provider,
                    expected_video_id=task_id, expected_manifest_sha256=target_digest,
                )
                verify.unlink()
            elif version == ENVELOPED_MANIFEST_VERSION:
                envelope = manifest.get("key_envelope")
                source_key_id = envelope.get("key_id") if isinstance(envelope, dict) else None
                if source_key_id is None:
                    raise RotationControlError("source key ID is unavailable")
                source_provider = self.source_provider_resolver(source_key_id)
                if source_provider.key_id != source_key_id:
                    raise RotationControlError("source key provider returned the wrong key")
                target = rewrap_enveloped_video(
                    source_directory, target_root, source_provider,
                    self.target_provider, expected_video_id=task_id,
                    expected_manifest_sha256=source_manifest_sha256,
                )
                target_manifest = dict(target.manifest)
                target_manifest_path = target.directory / MANIFEST_NAME
                target_digest = hashlib.sha256(target_manifest_path.read_bytes()).hexdigest()
            else:
                raise RotationControlError("unsupported source manifest version")
            os.replace(target.directory, target_directory)
            return target_manifest, target_digest

    def _verify_existing_target(self, target_directory, task_id, source_manifest):
        manifest, digest = self.repository._read_manifest(target_directory)
        if (manifest.get("version") != ENVELOPED_MANIFEST_VERSION
                or manifest.get("video_id") != task_id
                or manifest.get("source_sha256") != source_manifest.get("source_sha256")
                or self.repository._manifest_key_id(manifest, {}) != self.target_provider.key_id):
            raise RotationControlError("existing rotation target does not match the planned output")
        with tempfile.TemporaryDirectory(prefix=".rotation-verify-", dir=self.storage_root) as stage:
            restored = Path(stage) / "verified.bin"
            restore_enveloped_video(
                target_directory, restored, self.target_provider,
                expected_video_id=task_id, expected_manifest_sha256=digest,
            )
        return manifest, digest

    def process_pending_cache_artifacts(
        self, job_id: str, *, task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        job = self.repository.get_job(job_id)
        completed = []
        for artifact in job["artifacts"]:
            if (artifact["kind"] != "feature_cache" or artifact["state"] not in {"pending", "failed"}
                    or (task_id is not None and artifact.get("task_id") != task_id)):
                continue
            if self.feature_store is None:
                raise RotationControlError("feature cache artifact requires a configured FeatureStore")
            cache_id = Path(artifact["source_ref"]).name
            result = self.feature_store.invalidate(cache_id)
            evidence = hashlib.sha256(json.dumps(
                result, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            self.repository.finish_artifact(
                job_id, artifact["artifact_id"], disposition="deleted",
                evidence="invalidated feature cache sha256=" + evidence,
            )
            completed.append({"artifact_id": artifact["artifact_id"], **result,
                              "evidence_sha256": evidence})
        return completed
