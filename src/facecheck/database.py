from __future__ import annotations

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_STATUSES = ("pending", "processing", "completed", "failed")
AUDIT_CHAIN_VERSION = 1
AUDIT_CHAIN_GENESIS = hashlib.sha256(b"facecheck-audit-v1:genesis").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    source_size INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'processing', 'completed', 'failed')
                    ),
                    storage_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    detail TEXT,
                    previous_hash TEXT,
                    event_hash TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                    ON tasks(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_task_id
                    ON audit_events(task_id);
                """
            )
            # Keep additive migrations and any legacy-chain backfill atomic.
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "processing_metrics" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN processing_metrics TEXT")

            # Existing databases predate the audit chain. Backfill their current
            # rows exactly once, in event_id order. The backfill establishes an
            # integrity baseline; it cannot prove the history before migration.
            audit_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(audit_events)").fetchall()
            }
            legacy_chain = "event_hash" not in audit_columns
            if "previous_hash" not in audit_columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN previous_hash TEXT")
            if "event_hash" not in audit_columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN event_hash TEXT")
            if legacy_chain:
                previous_hash = AUDIT_CHAIN_GENESIS
                rows = connection.execute(
                    "SELECT event_id, task_id, event_type, created_at, detail "
                    "FROM audit_events ORDER BY event_id"
                ).fetchall()
                for row in rows:
                    event = dict(row)
                    event_hash = self._event_hash(event, previous_hash)
                    connection.execute(
                        "UPDATE audit_events SET previous_hash=?, event_hash=? WHERE event_id=?",
                        (previous_hash, event_hash, event["event_id"]),
                    )
                    previous_hash = event_hash

    def create_task(self, task_id: str, filename: str, source_size: int) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, original_filename, source_size, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (task_id, filename, source_size, now, now),
            )
            self._audit(connection, task_id, "task_created", None)

    def transition(
        self,
        task_id: str,
        status: str,
        *,
        storage_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if status not in TASK_STATUSES:
            raise ValueError(f"invalid task status: {status}")
        with self._connect() as connection:
            current = connection.execute(
                "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if current is None:
                raise KeyError(task_id)
            allowed = {
                "pending": {"processing", "failed"},
                "processing": {"completed", "failed"},
                "completed": set(),
                "failed": set(),
            }
            if status not in allowed[current["status"]]:
                raise ValueError(
                    f"invalid task transition: {current['status']} -> {status}"
                )
            now = utc_now()
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, storage_id = COALESCE(?, storage_id),
                    error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status, storage_id, error_message, now, task_id),
            )
            self._audit(connection, task_id, f"task_{status}", error_message)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    def list_audit(
        self, limit: int, offset: int, task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT event_id, task_id, event_type, created_at, detail,
                   previous_hash, event_hash
            FROM audit_events
        """
        parameters: list[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            parameters.append(task_id)
        query += " ORDER BY event_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def audit_snapshot(
        self, task_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Read matching events and full-chain status from the same DB snapshot."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            integrity = self._audit_integrity(connection)
            sql = (
                "SELECT event_id, task_id, event_type, created_at, detail, "
                "previous_hash, event_hash FROM audit_events"
            )
            parameters: tuple[Any, ...] = ()
            if task_id is not None:
                sql += " WHERE task_id=?"
                parameters = (task_id,)
            rows = [dict(row) for row in connection.execute(sql + " ORDER BY event_id", parameters)]
        return rows, integrity

    def audit_integrity(self) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN")
            return self._audit_integrity(connection)

    @classmethod
    def _audit_integrity(cls, connection: sqlite3.Connection) -> Dict[str, Any]:
        summary = connection.execute(
            "SELECT COUNT(*) AS event_count, MIN(event_id) AS first_event_id, "
            "MAX(event_id) AS last_event_id FROM audit_events"
        ).fetchone()
        event_count = int(summary["event_count"])
        rows = connection.execute(
            "SELECT event_id, task_id, event_type, created_at, detail, "
            "previous_hash, event_hash FROM audit_events ORDER BY event_id"
        )
        previous_hash = AUDIT_CHAIN_GENESIS
        first_event_id: Optional[int] = summary["first_event_id"]
        last_event_id: Optional[int] = summary["last_event_id"]
        checked = 0
        failure: Optional[Dict[str, Any]] = None
        for row in rows:
            event = dict(row)
            event_id = int(event["event_id"])
            expected_hash = cls._event_hash(event, previous_hash)
            if event.get("previous_hash") != previous_hash:
                failure = {"event_id": event_id, "reason": "previous_hash_mismatch"}
                break
            if event.get("event_hash") != expected_hash:
                failure = {"event_id": event_id, "reason": "event_hash_mismatch"}
                break
            previous_hash = expected_hash
            checked += 1
        return {
            "chain_version": AUDIT_CHAIN_VERSION,
            "verified": failure is None,
            "status": "verified" if failure is None else "failed",
            "checked_events": checked,
            "event_count": event_count,
            "first_event_id": first_event_id,
            "last_event_id": last_event_id,
            "head_hash": previous_hash if failure is None else None,
            "failure": failure,
            "coverage": "events recorded since chain initialization; legacy rows were backfilled when this schema was first introduced",
            "external_anchor": False,
            "limitation": "A local hash chain detects row edits and many internal deletions/reorders, but cannot prove resistance to tail truncation, whole-database rollback, or an attacker who can rewrite the database and recompute the chain.",
        }

    @staticmethod
    def _event_hash(event: Dict[str, Any], previous_hash: str) -> str:
        payload = {
            "event_id": int(event["event_id"]),
            "task_id": event.get("task_id"),
            "event_type": event["event_type"],
            "created_at": event["created_at"],
            "detail": event.get("detail"),
            "previous_hash": previous_hash,
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(b"facecheck-audit-v1\0" + canonical).hexdigest()

    def audit_count(self, task_id: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM audit_events"
        parameters: list[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            parameters.append(task_id)
        with self._connect() as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def record_event(
        self, task_id: Optional[str], event_type: str, detail: Optional[str] = None
    ) -> None:
        with self._connect() as connection:
            self._audit(connection, task_id, event_type, detail)

    def update_metrics(self, task_id: str, metrics: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET processing_metrics = ?, updated_at = ? WHERE task_id = ?",
                (json.dumps(metrics, sort_keys=True), utc_now(), task_id),
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        raw_metrics = value.get("processing_metrics")
        if raw_metrics:
            try:
                value["processing_metrics"] = json.loads(raw_metrics)
            except json.JSONDecodeError:
                value["processing_metrics"] = {}
        else:
            value["processing_metrics"] = {}
        return value

    def _audit(
        self, connection: sqlite3.Connection, task_id: Optional[str], event_type: str, detail: Optional[str]
    ) -> None:
        # record_event() starts without a write lock. Serialize tail lookup and
        # insert so concurrent writers cannot fork the chain.
        if not connection.in_transaction:
            connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT event_id, event_hash FROM audit_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else AUDIT_CHAIN_GENESIS
        created_at = utc_now()
        connection.execute(
            """
            INSERT INTO audit_events (
                task_id, event_type, created_at, detail, previous_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, event_type, created_at, detail, previous_hash, ""),
        )
        event_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        event_hash = self._event_hash(
            dict(
                event_id=event_id,
                task_id=task_id,
                event_type=event_type,
                created_at=created_at,
                detail=detail,
            ),
            previous_hash,
        )
        connection.execute(
            "UPDATE audit_events SET event_hash=? WHERE event_id=?",
            (event_hash, event_id),
        )
