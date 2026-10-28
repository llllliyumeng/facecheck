from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_STATUSES = ("pending", "processing", "completed", "failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
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
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                    ON tasks(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_task_id
                    ON audit_events(task_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "processing_metrics" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN processing_metrics TEXT")

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
            SELECT event_id, task_id, event_type, created_at, detail
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
        self, connection: sqlite3.Connection, task_id: str, event_type: str, detail: Optional[str]
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (task_id, event_type, created_at, detail)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, event_type, utc_now(), detail),
        )
