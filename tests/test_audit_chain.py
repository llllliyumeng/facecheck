import csv
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from facecheck.database import AUDIT_CHAIN_GENESIS, TaskRepository
from facecheck.main import create_app


def test_new_events_form_a_verifiable_chain(tmp_path):
    repo = TaskRepository(tmp_path / "audit.sqlite")
    for index in range(3):
        repo.record_event(None, "example", str(index))

    integrity = repo.audit_integrity()
    events = list(reversed(repo.list_audit(10, 0)))
    assert integrity["verified"] is True
    assert integrity["checked_events"] == integrity["event_count"] == 3
    assert events[0]["previous_hash"] == AUDIT_CHAIN_GENESIS
    assert len(events[0]["event_hash"]) == 64
    assert events[1]["previous_hash"] == events[0]["event_hash"]
    assert events[2]["previous_hash"] == events[1]["event_hash"]


def test_concurrent_writers_do_not_fork_the_chain(tmp_path):
    repo = TaskRepository(tmp_path / "concurrent.sqlite")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: repo.record_event(None, "parallel", str(index)), range(64)))

    integrity = repo.audit_integrity()
    assert integrity["verified"] is True
    assert integrity["event_count"] == 64


@pytest.mark.parametrize("tamper", ["edit", "delete_middle", "reorder"])
def test_chain_detects_event_edits_deletions_and_reordering(tmp_path, tamper):
    path = tmp_path / f"{tamper}.sqlite"
    repo = TaskRepository(path)
    for index in range(4):
        repo.record_event(None, "example", str(index))

    with repo._connect() as db:
        if tamper == "edit":
            db.execute("UPDATE audit_events SET detail='changed' WHERE event_id=2")
        elif tamper == "delete_middle":
            db.execute("DELETE FROM audit_events WHERE event_id=2")
        else:
            db.execute("UPDATE audit_events SET event_type='swapped' WHERE event_id=1")
            db.execute("UPDATE audit_events SET event_type='example' WHERE event_id=2")

    integrity = repo.audit_integrity()
    assert integrity["verified"] is False
    assert integrity["failure"]["event_id"] in {1, 2, 3}


def test_legacy_database_is_backfilled_once_and_remains_verifiable(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY, original_filename TEXT NOT NULL,
                source_size INTEGER NOT NULL, status TEXT NOT NULL,
                storage_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                error_message TEXT
            );
            CREATE TABLE audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
                event_type TEXT NOT NULL, created_at TEXT NOT NULL, detail TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            );
            INSERT INTO audit_events(task_id,event_type,created_at,detail)
                VALUES(NULL,'legacy_one','2026-01-01T00:00:00+00:00','a');
            INSERT INTO audit_events(task_id,event_type,created_at,detail)
                VALUES(NULL,'legacy_two','2026-01-02T00:00:00+00:00','b');
            """
        )

    repo = TaskRepository(path)
    first = repo.audit_integrity()
    assert first["verified"] is True
    assert first["event_count"] == 2
    hashes_before_restart = [event["event_hash"] for event in reversed(repo.list_audit(10, 0))]

    reopened = TaskRepository(path)
    hashes_after_restart = [event["event_hash"] for event in reversed(reopened.list_audit(10, 0))]
    assert hashes_after_restart == hashes_before_restart
    assert reopened.audit_integrity()["verified"] is True


def test_integrity_endpoint_and_exports_include_snapshot_status(tmp_path):
    database = tmp_path / "audit.sqlite"
    app = create_app(database_path=database, storage_root=tmp_path / "storage")
    repo = TaskRepository(database)
    repo.record_event(None, "example", "first")
    repo.record_event(None, "example", "second")

    with TestClient(app) as client:
        status = client.get("/api/v1/platform/audit/integrity")
        assert status.status_code == 200
        assert status.json()["verified"] is True
        assert status.json()["event_count"] == 2

        exported = client.get("/api/v1/platform/audit/export?format=json").json()
        assert exported["count"] == 2
        assert exported["integrity"]["verified"] is True
        assert exported["integrity"]["event_count"] == 2
        assert exported["items"][0]["event_hash"]

        response = client.get("/api/v1/platform/audit/export?format=csv")
        assert response.headers["x-facecheck-audit-chain-verified"] == "true"
        records = list(csv.DictReader(response.content.decode("utf-8-sig").splitlines()))
        assert records[0]["record_type"] == "integrity_summary"
        assert records[0]["chain_verified"] == "true"
        assert records[1]["record_type"] == "event"

        markdown = client.get("/api/v1/platform/audit/export?format=md").text
        assert "全链校验：通过" in markdown
        assert "不能证明尾部未截断" in markdown

        with repo._connect() as db:
            db.execute("UPDATE audit_events SET detail='tampered' WHERE event_id=1")
        failed_export = client.get("/api/v1/platform/audit/export?format=json").json()
        assert failed_export["integrity"]["verified"] is False
        assert failed_export["integrity"]["failure"]["event_id"] == 1
