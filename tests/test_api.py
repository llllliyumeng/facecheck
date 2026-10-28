import json
from pathlib import Path

from fastapi.testclient import TestClient

from facecheck.main import create_app


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        database_path=tmp_path / "db.sqlite3",
        storage_root=tmp_path / "storage",
        encryption_key=b"0123456789abcdef",
        mac_key=b"independent-mac-key",
    )
    return TestClient(app)


def test_health_and_upload_task_lifecycle(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    upload = client.post(
        "/api/v1/videos",
        files={"file": ("authorized.mp4", b"not-a-video", "video/mp4")},
    )
    assert upload.status_code == 201
    task = upload.json()
    assert task["status"] == "failed"
    assert task["original_filename"] == "authorized.mp4"
    assert str(tmp_path) not in json.dumps(task)
    assert "key" not in json.dumps(task).lower()

    queried = client.get(f"/api/v1/tasks/{task['task_id']}")
    assert queried.status_code == 200
    assert queried.json()["status"] == "failed"


def test_upload_valid_video_and_persistent_listing(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 4, (16, 16))
    assert writer.isOpened()
    writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    client = make_client(tmp_path)
    with source.open("rb") as video:
        upload = client.post(
            "/api/v1/videos",
            files={"file": ("authorized.mp4", video, "video/mp4")},
        )
    assert upload.status_code == 201
    task = upload.json()
    assert task["status"] == "completed"
    assert task["storage_id"] == task["task_id"]
    assert task["encryption_algorithm"] == "SM4-CBC"
    assert task["mac_algorithm"] == "HMAC-SM3"
    assert task["integrity_status"] == "verified"
    assert task["processing_metrics"]["chunk_count"] == 1
    assert Path(tmp_path / "storage" / task["storage_id"] / "manifest.json").is_file()

    listing = client.get("/api/v1/tasks?limit=1&offset=0")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert len(listing.json()["items"]) == 1

    with source.open("rb") as video:
        second_upload = client.post(
            "/api/v1/videos",
            files={"file": ("same-name.mp4", video, "video/mp4")},
        )
    assert second_upload.status_code == 201
    assert second_upload.json()["task_id"] != task["task_id"]
    second_page = client.get("/api/v1/tasks?limit=1&offset=1")
    assert second_page.status_code == 200
    assert second_page.json()["total"] == 2
    assert len(second_page.json()["items"]) == 1

    restarted = make_client(tmp_path)
    assert restarted.get(f"/api/v1/tasks/{task['task_id']}").json()["status"] == "completed"

    audit = restarted.get("/api/v1/audit?limit=20")
    assert audit.status_code == 200
    assert audit.json()["total"] >= 3
    assert any(
        event["event_type"] == "task_completed" for event in audit.json()["items"]
    )
    summary = restarted.get("/api/v1/security/summary")
    assert summary.status_code == 200
    assert summary.json()["manifest_authentication"] == "not independently authenticated"


def test_upload_rejects_oversize_and_missing_task(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/videos",
        files={"file": ("too-large.bin", b"x" * (50 * 1024 * 1024 + 1), "application/octet-stream")},
    )
    assert response.status_code == 413
    assert client.get("/api/v1/tasks/not-found").status_code == 404
    assert not list(tmp_path.glob("facecheck-upload-*"))


def test_missing_key_configuration_creates_failed_task_without_secrets(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "db.sqlite3",
        storage_root=tmp_path / "storage",
        encryption_key=None,
        mac_key=None,
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos",
        files={"file": ("sample.bin", b"invalid", "application/octet-stream")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert "configuration" in body["error_message"]
    assert "0123456789abcdef" not in json.dumps(body)


def test_video_listing_and_search_response_contract(tmp_path: Path, monkeypatch) -> None:
    client = make_client(tmp_path)
    monkeypatch.setattr(
        "facecheck.service.VideoTaskService.search_task",
        lambda self, task_id, query_image, *, top_k, min_similarity: [
            {
                "candidate_id": "frame-0-face-0",
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "similarity": 1.0,
            }
        ],
    )
    response = client.get("/api/v1/videos")
    assert response.status_code == 200
    assert response.json()["items"] == []

    missing = client.post(
        "/api/v1/videos/not-a-real-task/search",
        files={"query_image": ("query.png", b"not-an-image", "image/png")},
    )
    assert missing.status_code == 404


def test_search_rejects_oversize_query_image(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/videos/safe-task/search",
        files={
            "query_image": (
                "query.png",
                b"x" * (10 * 1024 * 1024 + 1),
                "image/png",
            )
        },
    )
    assert response.status_code == 413


def test_upload_sanitizes_filename_and_failed_storage_has_no_orphan(
    tmp_path: Path, monkeypatch
) -> None:
    client = make_client(tmp_path)
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 4, (16, 16))
    assert writer.isOpened()
    writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    from facecheck.database import TaskRepository

    original_transition = TaskRepository.transition

    def fail_completion(self, task_id, status, **kwargs):
        if status == "completed":
            raise OSError("/private/internal/storage/path")
        return original_transition(self, task_id, status, **kwargs)

    monkeypatch.setattr(TaskRepository, "transition", fail_completion)
    with source.open("rb") as video:
        response = client.post(
            "/api/v1/videos",
            files={"file": ("../../秘密 video.mp4", video, "video/mp4")},
        )

    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert "internal" not in response.text
    assert not list((tmp_path / "storage").glob("*"))
    assert not list(tmp_path.glob("facecheck-upload-*"))
