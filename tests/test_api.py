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
