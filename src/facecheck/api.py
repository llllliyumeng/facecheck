from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict

from .database import TaskRepository
from .service import (
    MAX_QUERY_IMAGE_SIZE,
    MAX_UPLOAD_SIZE,
    VideoTaskService,
    clean_error,
)
from .storage import StorageError


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    original_filename: str
    source_size: int
    status: str
    storage_id: Optional[str] = None
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    processing_metrics: dict[str, object] = {}
    encryption_algorithm: Optional[str] = None
    mac_algorithm: Optional[str] = None
    integrity_status: Optional[str] = None
    key_version: Optional[str] = None
    chunk_count: Optional[int] = None


class TaskListResponse(BaseModel):
    items: list[TaskResponse]
    limit: int
    offset: int
    total: int


class FaceSearchResultResponse(BaseModel):
    candidate_id: str
    frame_index: int
    timestamp_seconds: float
    similarity: float


class FaceSearchResponse(BaseModel):
    task_id: str
    status: str
    similarity_note: str
    results: list[FaceSearchResultResponse]
    timings: dict[str, object] = {}


class AuditEventResponse(BaseModel):
    event_id: int
    task_id: Optional[str] = None
    event_type: str
    created_at: str
    detail: Optional[str] = None


class AuditListResponse(BaseModel):
    items: list[AuditEventResponse]
    limit: int
    offset: int
    total: int


def task_response(task: dict) -> TaskResponse:
    metrics = task.get("processing_metrics") or {}
    completed = task.get("status") == "completed"
    data = dict(task)
    data["processing_metrics"] = metrics
    return TaskResponse(
        **data,
        encryption_algorithm="SM4-CBC" if completed else None,
        mac_algorithm="HMAC-SM3" if completed else None,
        integrity_status="verified" if completed else None,
        key_version=metrics.get("key_version") if completed else None,
        chunk_count=metrics.get("chunk_count") if completed else None,
    )


def create_router(repository: TaskRepository, service: VideoTaskService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.post(
        "/videos",
        response_model=TaskResponse,
        response_model_exclude_none=True,
        status_code=201,
    )
    async def upload_video(file: UploadFile = File(...)) -> TaskResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="filename is required")
        size = 0
        received_started = time.perf_counter()
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="facecheck-upload-", suffix=".bin", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    block = await file.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds {MAX_UPLOAD_SIZE} byte limit",
                        )
                    temporary.write(block)
            result = service.create_and_process(
                temporary_path,
                file.filename,
                size,
                upload_receive_seconds=time.perf_counter() - received_started,
            )
            return task_response(result)
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @router.get(
        "/tasks/{task_id}",
        response_model=TaskResponse,
        response_model_exclude_none=True,
    )
    def get_task(task_id: str) -> TaskResponse:
        task = repository.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task_response(task)

    @router.get(
        "/tasks",
        response_model=TaskListResponse,
        response_model_exclude_none=True,
    )
    def list_tasks(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> TaskListResponse:
        items = [task_response(item) for item in repository.list(limit, offset)]
        return TaskListResponse(
            items=items, limit=limit, offset=offset, total=repository.count()
        )

    @router.get(
        "/videos",
        response_model=TaskListResponse,
        response_model_exclude_none=True,
    )
    def list_videos(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> TaskListResponse:
        items = [task_response(item) for item in repository.list(limit, offset)]
        return TaskListResponse(
            items=items, limit=limit, offset=offset, total=repository.count()
        )

    @router.get("/audit", response_model=AuditListResponse)
    def list_audit(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        task_id: Optional[str] = Query(default=None, min_length=1, max_length=64),
    ) -> AuditListResponse:
        items = [
            AuditEventResponse.model_validate(item)
            for item in repository.list_audit(limit, offset, task_id)
        ]
        return AuditListResponse(
            items=items,
            limit=limit,
            offset=offset,
            total=repository.audit_count(task_id),
        )

    @router.get("/security/summary")
    def security_summary() -> dict[str, object]:
        tasks = repository.list(100, 0)
        completed = [task for task in tasks if task["status"] == "completed"]
        failures = [
            event
            for event in repository.list_audit(200, 0)
            if event["event_type"]
            in {"face_search_failed", "integrity_validation_failed", "task_failed"}
        ]
        return {
            "encryption": "SM4-CBC",
            "authentication": "HMAC-SM3",
            "integrity_check": "chunk MAC + SHA-256",
            "manifest_authentication": "not independently authenticated",
            "authentication_required": False,
            "completed_video_count": len(completed),
            "audit_event_count": repository.audit_count(),
            "failure_count": len(failures),
        }

    @router.post(
        "/videos/{task_id}/search",
        response_model=FaceSearchResponse,
    )
    async def search_video(
        task_id: str,
        query_image: UploadFile = File(...),
        top_k: int = Query(default=10, ge=1, le=100),
        min_similarity: Optional[float] = Query(default=None, ge=-1, le=1),
    ) -> FaceSearchResponse:
        try:
            data = await query_image.read(MAX_QUERY_IMAGE_SIZE + 1)
            if len(data) > MAX_QUERY_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail="query image is too large")
            results = service.search_task(
                task_id,
                data,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            task = repository.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            return FaceSearchResponse(
                task_id=task_id,
                status=task["status"],
                similarity_note="相似度是向量间的数学相似度，不是身份确认概率。",
                results=[FaceSearchResultResponse.model_validate(item) for item in results],
                timings=service.last_search_metrics,
            )
        except HTTPException:
            raise
        except KeyError:
            raise HTTPException(status_code=404, detail="task not found")
        except (ValueError, StorageError) as error:
            raise HTTPException(status_code=400, detail=clean_error(error))
        except Exception as error:
            raise HTTPException(status_code=500, detail="face search failed") from error
        finally:
            await query_image.close()

    return router
