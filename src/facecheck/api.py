from __future__ import annotations

import tempfile
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


def create_router(repository: TaskRepository, service: VideoTaskService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.post("/videos", response_model=TaskResponse, status_code=201)
    async def upload_video(file: UploadFile = File(...)) -> TaskResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="filename is required")
        size = 0
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
            result = service.create_and_process(temporary_path, file.filename, size)
            return TaskResponse.model_validate(result)
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @router.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> TaskResponse:
        task = repository.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return TaskResponse.model_validate(task)

    @router.get("/tasks", response_model=TaskListResponse)
    def list_tasks(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> TaskListResponse:
        items = [TaskResponse.model_validate(item) for item in repository.list(limit, offset)]
        return TaskListResponse(
            items=items, limit=limit, offset=offset, total=repository.count()
        )

    @router.get("/videos", response_model=TaskListResponse)
    def list_videos(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> TaskListResponse:
        items = [TaskResponse.model_validate(item) for item in repository.list(limit, offset)]
        return TaskListResponse(
            items=items, limit=limit, offset=offset, total=repository.count()
        )

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
