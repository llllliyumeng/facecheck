from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict

from .database import TaskRepository
from .service import MAX_UPLOAD_SIZE, VideoTaskService


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

    return router
