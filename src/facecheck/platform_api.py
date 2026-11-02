"""Course-platform API: fixed registries, durable jobs and read-only evidence."""
from __future__ import annotations

import csv
import io
import itertools
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from .crypto_backends import get_crypto_backend
from .platform_catalog import PlatformCatalog, safe_file, sha256
from .platform_jobs import JobManager, JobConflict
from .rotation_control import RotationControlError, RotationControlRepository
from .recognition import DEFAULT_RECOGNITION_MODEL, DEFAULT_ARCFACE_THRESHOLD
from .service import SAFE_ID, KEY_VERSION, clean_error
from .storage import ENVELOPED_MANIFEST_VERSION, MANIFEST_NAME, restore_enveloped_video


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: Literal["ytf_smoke20", "ytf_selection"] = "ytf_smoke20"
    model: Literal["arcface", "sface"] = "arcface"
    kind: Literal["batch", "retrieval"] = "batch"
    query_id: Optional[str] = Field(default=None, max_length=80)
    mode: Literal["plaintext", "encrypted_restore"] = "plaintext"
    top_k: int = Field(default=5, ge=1, le=100)


class KeyRotationPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_key_id: str = Field(min_length=1, max_length=64)
    target_key_id: str = Field(min_length=1, max_length=64)


class KeyRotationArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["migrated", "deleted"]
    evidence: str = Field(min_length=1, max_length=400)
    target_ref: Optional[str] = Field(default=None, max_length=2048)
    target_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)


def create_platform_router(repository, service, manager: JobManager,
                           rotation_repository: RotationControlRepository | None = None,
                           key_revoker=None):
    router = APIRouter(prefix="/api/v1/platform", tags=["course platform"])
    catalog = manager.catalog
    rotation_repository = rotation_repository or RotationControlRepository(repository.path)

    def rotation_actor_id(request: Request):
        principal = getattr(request.state, "principal", None)
        return getattr(principal, "id", None)

    @router.get("/overview")
    def overview(request: Request):
        backend = get_crypto_backend()
        return dict(crypto_backend_requested=os.getenv("FACECHECK_CRYPTO_BACKEND", "auto"),
                    crypto_library_version=getattr(backend, "version", None),
                    crypto_library=getattr(backend, "library_path", None),
                    default_model=os.getenv("FACECHECK_RECOGNITION_MODEL", DEFAULT_RECOGNITION_MODEL),
                    arcface_threshold=DEFAULT_ARCFACE_THRESHOLD,
                    crypto_backend=backend.name,
                    key_management=dict(key_version=KEY_VERSION, derivation="sm3-kdf-v1（按密钥版本派生）",
                                        configured=bool(service.write_key_provider or (service.encryption_key and service.mac_key)),
                                        storage=("注入式 VideoKeyProvider；当前部署仍需提供真实 KMS/HSM adapter"
                                                 if service.write_key_provider else
                                                 "运行时环境变量、私有密钥文件或显式构造参数；无KMS"),
                                        rotation="SQLite持久化队列和数据库原子引用切换原型；全局撤销需完整缓存/备份清单及真实KMS回执",
                                        authorization="已启用组织级角色权限（admin/operator/auditor）" if request.app.state.security.enabled else "本机演示未启用；云端模式强制认证"),
                    sm4_implementation=os.getenv("FACECHECK_SM4_NATIVE_IMPL", "gmssl"),
                    protected_feature_cache=service.feature_store is not None,
                    datasets=catalog.datasets(), tasks=repository.count(),
                    recent_jobs=manager.store.list(5)["items"],
                    formal_test=catalog.history("arcface_test"),
                    retrieval_note=("首次认证恢复后建立加密特征缓存；命中时验证缓存MAC并核对完整源密文摘要，再在可信进程评分；不是密文域ANN。"
                                    if service.feature_store is not None else "先认证和解密恢复，再进行明文人脸检索；未实现密文域ANN。"),
                    resource_note="批量/数据集检索共用单任务槽；在线上传与检索在工作线程执行，不阻塞HTTP事件循环。")

    @router.get("/key-rotations")
    def key_rotations():
        return {"items": rotation_repository.list_jobs()}

    @router.post("/key-rotations/plan", status_code=201)
    def plan_key_rotation(payload: KeyRotationPlanRequest, request: Request):
        configured_backups = os.getenv("FACECHECK_BACKUP_ROOTS", "")
        backup_roots = [Path(value) for value in configured_backups.split(os.pathsep) if value]
        backup_inventory_complete = os.getenv("FACECHECK_BACKUP_INVENTORY_COMPLETE") == "1"
        writer_fenced = (
            os.getenv("FACECHECK_ROTATION_WRITERS_FENCED") == "1"
            and service.write_key_provider is not None
            and service.write_key_provider.key_id == payload.target_key_id
        )
        readers_drained = os.getenv("FACECHECK_ROTATION_READERS_DRAINED") == "1"
        try:
            job_id = rotation_repository.create_plan_from_storage(
                service.storage_root, payload.source_key_id, payload.target_key_id,
                feature_store=service.feature_store,
                backup_roots=backup_roots,
                backup_inventory_complete=backup_inventory_complete,
                writers_fenced=writer_fenced,
                readers_drained=readers_drained,
                actor_id=rotation_actor_id(request),
            )
            result = rotation_repository.get_job(job_id)
            result["revocation_readiness"] = rotation_repository.revocation_readiness(job_id)
            result["worker"] = "persistent SQLite queue created; run scripts/run_key_rotation_worker.py with the trusted provider adapter"
            return result
        except (RotationControlError, OSError, ValueError) as exc:
            raise HTTPException(400, clean_error(exc)) from exc

    @router.get("/key-rotations/{job_id}")
    def key_rotation_status(job_id: str):
        try:
            result = rotation_repository.get_job(job_id)
            result["revocation_readiness"] = rotation_repository.revocation_readiness(job_id)
            return result
        except KeyError:
            raise HTTPException(404, "轮换任务不存在")
        except RotationControlError as exc:
            raise HTTPException(400, clean_error(exc)) from exc

    @router.post("/key-rotations/{job_id}/revoke")
    def revoke_old_video_key(job_id: str, request: Request):
        if key_revoker is None:
            raise HTTPException(503, "KMS/HSM revocation adapter is not configured")
        try:
            return rotation_repository.revoke_old_key(
                job_id, key_revoker, actor_id=rotation_actor_id(request),
            )
        except KeyError:
            raise HTTPException(404, "轮换任务不存在")
        except RotationControlError as exc:
            raise HTTPException(409, clean_error(exc)) from exc

    @router.post("/key-rotations/{job_id}/artifacts/{artifact_id}/resolve")
    def resolve_rotation_artifact(job_id: str, artifact_id: str,
                                  payload: KeyRotationArtifactRequest,
                                  request: Request):
        try:
            plan = rotation_repository.get_job(job_id)
            artifact = next((row for row in plan["artifacts"]
                             if row["artifact_id"] == artifact_id), None)
            if artifact is None:
                raise HTTPException(404, "轮换清单项不存在")
            source_ref = artifact["source_ref"]
            source_path = Path(source_ref)
            if not source_path.is_absolute():
                if not SAFE_ID.fullmatch(source_ref):
                    raise HTTPException(400, "轮换清单源路径无效")
                source_path = service.storage_root / source_path
            allowed_roots = [service.storage_root.resolve(),
                             (service.storage_root.parent / "protected-features").resolve()]
            backup_roots = [Path(value) for value in
                            os.getenv("FACECHECK_BACKUP_ROOTS", "").split(os.pathsep) if value]
            allowed_roots.extend(path.resolve() for path in backup_roots)
            source_real = source_path.resolve()
            if not any(source_real == root or root in source_real.parents for root in allowed_roots):
                raise HTTPException(400, "源路径不在已配置的对象/缓存/备份根目录内")
            if source_path.is_symlink() or any(parent.is_symlink() for parent in source_path.parents):
                raise HTTPException(400, "源路径不能包含符号链接")
            if payload.disposition == "deleted":
                if source_path.exists() or source_path.is_symlink():
                    raise HTTPException(409, "源对象仍存在；清理后再登记")
                rotation_repository.finish_artifact(
                    job_id, artifact_id, disposition="deleted",
                    evidence=payload.evidence, actor_id=rotation_actor_id(request),
                )
                return {"job_id": job_id, "artifact_id": artifact_id, "state": "deleted"}

            if not payload.target_ref or not payload.target_sha256:
                raise HTTPException(400, "迁移登记需要目标路径和 manifest 摘要")
            if source_path.exists() or source_path.is_symlink():
                raise HTTPException(409, "旧密钥源副本仍存在；迁移并清理源副本后再登记")
            target_path = Path(payload.target_ref)
            target_real = target_path.resolve()
            if not any(target_real == root or root in target_real.parents for root in allowed_roots):
                raise HTTPException(400, "目标路径不在已配置的对象/备份根目录内")
            if target_path.is_symlink() or any(parent.is_symlink() for parent in target_path.parents):
                raise HTTPException(400, "目标路径不能包含符号链接")
            target_manifest, actual_hash = rotation_repository._read_manifest(target_real)
            if (actual_hash != payload.target_sha256
                    or target_manifest.get("version") != ENVELOPED_MANIFEST_VERSION
                    or rotation_repository._manifest_key_id(target_manifest, {}) != plan["job"]["target_key_id"]):
                raise HTTPException(409, "迁移目标摘要或 key ID 不匹配")
            if service.key_provider_resolver is None:
                raise HTTPException(503, "目标密钥 provider 未配置")
            provider = service.key_provider_resolver(plan["job"]["target_key_id"])
            with tempfile.TemporaryDirectory(prefix="facecheck-artifact-verify-") as temporary:
                restore_enveloped_video(
                    target_real, Path(temporary) / "verified.bin", provider,
                    expected_video_id=target_manifest.get("video_id"),
                    expected_manifest_sha256=actual_hash,
                )
            rotation_repository.finish_artifact(
                job_id, artifact_id, disposition="migrated",
                evidence=payload.evidence, target_ref=str(target_real),
                target_sha256=actual_hash, actor_id=rotation_actor_id(request),
            )
            return {"job_id": job_id, "artifact_id": artifact_id,
                    "state": "migrated", "target_manifest_sha256": actual_hash}
        except HTTPException:
            raise
        except KeyError:
            raise HTTPException(404, "轮换任务不存在")
        except (RotationControlError, OSError, ValueError) as exc:
            raise HTTPException(400, clean_error(exc)) from exc

    @router.get("/datasets")
    def datasets():
        return {"items": catalog.datasets()}

    @router.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        try:
            return catalog.dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "未登记的数据集")

    @router.get("/datasets/{dataset_id}/queries")
    def queries(dataset_id: str):
        if dataset_id != "ytf_smoke20":
            raise HTTPException(400, "交互检索只开放固定20身份集")
        records, assignments = catalog.records(dataset_id)
        return {"items": [dict(query_id=r["video_id"], identity_id=r["identity_id"], split=assignments[r["identity_id"]]) for r in records if r["split"] == "query"]}

    def prepare(request):
        try:
            return manager.prepare(**request.model_dump())
        except (KeyError, ValueError, OSError) as exc:
            raise HTTPException(400, clean_error(exc))

    @router.post("/jobs/preview")
    def preview(request: JobRequest):
        config = prepare(request)
        manager.output_root.mkdir(parents=True, exist_ok=True)
        return dict(config=config, free_disk_bytes=shutil.disk_usage(manager.output_root).free,
                    required_disk_bytes=max(512 * 1024**2, config["source_bytes"] * 3),
                    modes=["plaintext", "encrypted_restore"] if request.kind == "batch" else [request.mode],
                    note="这里只检查资源，不启动任务；完整selection耗时较长。")

    @router.post("/jobs", status_code=202)
    def create_job(request: JobRequest):
        config = prepare(request)
        try:
            return manager.launch(config)
        except JobConflict as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, clean_error(exc))
        except Exception as exc:
            raise HTTPException(500, "工作进程启动失败，状态已记录") from exc

    @router.get("/jobs")
    def jobs(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        manager.store.reconcile()
        return manager.store.list(limit, offset)

    def get_job(job_id):
        try:
            return manager.store.get(job_id)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        result = get_job(job_id)
        if result["status"] == "completed":
            result["result"] = catalog.result(manager.result_dir(job_id))
        return result

    @router.get("/jobs/{job_id}/logs")
    def logs(job_id: str, after: int = Query(0, ge=0)):
        get_job(job_id)
        return {"items": manager.store.logs(job_id, after)}

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        get_job(job_id)
        try:
            return manager.store.cancel(job_id)
        except JobConflict as exc:
            raise HTTPException(409, str(exc))

    @router.get("/jobs/{job_id}/files/{filename}")
    def job_file(job_id: str, filename: str):
        current = get_job(job_id)
        if current["status"] != "completed":
            raise HTTPException(409, "任务尚未完整完成；只开放状态与日志")
        return download(manager.result_dir(job_id), filename)

    def download(directory, filename):
        try:
            return FileResponse(safe_file(directory, filename), filename=filename)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "文件不存在或不在白名单")

    @router.get("/histories")
    def histories():
        return {"items": catalog.histories()}

    @router.get("/histories/{history_id}")
    def history(history_id: str):
        try:
            return catalog.history(history_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "实验未登记或目录不合法")

    def preview_csv(directory, filename, limit):
        if filename not in {"feature_failures.csv", "failed_queries_plaintext.csv", "query_results.csv", "model_comparison.csv"}:
            raise HTTPException(404, "只支持已登记的诊断CSV预览")
        try:
            with safe_file(directory, filename).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(itertools.islice(reader, limit + 1))
                return dict(columns=reader.fieldnames or [], items=rows[:limit], truncated=len(rows) > limit,
                            note="只预览前若干行；完整记录请下载原始CSV。")
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "诊断文件不存在")

    @router.get("/histories/{history_id}/preview/{filename}")
    def history_csv(history_id: str, filename: str, limit: int = Query(20, ge=1, le=100)):
        try:
            return preview_csv(catalog.history_dir(history_id), filename, limit)
        except (KeyError, ValueError):
            raise HTTPException(404, "实验未登记或目录不合法")

    @router.get("/jobs/{job_id}/preview/{filename}")
    def job_csv(job_id: str, filename: str, limit: int = Query(20, ge=1, le=100)):
        current = get_job(job_id)
        if current["status"] != "completed":
            raise HTTPException(409, "任务未完整完成")
        return preview_csv(manager.result_dir(job_id), filename, limit)

    @router.get("/histories/{history_id}/files/{filename}")
    def history_file(history_id: str, filename: str):
        try:
            return download(catalog.history_dir(history_id), filename)
        except (KeyError, ValueError):
            raise HTTPException(404, "实验未登记或目录不合法")

    @router.get("/benchmarks")
    def benchmarks():
        return {"items": catalog.benchmarks(), "note": "历史多轮记录逐份显示环境/口径；本页面不自动重跑性能测试。MB_decimal 为十进制，旧版scalar记录口径需参照历史报告。"}

    @router.get("/benchmarks/{benchmark_id}")
    def benchmark_file(benchmark_id: str):
        path = catalog.benchmark_paths().get(benchmark_id)
        if path is None:
            raise HTTPException(404, "benchmark未登记")
        return FileResponse(path, filename=path.name)

    @router.get("/requirements")
    def requirements():
        return catalog.requirements()

    @router.get("/audit/integrity")
    def audit_integrity():
        return repository.audit_integrity()

    @router.get("/audit/export")
    def export_audit(format: Literal["csv", "json", "md"] = "csv",
                     task_id: Optional[str] = Query(default=None, min_length=1, max_length=64)):
        # Events and full-chain status come from one SQLite read snapshot.
        rows, integrity = repository.audit_snapshot(task_id)
        if format == "csv":
            buffer = io.StringIO()
            fields = [
                "record_type", "event_id", "task_id", "event_type", "created_at",
                "detail", "previous_hash", "event_hash", "chain_verified",
                "chain_event_count", "chain_head_hash",
            ]
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(
                record_type="integrity_summary",
                chain_verified=str(integrity["verified"]).lower(),
                chain_event_count=integrity["event_count"],
                chain_head_hash=integrity["head_hash"] or "",
            ))
            # Protect spreadsheet exports against formula interpretation.
            for row in rows:
                safe_row = {
                    key: "'" + value
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                    else value
                    for key, value in row.items()
                }
                writer.writerow({**safe_row, "record_type": "event"})
            content, media = "\ufeff" + buffer.getvalue(), "text/csv"
        elif format == "json":
            content, media = json.dumps(dict(
                count=len(rows), items=rows, integrity=integrity,
                note="完整性校验覆盖同一SQLite快照中的全量审计链；本地链无外部锚点，不能证明尾部未截断、数据库未回滚或抵御可重算数据库的攻击者。",
            ), ensure_ascii=False, indent=2), "application/json"
        else:
            content = (
                "# FaceCheck 审计快照报告\n\n"
                f"导出事件数：{len(rows)}\n\n"
                f"全链校验：{'通过' if integrity['verified'] else '失败'}；"
                f"检查事件数：{integrity['checked_events']}/{integrity['event_count']}。\n\n"
                "此本地哈希链没有独立外部锚点，不能证明尾部未截断或数据库未回滚，"
                "也不构成99.9%日志完整率或99%报告生成成功率的测量。\n\n"
                "```json\n" + json.dumps(dict(integrity=integrity, items=rows), ensure_ascii=False, indent=2) + "\n```\n"
            )
            media = "text/markdown"
        repository.record_event(None, "audit_report_exported", json.dumps({
            "format": format, "events": len(rows), "chain_verified": integrity["verified"],
            "chain_head_hash": integrity["head_hash"],
        }))
        return Response(content, media_type=media, headers={
            "Content-Disposition": f'attachment; filename="facecheck_audit.{format}"',
            "X-FaceCheck-Audit-Chain-Verified": str(integrity["verified"]).lower(),
            "X-FaceCheck-Audit-Event-Count": str(integrity["event_count"]),
            "X-FaceCheck-Audit-Head-Hash": integrity["head_hash"] or "",
        })

    def restored(task_id):
        if not SAFE_ID.fullmatch(task_id):
            raise HTTPException(400, "非法任务ID")
        task = repository.get(task_id)
        if not task:
            raise HTTPException(404, "视频不存在")
        if task["status"] != "completed" or not task.get("storage_id"):
            raise HTTPException(409, "视频任务未完成")
        storage_id = task["storage_id"]
        if not SAFE_ID.fullmatch(storage_id):
            raise HTTPException(400, "非法存储ID")
        directory = service.storage_root / storage_id
        if directory.resolve().parent != service.storage_root.resolve():
            raise HTTPException(400, "非法存储目录")
        temp = tempfile.TemporaryDirectory(prefix="facecheck-verified-download-")
        output = Path(temp.name) / "restored.mp4"
        started = time.perf_counter()
        try:
            anchored = service.check_storage_manifest(task, directory)
            service.restore_task_storage(task, directory, output)
            result = dict(task_id=task_id, integrity_verified=True, sha256=sha256(output),
                          trusted_manifest_anchor_verified=anchored,
                          bytes=output.stat().st_size, restore_seconds=time.perf_counter() - started,
                          note="逐块HMAC验证后解密；大小与恢复SHA-256验证通过。" + ("manifest摘要与可信任务库一致。" if anchored else "历史本机记录无独立manifest锚点。"))
            repository.record_event(task_id, "restore_integrity_verified", json.dumps(result, ensure_ascii=False))
            return temp, output, result
        except Exception as exc:
            temp.cleanup()
            repository.record_event(task_id, "integrity_validation_failed", clean_error(exc))
            raise HTTPException(400, clean_error(exc))

    @router.post("/videos/{task_id}/verify")
    def verify_video(task_id: str):
        temp, _, result = restored(task_id)
        temp.cleanup()
        return result

    @router.post("/videos/{task_id}/restore")
    def restore_video(task_id: str):
        temp, output, _ = restored(task_id)
        return FileResponse(output, filename="restored.mp4", media_type="video/mp4", background=BackgroundTask(temp.cleanup))

    return router
