"""Durable single-slot jobs. HTTP validates presets; a separate process does inference."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import TaskRepository, utc_now
from .platform_catalog import PlatformCatalog, ROOT, sha256

ACTIVE = ("queued", "running")
TERMINAL = ("completed", "failed", "cancelled")


class JobConflict(ValueError):
    pass


class JobStore:
    def __init__(self, database: Path):
        self.database = database.resolve()
        TaskRepository(self.database)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS experiment_jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    config TEXT NOT NULL, output_dir TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    processed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL,
                    stage TEXT NOT NULL DEFAULT '等待工作进程', error TEXT,
                    elapsed_seconds REAL NOT NULL DEFAULT 0,
                    slot INTEGER NOT NULL DEFAULT 1
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_experiment
                    ON experiment_jobs(slot) WHERE status IN ('queued','running');
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, message TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES experiment_jobs(id)
                );
            ''')

    def connect(self):
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def decode(row):
        result = dict(row)
        result["config"] = json.loads(result["config"])
        result["cancel_requested"] = bool(result["cancel_requested"])
        result["complete"] = result["status"] == "completed"
        return result

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM experiment_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self.decode(row)

    def list(self, limit=50, offset=0):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM experiment_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            total = db.execute("SELECT count(*) FROM experiment_jobs").fetchone()[0]
        return {"items": [self.decode(r) for r in rows], "total": total}

    def log(self, job_id, message):
        # Callers only pass stages, registered IDs, counts, or sanitized errors.
        with self.connect() as db:
            db.execute("INSERT INTO experiment_logs(job_id,created_at,message) VALUES(?,?,?)", (job_id, utc_now(), message[:500]))

    def logs(self, job_id, after=0):
        self.get(job_id)
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT seq,created_at,message FROM experiment_logs WHERE job_id=? AND seq>? ORDER BY seq LIMIT 200", (job_id, after))]

    def progress(self, job_id, stage, processed=None, elapsed=0):
        with self.connect() as db:
            db.execute("UPDATE experiment_jobs SET stage=?, processed=COALESCE(?,processed), elapsed_seconds=?, updated_at=? WHERE id=? AND status='running'", (stage, processed, elapsed, utc_now(), job_id))

    def finish(self, job_id, status, error=None):
        if status not in TERMINAL:
            raise ValueError(status)
        with self.connect() as db:
            row = db.execute("SELECT status FROM experiment_jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] not in ACTIVE:
                return
            db.execute("UPDATE experiment_jobs SET status=?,stage=?,error=?,updated_at=?,finished_at=? WHERE id=?", (status, status, error, utc_now(), utc_now(), job_id))
        self.log(job_id, f"{status}: {error or '工作进程已结束'}")
        TaskRepository(self.database).record_event(None, "experiment_" + status, json.dumps({"job_id": job_id, "error": error}, ensure_ascii=False))

    def cancel(self, job_id):
        job = self.get(job_id)
        if job["status"] not in ACTIVE:
            raise JobConflict("任务已结束")
        with self.connect() as db:
            db.execute("UPDATE experiment_jobs SET cancel_requested=1,updated_at=? WHERE id=? AND status IN ('queued','running')", (utc_now(), job_id))
        self.log(job_id, "已请求取消；在媒体边界/评分结束后安全退出，退出前仍占用任务槽。")
        TaskRepository(self.database).record_event(None, "experiment_cancel_requested", job_id)
        return self.get(job_id)

    def reconcile(self):
        """Do not infer failure from a slow inference heartbeat. Check worker identity."""
        with self.connect() as db:
            active = db.execute("SELECT * FROM experiment_jobs WHERE status IN ('queued','running')").fetchall()
        for row in active:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).total_seconds()
            pid = row["pid"]
            if not pid and age < 120:
                continue
            alive = False
            if pid:
                proc = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
                alive = proc.returncode == 0 and "facecheck.platform_worker" in proc.stdout and row["id"] in proc.stdout
            if not alive:
                self.finish(row["id"], "failed", "工作进程已退出；结果 incomplete，不会自动续跑或作为正式结果。")


class JobManager:
    def __init__(self, database: Path, output_root: Path, catalog=None):
        self.store = JobStore(database)
        self.output_root = output_root.resolve()
        self.catalog = catalog or PlatformCatalog()
        self.store.reconcile()

    def prepare(self, dataset_id, model, kind="batch", query_id=None, mode="plaintext", top_k=5):
        if model not in ("arcface", "sface") or kind not in ("batch", "retrieval"):
            raise ValueError("未登记的任务配置")
        if mode not in ("plaintext", "encrypted_restore") or not 1 <= top_k <= 100:
            raise ValueError("无效检索配置")
        dataset = self.catalog.dataset(dataset_id)
        if not dataset["available"]:
            raise ValueError("数据集不可用")
        records, _ = self.catalog.records(dataset_id)
        if kind == "retrieval":
            # Keep classroom interactive retrieval bounded to the registered smoke set.
            if dataset_id != "ytf_smoke20":
                raise ValueError("交互检索仅开放已登记20身份集")
            if query_id not in {r["video_id"] for r in records if r["split"] == "query"}:
                raise ValueError("Query 不属于已登记数据集")
            query = next(r for r in records if r["video_id"] == query_id)
            _, assignments = self.catalog.records(dataset_id)
            records = [r for r in records if r["video_id"] == query_id or (r["split"] == "gallery" and assignments[r["identity_id"]] == assignments[query["identity_id"]])]
        missing = sum(not Path(r["video_path"]).is_file() for r in records)
        if missing:
            raise ValueError(f"数据缺失：{missing} 个媒体文件")
        model_file = self.catalog.root / "models" / ("arcface_w600k_r50.onnx" if model == "arcface" else "face_recognition_sface_2021dec.onnx")
        detector = self.catalog.root / "models/face_detection_yunet_2023mar.onnx"
        for path in (model_file, detector):
            if not path.is_file():
                raise ValueError("所选模型文件不可用")
        manifest, split, scope = self.catalog.inputs(dataset_id)
        return dict(dataset_id=dataset_id, model=model, kind=kind, query_id=query_id,
                    mode=mode if kind == "retrieval" else "paired", top_k=top_k,
                    evaluation_scope=scope, formal=False,
                    frame_interval_seconds=1.0, score_threshold=.9, score_aggregation="max",
                    gallery_quality_filter=False, gallery_feature_aggregation="none", query_use_annotation_box=False,
                    manifest_sha256=sha256(manifest), split_sha256=sha256(split),
                    model_sha256=sha256(model_file), detector_sha256=sha256(detector),
                    total=len(records) * 2 if kind == "batch" else 1 + sum(r["split"] == "gallery" for r in records),
                    source_bytes=sum(Path(r["video_path"]).stat().st_size for r in records),
                    estimated_duration=dataset["estimated_minutes"],
                    output_policy="独立探索性任务；不运行正式test；阈值仅来自该次plaintext selection")

    def launch(self, config):
        self.store.reconcile()
        self.output_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.output_root).free
        # Encrypted work copies and output tables plus generous reserve.
        required = max(512 * 1024**2, config["source_bytes"] * 3)
        if free < required:
            raise ValueError("磁盘空间不足：需至少三倍输入大小及512MiB预留")
        job_id = uuid.uuid4().hex
        output = self.output_root / job_id
        output.mkdir(exist_ok=False)
        now = utc_now()
        try:
            with self.store.connect() as db:
                db.execute("INSERT INTO experiment_jobs(id,status,config,output_dir,created_at,updated_at,total) VALUES(?,'queued',?,?,?,?,?)", (job_id, json.dumps(config), str(output), now, now, config["total"]))
        except sqlite3.IntegrityError as exc:
            output.rmdir()
            raise JobConflict("已有批量/检索任务占用资源；请等待结束或安全取消") from exc
        (output / "request.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.log(job_id, "queued：固定配置已保存，等待独立工作进程。")
        TaskRepository(self.store.database).record_event(None, "experiment_queued", json.dumps({"job_id": job_id, "dataset": config["dataset_id"], "model": config["model"], "kind": config["kind"]}))
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            process = self.spawn_worker([sys.executable, "-m", "facecheck.platform_worker", str(self.store.database), job_id],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            with self.store.connect() as db:
                db.execute("UPDATE experiment_jobs SET pid=? WHERE id=? AND pid IS NULL", (process.pid, job_id))
            self.watch_process(process, job_id)
        except Exception:
            self.store.finish(job_id, "failed", "工作进程启动失败")
            raise
        return self.store.get(job_id)

    @staticmethod
    def spawn_worker(*args, **kwargs):
        return subprocess.Popen(*args, **kwargs)

    def watch_process(self, process, job_id):
        def reap():
            process.wait()
            job = self.store.get(job_id)
            if job["status"] in ACTIVE:
                self.store.finish(job_id, "failed", "工作进程异常退出；结果 incomplete。")
        threading.Thread(target=reap, daemon=True).start()

    def result_dir(self, job_id):
        job = self.store.get(job_id)
        path = Path(job["output_dir"])
        if path.resolve().parent != self.output_root or path.is_symlink():
            raise ValueError("任务目录不合法")
        result = path / "result" if job["config"]["kind"] == "batch" else path
        if result != path and (result.is_symlink() or result.resolve().parent != path.resolve()):
            raise ValueError("结果目录不合法")
        return result
