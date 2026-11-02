"""Private subprocess entry point; no HTTP-supplied commands or filesystem paths."""
from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .database import TaskRepository, utc_now
from .face_detection import FaceDetectionConfig
from .platform_catalog import PlatformCatalog, sha256
from .platform_jobs import JobStore
from .recognition import recognition_config
from .service import clean_error
from .storage import store_encrypted_video, restore_encrypted_video
from .ytf_runner import (RunnerConfig, OpenCVFeatureBackend, CachedMediaFeatures,
                         _score_matrix, run_ytf_runner)


class Cancelled(RuntimeError):
    pass


def run_job(store: JobStore, job_id: str, *, catalog=None, backend_type=OpenCVFeatureBackend,
            runner=run_ytf_runner):
    catalog = catalog or PlatformCatalog()
    job = store.get(job_id)
    config = job["config"]
    started = time.perf_counter()
    count = 0

    def checkpoint(stage=None):
        if store.get(job_id)["cancel_requested"]:
            raise Cancelled("用户请求取消")
        if stage:
            store.progress(job_id, stage, count, time.perf_counter() - started)

    class ProgressBackend(backend_type):
        def extract(self, record):
            nonlocal count
            mode = record.get("_experiment_mode", config["mode"])
            stage = f"{mode} · {record['split']} · {record['video_id']}"
            checkpoint(stage)
            result = super().extract(record)
            count += 1
            next_stage = (f"{mode} · 相似度评分 / 指标 / 报告"
                          if config["kind"] == "batch" and count % (config["total"] // 2) == 0
                          else "媒体特征已处理；剩余步骤包括评分/报告")
            store.progress(job_id, next_stage, count, time.perf_counter() - started)
            store.log(job_id, f"{stage} 完成；有效特征={len(result.features)}；提取失败={result.feature_extraction_failure_count}")
            checkpoint()
            return result

    with store.connect() as db:
        changed = db.execute("UPDATE experiment_jobs SET status='running',started_at=?,updated_at=?,pid=? WHERE id=? AND status='queued'", (utc_now(), utc_now(), os.getpid(), job_id)).rowcount
    if not changed:
        return
    store.log(job_id, "running：验证冻结输入摘要后启动；特征只在内存中处理。")
    TaskRepository(store.database).record_event(None, "experiment_running", job_id)
    try:
        checkpoint("验证数据与模型")
        manifest, split, scope = catalog.inputs(config["dataset_id"])
        model = recognition_config(config["model"], arcface_path=catalog.root / "models/arcface_w600k_r50.onnx", sface_path=catalog.root / "models/face_recognition_sface_2021dec.onnx")
        detection = FaceDetectionConfig(catalog.root / "models/face_detection_yunet_2023mar.onnx", frame_interval_seconds=1.0, score_threshold=.9)
        for path, key in ((manifest, "manifest_sha256"), (split, "split_sha256"), (model.model_path, "model_sha256"), (detection.model_path, "detector_sha256")):
            if sha256(path) != config[key]:
                raise ValueError("启动后输入或模型文件发生变化，拒绝运行")
        output = Path(job["output_dir"])
        # Isolated, random test keys: no persistent production keys and no logging.
        enc_key, mac_key = os.urandom(16), os.urandom(32)
        if config["kind"] == "batch":
            runner_config = RunnerConfig(manifest, split, output / "result", detection, model,
                                         top_k=config["top_k"], evaluation_scope=scope)
            runner(runner_config, encryption_key=enc_key, mac_key=mac_key, backend_factory=ProgressBackend)
        else:
            checkpoint("准备已登记Query及Gallery")
            records, assignment = catalog.records(config["dataset_id"])
            query = next(r for r in records if r["video_id"] == config["query_id"] and r["split"] == "query")
            # Keep retrieval identity-disjoint: candidates share Query's assigned split.
            galleries = [r for r in records if r["split"] == "gallery" and assignment[r["identity_id"]] == assignment[query["identity_id"]]]
            backend = ProgressBackend(detection, model)
            cache, availability, integrity = {}, [], []
            with tempfile.TemporaryDirectory(prefix="facecheck-retrieval-") as temp:
                for record in [query] + galleries:
                    checkpoint()
                    current = dict(record, _experiment_mode=config["mode"])
                    if config["mode"] == "encrypted_restore" and record["split"] == "gallery":
                        source = Path(record["video_path"])
                        stored = store_encrypted_video(source, Path(temp) / "cipher", enc_key, mac_key,
                                                       chunk_size=1024 * 1024, key_version="platform-test-v1", workers=4)
                        restored = Path(temp) / (record["video_id"] + ".mp4")
                        restore_encrypted_video(stored.directory, restored, enc_key, mac_key, workers=4)
                        verified = sha256(source) == sha256(restored)
                        if not verified:
                            raise ValueError("恢复 SHA-256 不一致")
                        integrity.append(dict(video_id=record["video_id"], sha256_equal=True))
                        current["video_path"] = str(restored)
                    features = backend.extract(current)
                    cache[record["video_id"]] = CachedMediaFeatures(record["identity_id"], record["video_id"], assignment[record["identity_id"]], record.get("media_type", "video"), features.features, features.sampled_frame_count)
                    availability.append(dict(video_id=record["video_id"], role=record["split"], feature_count=len(features.features), failures=features.feature_extraction_failure_count))
            checkpoint("1:N cosine/max 评分")
            scores, provenance, elapsed, _ = _score_matrix([query], galleries, cache, cache, 16, expected_dimension=model.expected_dimension)
            candidates = []
            for gallery in galleries:
                pair = (query["video_id"], gallery["video_id"])
                if scores[pair] is not None:
                    candidates.append(dict(video_id=gallery["video_id"], identity_id=gallery["identity_id"], similarity=scores[pair], **provenance.get(pair, {})))
            candidates.sort(key=lambda r: (-r["similarity"], r["video_id"]))
            result = dict(status="complete", exploratory=True, mode=config["mode"], query_id=query["video_id"],
                          gallery_count=len(galleries), unscored_pair_count=len(galleries) - len(candidates),
                          results=candidates[:config["top_k"]], feature_availability=availability,
                          restore_integrity=integrity, similarity_seconds=elapsed,
                          total_seconds=time.perf_counter() - started,
                          note="实时1:N；encrypted_restore 为解密恢复并验证后检索，不是密文域ANN。")
            checkpoint()
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "report.md").write_text("# 已登记数据集1:N实时检索\n\n" + result["note"] + "\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
        checkpoint("结果完整发布")
        store.progress(job_id, "完成", count, time.perf_counter() - started)
        store.finish(job_id, "completed")
    except Cancelled:
        store.progress(job_id, "已安全取消", count, time.perf_counter() - started)
        store.finish(job_id, "cancelled", "用户取消；未发布为完整结果。")
    except Exception as exc:
        store.progress(job_id, "失败", count, time.perf_counter() - started)
        store.finish(job_id, "failed", clean_error(exc))


def main():
    database, job_id = Path(sys.argv[1]), sys.argv[2]
    store = JobStore(database)
    # Also guard against multiple web processes/SQLite reconciliation races.
    with database.with_suffix(".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            store.finish(job_id, "failed", "另一工作进程持有运行锁")
            return
        run_job(store, job_id)


if __name__ == "__main__":
    main()
