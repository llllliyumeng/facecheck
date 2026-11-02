"""Read-only, allowlisted course-platform inputs. Never discovers arbitrary paths."""
from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "ytf_smoke20": ("YTF 固定 20 身份探索集", "data/ytf/smoke_20/experiment_smoke_20.json", "data/ytf/smoke_20/identity_split_smoke_20.json", "full"),
    "ytf_selection": ("YTF 完整 selection（502 身份）", "data/ytf/experiment_full.json", "data/ytf/query_gallery_identity_split.json", "threshold_selection"),
}
HISTORIES = {
    "arcface_test": ("ArcFace 正式 test", "data/ytf/experiments/ytf_arcface_test_20261101", "formal_test"),
    "arcface_selection": ("ArcFace 完整 selection", "data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext", "selection"),
    "sface_baseline": ("SFace 正式历史基线（selection + test）", "data/ytf/experiments/ytf_full_run_02", "historical_full"),
    "models_sample100": ("SFace / ArcFace 100 身份探索", "data/ytf/experiments/ytf_models_sample_100_20261101", "exploratory"),
    "gallery_quality": ("Gallery 质量聚合失败实验留档", "data/ytf/experiments/ytf_gallery_quality_20261101/selection", "failed_strategy"),
}
ARTIFACT_NAMES = {
    "config.json", "metrics.json", "feature_summary.json", "scoring_diagnostics.json",
    "query_results.csv", "candidate_results.csv", "genuine_impostor_pairs.csv", "feature_failures.csv",
    "timing.csv", "report.md", "course_practice_report.md", "analysis.json", "validation.json",
    "model_comparison.csv", "query_diagnostics.csv", "failed_queries_plaintext.csv",
    "restore_sha256_audit.csv", "failure_summary.json", "result.json", "request.json", "identities.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_file(directory: Path, name: str) -> Path:
    if name not in ARTIFACT_NAMES:
        raise ValueError("文件不在结果白名单内")
    resolved = (directory / name).resolve()
    if resolved.parent != directory.resolve() or not resolved.is_file():
        raise FileNotFoundError(name)
    return resolved


@lru_cache(maxsize=32)
def file_hash(path: Path, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    stat = path.stat()
    return file_hash(path.resolve(), stat.st_size, stat.st_mtime_ns)


class PlatformCatalog:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()

    def inputs(self, dataset_id: str):
        _, manifest, split, scope = DATASETS[dataset_id]
        return self.root / manifest, self.root / split, scope

    def dataset(self, dataset_id: str) -> dict:
        name, manifest, split, scope = DATASETS[dataset_id]
        mpath, spath, _ = self.inputs(dataset_id)
        if not mpath.is_file() or not spath.is_file():
            return dict(id=dataset_id, name=name, available=False, manifest=manifest, scope=scope)
        data, assignments = read_json(mpath), read_json(spath)["assignments"]
        identities = {r["identity_id"] for r in assignments if scope == "full" or r["split"] == scope}
        records = [r for r in data["records"] if r["identity_id"] in identities]
        galleries = [r for r in records if r["split"] == "gallery"]
        return dict(id=dataset_id, name=name, available=True, manifest=manifest, identity_split=split,
                    scope=scope, identity_count=len(identities), gallery_count=len(galleries),
                    query_count=sum(r["split"] == "query" for r in records),
                    frame_count=sum(int(r.get("ytf_frame_count", 0)) for r in galleries),
                    frame_count_basis="manifest 中 Gallery 重建视频帧数之和，非已检测人脸数",
                    note=data.get("protocol_note", ""),
                    estimated_minutes="约 2–10 分钟，取决于模型/机器" if scope == "full" else "约 1–3 小时（两种模式）；仅供资源规划，非实测承诺",
                    modes=["plaintext", "encrypted_restore"],
                    model_options=["arcface", "sface"])

    def datasets(self):
        return [self.dataset(key) for key in DATASETS]

    def records(self, dataset_id: str):
        from .ytf_runner import _load_inputs
        manifest, split, scope = self.inputs(dataset_id)
        rows, assignments, _, _ = _load_inputs(manifest, split)
        return [r for r in rows if scope == "full" or assignments[r["identity_id"]] == scope], assignments

    def history_dir(self, history_id: str) -> Path:
        directory = self.root / HISTORIES[history_id][1]
        allowed = (self.root / "data/ytf/experiments").resolve()
        if not directory.resolve().is_relative_to(allowed):
            raise ValueError("历史目录越出登记范围")
        return directory

    def result(self, directory: Path) -> dict:
        artifacts = []
        for name in sorted(ARTIFACT_NAMES):
            try:
                path = safe_file(directory, name)
                artifacts.append(dict(name=name, bytes=path.stat().st_size))
            except (ValueError, FileNotFoundError):
                pass
        result: dict[str, Any] = {"artifacts": artifacts, "complete": False}
        for name in ("metrics", "config", "analysis", "validation", "result", "request"):
            if any(a["name"] == name + ".json" for a in artifacts):
                result[name] = read_json(safe_file(directory, name + ".json"))
        metrics = result.get("metrics", {})
        names = {a["name"] for a in artifacts}
        bundle_files = {"config.json", "metrics.json", "query_results.csv", "candidate_results.csv",
                        "genuine_impostor_pairs.csv", "feature_summary.json", "feature_failures.csv", "timing.csv", "report.md"}
        result["complete"] = bool((metrics and (
            metrics.get("status") == "complete" or ("plaintext" in metrics and bundle_files <= names)))
            or result.get("result", {}).get("status") == "complete")
        # The archived two-model exploration uses a different, pre-existing bundle schema.
        sample_size = result.get("config", {}).get("sample_size")
        models = metrics.get("models", {})
        if sample_size and {"config.json", "metrics.json", "identities.json", "report.md"} <= names:
            identities = read_json(safe_file(directory, "identities.json")).get("identity_ids", [])
            result["complete"] = (set(models) == {"sface", "arcface"}
                and len(set(identities)) == len(identities) == sample_size
                and all(model.get("retrieval", {}).get("query_count") == sample_size
                        and model.get("pair_count") == sample_size ** 2 for model in models.values()))
        # Counts only; raw private embedding vectors are never published by Runner.
        summary = directory / "feature_summary.json"
        if summary.is_file() and not summary.is_symlink():
            data = read_json(summary)
            if isinstance(data, dict):
                result["feature_counts"] = {mode: dict(media=len(rows), with_features=sum(r.get("feature_count", 0) > 0 for r in rows),
                    without_features=sum(r.get("feature_count", 0) == 0 for r in rows),
                    failures=sum(r.get("feature_extraction_failure_count", 0) for r in rows))
                    for mode, rows in data.items() if isinstance(rows, list)}
        return result

    def history(self, history_id: str) -> dict:
        name, path, kind = HISTORIES[history_id]
        result = self.result(self.history_dir(history_id))
        return dict(id=history_id, name=name, path=path, kind=kind, readonly=True, **result)

    def histories(self):
        return [self.history(key) for key in HISTORIES]

    def benchmark_paths(self):
        # Restricted to the immutable indicator output naming convention.
        return {p.name: p for p in sorted((self.root / "results").glob("indicator_benchmark_*.json"))
                if p.is_file() and p.resolve().parent == (self.root / "results").resolve()}

    def benchmarks(self):
        return [dict(id=name, source="results/" + name, data=read_json(path))
                for name, path in self.benchmark_paths().items()]

    def requirements(self):
        arc = self.history("arcface_test")
        metrics = arc.get("metrics", {})
        plain = metrics.get("plaintext", {})
        retrieval, tf = plain.get("retrieval", {}), plain.get("tar_far_test", {})
        source = HISTORIES["arcface_test"][1] + "/metrics.json"
        bench_path = self.root / "results/indicator_benchmark_20261101_sm4_final.json"
        bench = read_json(bench_path) if bench_path.is_file() else {}
        bsource = "results/indicator_benchmark_20261101_sm4_final.json"
        old_path = self.root / "results/indicator_benchmark_20261031.json"
        security = read_json(old_path).get("security_rejection", {}) if old_path.is_file() else {}
        cloud_path = self.root / "results/cloud_validation_20261101_v2.json"
        cloud = read_json(cloud_path) if cloud_path.is_file() else {}
        csource = "results/cloud_validation_20261101_v2.json"
        rows = []
        def row(label, target, actual=None, status="未测试", evidence="尚无符合口径的测试", note=""):
            rows.append(dict(requirement=label, target=target, actual=actual, status=status, source=evidence, note=note))
        ds = self.dataset("ytf_selection")
        row("数据规模", "≥1万人、10万段视频、100万帧", f"test {retrieval.get('query_count', '未测试')} 身份；selection {ds.get('identity_count', '未测试')} 身份", "未达标" if retrieval else "未测试", source, "502 test identities 不满足万人目标；Gallery 是重建视频，Query 是图像。")
        for label, key, target in (("明文 Top-1", "top1_accuracy", .95),):
            value = retrieval.get(key, {}).get("value")
            row(label, "≥95%", value, "已达标" if value is not None and value >= target else "未达标" if value is not None else "未测试", source)
        value = tf.get("tar")
        row("TAR@FAR=1e-3", "≥90%", value, "已达标" if value is not None and value >= .9 and tf.get("operating_point_met") else "未达标" if value is not None else "未测试", source,
            f"实际 FAR={tf.get('far')}；阈值仅由独立 selection 选取。")
        row("密文域 Top-1 下降", "≤2个百分点", status="未实现", note="encrypted_restore 为解密恢复后检索；不能替代密文域指标。")
        row("密文域结果一致率", "≥99.9%", status="未实现", note=f"恢复后检索一致性证据：{metrics.get('mode_consistency', {})}，不等同密文域检索。")
        sm4 = bench.get("sm4", {})
        values = [sm4.get(k, {}).get("median") for k in ("envelope_sm4_hmac_sm3_encrypt", "envelope_sm4_hmac_sm3_decrypt")]
        row("SM4 完整加/解密吞吐", "均≥500 MB/s（十进制）", values if all(v is not None for v in values) else None,
            "已达标" if all(v is not None and v >= 500 for v in values) else "未达标" if all(v is not None for v in values) else "未测试", bsource,
            "CBC+HMAC-SM3，多轮中位数；其他构建记录在性能页分别展示。")
        for key, label, goal in (("sign", "SM2 签名", 1000), ("verify", "SM2 验签", 500)):
            value = bench.get("sm2", {}).get(key, {}).get("median")
            row(label, f"≥{goal} 次/s", value, "已达标" if value is not None and value >= goal else "未达标" if value is not None else "未测试", bsource, "本机 benchmark，非云端承诺。")
        sample_1080p = next((r for r in bench.get("1080p", {}).get("runs", []) if r.get("workers") == 4), None)
        sample_ms = (sample_1080p or {}).get("encryption_ms", {}).get("median")
        row("1080p 单路视频加密", "≤50 ms", sample_ms, "部分验证" if sample_ms is not None else "未测试",
            bsource, "已有两帧合成样本、4线程计时；不足以证明持续单路视频目标。")
        online = cloud.get("online_feature_cache", {})
        cold = online.get("cold", {}).get("http_seconds")
        warm = online.get("warm", {}).get("http_seconds")
        latency = {"cold_seconds": cold, "warm_seconds": warm} if cold is not None or warm is not None else None
        row("上传与检索端到端平均响应", "≤2 s", latency,
            "部分验证" if latency else "未测试", csource if latency else "尚无符合口径的测试",
            "冷/热各一个固定样本；冷启动样本超过2秒，不能估计总体平均或P95。")
        upload = cloud.get("ten_concurrent_https_uploads", {})
        row("终端/分辨率/并发", "≥3类终端、5种分辨率、10路并发",
            {"resolutions": len(bench.get("resolution_matrix", [])),
             "local_api": bench.get("api_10_concurrent"),
             "https_upload": upload or None},
            "部分验证" if upload or bench.get("resolution_matrix") else "未测试",
            csource if upload else bsource,
            "有本机合成分辨率和10请求证据；3类真实终端及持续10路推理未测。")
        row("轮换后旧密钥失效", "100%", "SQLite 单机轮换队列、DB 引用切换、缓存清理及撤销门槛原型", "部分验证",
            evidence="src/facecheck/rotation_control.py",
            note="真实 KMS/HSM、分布式队列、全部备份/灾备清单和全局撤销均未验收；100% 尚未证明。")
        row("未授权解密成功率", "≤0.1%",
            {"success_rate": security.get("unauthorized_success_rate"), "trials": security.get("unauthorized_trials")},
            "部分验证" if security.get("unauthorized_trials") is not None else "未测试",
            "results/indicator_benchmark_20261031.json",
            "历史错误密钥拒绝试验不覆盖API越权、多租户和密钥失陷。")
        row("视频与特征篡改检出率", "≥99.9%",
            {"video_detection_rate": security.get("tamper_detection_rate"),
             "video_trials": security.get("tamper_trials"), "feature_detection": None},
            "部分验证" if security.get("tamper_trials") is not None else "未测试",
            "results/indicator_benchmark_20261031.json",
            "视频分块随机篡改有100次样本；特征检出总体率未测。")
        api_rate = bench.get("api_10_concurrent")
        row("系统接口调用成功率", "≥99.5%",
            {"local_api": api_rate, "https_upload": upload or None},
            "部分验证" if api_rate or upload else "未测试",
            csource if upload else bsource,
            "10个本机请求/合成上传只作功能证据，不是长期可用性统计。")
        row("日均处理能力", "≥10万段或100万帧")
        row("自动审计报告成功率", "≥99%", "按需审计导出接口已实现", "部分验证",
            note="功能验证不等于长期运行成功率。")
        row("审计日志完整率", "≥99.9%", note="本地 SHA-256 事件链及校验接口已实现；无独立外部锚点，尾部截断/整库回滚无法由本地链证明，完整率未测。")
        research = [
            ("SM2/SM4 数字信封", "原型阶段", "opt-in v2 信封、SHA-256 pin 公钥目录和 provider adapter 接口已实现；Web/cloud 默认写 v1，未部署可信目录或真实 KMS。"),
            ("视频分块与选择性加密", "部分实现", "全视频分块 SM4-CBC + HMAC-SM3；选择性加密未实现。"),
            ("密钥派生与轮换", "原型阶段", "v1 重加密、v2 rewrap、单机 SQLite 队列、DB 内引用切换和缓存失效原型已实现；无生产分布式队列/KMS，旧密钥全局失效未证明。"),
            ("特征保护与可撤销模板", "未实现", "特征仅内存使用；无可撤销模板。"),
            ("密文域索引与 ANN", "未实现", "恢复认证后进行明文 cosine/max 检索，无密文域 ANN。"),
            ("边缘-云协同推理", "当前原型不支持", "当前为本机 OpenCV DNN CPU 推理。"),
            ("完整性与审计追溯", "部分实现", "块 MAC、恢复 SHA-256、本地审计 SHA-256 链、快照校验和带摘要导出；无外部锚点，防删证明和完整率未测。"),
            ("平台集成", "已实现", "视频演示、独立进程批量任务、只读历史结果、指标对照。"),
        ]
        return dict(indicators=rows, research=[dict(name=n, status=s, description=d) for n, s, d in research],
                    scope_note="本机课程原型；未测试为 null，非零。研究目标与实现状态分别列示。")
