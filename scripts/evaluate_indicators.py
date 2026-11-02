#!/usr/bin/env python3
"""Compare saved FaceCheck evidence with the course acceptance targets.

The evaluator accepts both early scalar benchmark files and the current
multi-round benchmark schema. It never runs recognition or benchmarks, and it
labels small-sample evidence as partial rather than treating a point result as
a production reliability estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _stat(value: Any, name: str = "median") -> Optional[float]:
    """Read a scalar or a named statistic from old and current records."""
    if isinstance(value, dict):
        return _number(value.get(name))
    return _number(value)


def _at(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _fmt(value: Optional[float], suffix: str = "", digits: int = 3) -> str:
    return "not measured" if value is None else f"{value:.{digits}f}{suffix}"


def _fmt_percent(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value * 100:.3f}%"


def _row(
    rows: list[dict[str, Any]], metric: str, value: Any, target: str,
    status: str, source: str, note: str = "",
) -> None:
    rows.append({
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "source": source,
        "note": note,
    })


def _dataset_counts(manifest: dict[str, Any]) -> dict[str, int]:
    records = manifest.get("records")
    if not isinstance(records, list):
        return {}
    rows = [row for row in records if isinstance(row, dict)]
    gallery = [row for row in rows if row.get("split") == "gallery"]
    query = [row for row in rows if row.get("split") == "query"]
    return {
        "identities": len({row.get("identity_id") for row in rows if row.get("identity_id") is not None}),
        "gallery_videos": len(gallery),
        "query_images": len(query),
        "gallery_frames": sum(
            int(row.get("ytf_frame_count", 0))
            for row in gallery
            if isinstance(row.get("ytf_frame_count", 0), (int, float))
        ),
    }


def build_report(
    metrics: dict[str, Any],
    benchmark: Optional[dict[str, Any]] = None,
    manifest: Optional[dict[str, Any]] = None,
    cloud_validation: Optional[dict[str, Any]] = None,
    security_benchmark: Optional[dict[str, Any]] = None,
    sources: Optional[dict[str, str]] = None,
    key_management_evidence: Optional[dict[str, Any]] = None,
    rotation_control_evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    cloud_validation = cloud_validation if isinstance(cloud_validation, dict) else {}
    security_benchmark = security_benchmark if isinstance(security_benchmark, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    key_management_evidence = (
        key_management_evidence if isinstance(key_management_evidence, dict) else {}
    )
    rotation_control_evidence = (
        rotation_control_evidence if isinstance(rotation_control_evidence, dict) else {}
    )
    execution = key_management_evidence.get("execution")
    test_runs = key_management_evidence.get("test_runs")
    if isinstance(execution, dict):
        execution_passed = (
            _number(execution.get("tests_passed")) is not None
            and _number(execution.get("exit_code")) == 0
        )
    elif isinstance(test_runs, list) and test_runs:
        execution_passed = all(
            isinstance(run, dict)
            and _number(run.get("tests_passed")) is not None
            and _number(run.get("tests_passed")) > 0
            and _number(run.get("exit_code")) == 0
            for run in test_runs
        )
    else:
        execution_passed = False
    key_management_smoke_passed = (
        key_management_evidence.get("status") == "passed" and execution_passed
    )
    rotation_control_execution = rotation_control_evidence.get("execution")
    rotation_control_test_runs = rotation_control_evidence.get("test_runs")
    if isinstance(rotation_control_execution, dict):
        rotation_control_execution_passed = (
            _number(rotation_control_execution.get("tests_passed")) is not None
            and _number(rotation_control_execution.get("exit_code")) == 0
        )
    elif isinstance(rotation_control_test_runs, list) and rotation_control_test_runs:
        rotation_control_execution_passed = all(
            isinstance(run, dict)
            and _number(run.get("tests_passed")) is not None
            and _number(run.get("tests_passed")) > 0
            and _number(run.get("exit_code")) == 0
            for run in rotation_control_test_runs
        )
    else:
        rotation_control_execution_passed = False
    rotation_control_smoke_passed = (
        rotation_control_evidence.get("status") == "passed"
        and rotation_control_execution_passed
    )
    rows: list[dict[str, Any]] = []
    plain = _at(metrics, "plaintext", default={})
    retrieval = _at(plain, "retrieval", default={})
    tar_far = _at(plain, "tar_far_test", default={})
    query_count = _number(_at(retrieval, "query_count"))

    counts = _dataset_counts(manifest)
    if counts:
        meets_scale = (
            counts["identities"] >= 10_000
            and counts["gallery_videos"] >= 100_000
            and counts["gallery_frames"] >= 1_000_000
        )
        _row(
            rows, "综合测试集规模",
            {**counts, "formal_test_queries": int(query_count or 0)},
            "≥1万人、10万段视频、100万帧",
            "PASS" if meets_scale else "FAIL",
            sources.get("manifest", "dataset manifest"),
            "manifest 规模是数据覆盖证据；合成或重复样本不得用于替代真实身份覆盖。",
        )
    else:
        _row(rows, "综合测试集规模", "not measured", "≥1万人、10万段视频、100万帧",
             "NOT_MEASURED", sources.get("manifest", "dataset manifest unavailable"))

    top1 = _number(_at(retrieval, "top1_accuracy", "value"))
    _row(rows, "明文 1:N Top-1", _fmt_percent(top1), "≥95%",
         "PASS" if top1 is not None and top1 >= .95 else "FAIL" if top1 is not None else "NOT_MEASURED",
         sources.get("metrics", "formal metrics"),
         f"正式 Query 数：{int(query_count or 0)}；分母保留未评分 Query。")

    tar = _number(_at(tar_far, "tar"))
    far = _number(_at(tar_far, "far"))
    tar_pass = tar is not None and tar >= .90 and far is not None and far <= .001
    _row(rows, "明文 TAR@FAR≤1e-3", _fmt_percent(tar), "TAR≥90%，FAR≤1e-3",
         "PASS" if tar_pass else "FAIL" if tar is not None else "NOT_MEASURED",
         sources.get("metrics", "formal metrics"),
         f"实际 FAR={'not measured' if far is None else f'{far:.9f}'}；阈值须来自独立 selection。")

    consistency = _at(metrics, "mode_consistency", default={})
    _row(rows, "密文域 Top-1 下降", "not implemented", "≤2个百分点",
         "NOT_IMPLEMENTED", sources.get("metrics", "formal metrics"),
         "当前 encrypted_restore 先解密恢复再做明文检索。")
    _row(rows, "密文域结果与明文一致率", consistency or "not measured", "≥99.9%",
         "PARTIAL" if consistency else "NOT_MEASURED",
         sources.get("metrics", "formal metrics"),
         "当前一致性只比较认证恢复后的明文检索，不是密文域 ANN。")

    sm4 = _at(benchmark, "sm4", default={})
    enc = _stat(_at(sm4, "envelope_sm4_hmac_sm3_encrypt"))
    dec = _stat(_at(sm4, "envelope_sm4_hmac_sm3_decrypt"))
    legacy_sm4 = enc is None and dec is None
    if legacy_sm4:
        enc = _number(_at(sm4, "encrypt_mb_per_second"))
        dec = _number(_at(sm4, "decrypt_mb_per_second"))
    _row(rows, "SM4 完整封装加/解密吞吐",
         {"encrypt_MB_s": enc, "decrypt_MB_s": dec}, "均≥500 MB/s（十进制）",
         "PASS" if enc is not None and dec is not None and min(enc, dec) >= 500
         else "FAIL" if enc is not None and dec is not None else "NOT_MEASURED",
         sources.get("benchmark", "benchmark"),
         "旧格式未区分封装层级；当前格式为完整 CBC+HMAC-SM3 多轮中位数。"
         if legacy_sm4 else "完整 CBC+HMAC-SM3 多轮中位数；原语与文件 I/O 另行计量。")

    sm2 = _at(benchmark, "sm2", default={})
    sign = _stat(_at(sm2, "sign"))
    verify = _stat(_at(sm2, "verify"))
    if sign is None:
        sign = _number(_at(sm2, "sign_ops_per_second"))
    if verify is None:
        verify = _number(_at(sm2, "verify_ops_per_second"))
    _row(rows, "SM2 签名/验签吞吐",
         {"sign_ops_s": sign, "verify_ops_s": verify}, "≥1000/500 次/s",
         "PASS" if sign is not None and verify is not None and sign >= 1000 and verify >= 500
         else "FAIL" if sign is not None and verify is not None else "NOT_MEASURED",
         sources.get("benchmark", "benchmark"), "本机 API 基准，不代表云端并发保证。")

    video = _at(benchmark, "1080p", default={})
    runs = video.get("runs", []) if isinstance(video, dict) else []
    four_workers = next((item for item in runs if isinstance(item, dict) and item.get("workers") == 4), None)
    sample_ms = _stat(_at(four_workers, "encryption_ms"))
    _row(rows, "1080p 单路加密时延", _fmt(sample_ms, " ms"), "≤50 ms",
         "PARTIAL" if sample_ms is not None else "NOT_MEASURED",
         sources.get("benchmark", "benchmark"),
         "当前记录为两帧合成 1080p 文件；不能证明持续单路码流。")

    online = _at(cloud_validation, "online_feature_cache", default={})
    cold = _number(_at(online, "cold", "http_seconds"))
    warm = _number(_at(online, "warm", "http_seconds"))
    latency = {"cold_seconds": cold, "warm_seconds": warm}
    _row(rows, "端到端上传/检索响应", latency, "平均≤2 s",
         "PARTIAL" if cold is not None or warm is not None else "NOT_MEASURED",
         sources.get("cloud_validation", "cloud validation"),
         "冷/热各一个固定样本；冷启动超2秒，不能据此估计总体平均或P95。")

    resolutions = _at(benchmark, "resolution_matrix", default=[])
    api = _at(benchmark, "api_10_concurrent", default={})
    concurrent_cloud = _at(cloud_validation, "ten_concurrent_https_uploads", default={})
    concurrency = {
        "synthetic_resolution_count": len(resolutions) if isinstance(resolutions, list) else 0,
        "local_api_success_rate": _number(_at(api, "success_rate")),
        "https_upload_success_rate": _number(_at(concurrent_cloud, "success_rate")),
        "https_upload_requests": _number(_at(concurrent_cloud, "requests")),
    }
    _row(rows, "终端/分辨率/并发支持", concurrency,
         "≥3类终端、5种分辨率、10路并发",
         "PARTIAL" if concurrency["synthetic_resolution_count"] or concurrency["https_upload_requests"]
         else "NOT_MEASURED",
         sources.get("cloud_validation", "benchmark and cloud validation"),
         "已做的 5 分辨率和 10 请求均为本机功能样本；三类真实终端及持续推理并发未测。")

    security = _at(security_benchmark, "security_rejection", default={})
    tamper_trials = _number(_at(security, "tamper_trials"))
    tamper_rate = _number(_at(security, "tamper_detection_rate"))
    unauthorized_trials = _number(_at(security, "unauthorized_trials"))
    unauthorized_rate = _number(_at(security, "unauthorized_success_rate"))
    rotation_value = (
        "v1 重加密、v2 rewrap、SQLite 单机队列/DB 引用切换、缓存清理和撤销门槛原型"
        if rotation_control_smoke_passed else
        "v1 单视频重加密；v2 per-video SM2 信封、单/批次 rewrap 原型"
        if key_management_smoke_passed else "版本隔离与单视频迁移助手"
    )
    rotation_note = (
        "小样本验证了 SQLite 持久队列、DB 内引用与 manifest 锚点事务切换、缓存清理和清单门槛；"
        "这不是分布式队列或跨存储原子事务。未接入生产 KMS/HSM，外部备份完整性仍依赖运维清单，"
        "没有真实全局撤销证明；旧密钥失效率 100% 尚未证明。"
        if rotation_control_smoke_passed else
        "小样本验证了 file-backed journal 续跑和新目标完整性；仍未接入服务数据库/任务队列，"
        "没有 KMS、缓存/备份处置或全局旧密钥撤销证明。"
        if key_management_smoke_passed else
        "没有批量迁移清单、全局撤销或 KMS 轮换证明。"
    )
    rotation_source_name = (
        "rotation_control_smoke" if rotation_control_smoke_passed
        else "key_management_smoke"
    )
    _row(rows, "密钥轮换后旧密钥失效", rotation_value,
         "全量旧密钥失效率=100%", "PARTIAL",
         sources.get(rotation_source_name, sources.get("code", "storage/key management")),
         rotation_note)
    _row(rows, "未授权解密成功率",
         {"success_rate": unauthorized_rate, "trials": unauthorized_trials}, "≤0.1%",
         "PARTIAL" if unauthorized_rate is not None else "NOT_MEASURED",
         sources.get("security_benchmark", "security benchmark"),
         "错误密钥小样本不能替代 RBAC、跨租户和密钥泄露攻击面验收。")
    _row(rows, "视频/特征篡改检出率",
         {"video_tamper_detection_rate": tamper_rate, "trials": tamper_trials}, "≥99.9%",
         "PARTIAL" if tamper_rate is not None else "NOT_MEASURED",
         sources.get("security_benchmark", "security benchmark"),
         "当前随机分块篡改样本不覆盖特征索引或数据库与密钥同时受控。")

    _row(rows, "接口调用成功率",
         {"local_api": _number(_at(api, "success_rate")),
          "https_upload": _number(_at(concurrent_cloud, "success_rate"))}, "≥99.5%",
         "PARTIAL" if _at(api, "success_rate") is not None or _at(concurrent_cloud, "success_rate") is not None
         else "NOT_MEASURED",
         sources.get("cloud_validation", "benchmark and cloud validation"),
         "仅小样本本机请求；没有持续可用性压测。")
    _row(rows, "日均处理能力", "not measured", "≥10万段视频或100万帧",
         "NOT_MEASURED", sources.get("cloud_validation", "no sustained load evidence"))
    _row(rows, "自动审计报告生成成功率", "按需导出接口已实现；成功率未测", "≥99%",
         "PARTIAL", sources.get("code", "platform audit export"),
         "功能验证不等于长期运行成功率。")
    _row(rows, "审计日志完整率", "本地哈希链校验原型已实现；完整率未测", "≥99.9%",
         "NOT_MEASURED", sources.get("code", "local SQLite audit chain prototype"),
         "无独立外部锚点；不能证明尾部未截断或全库未回滚，完整率仍需外部对账和持续故障注入测量。")

    envelope_note = (
        "小样本验证了 opt-in v2 逐视频随机 key material、SM2 接收方封装和 HMAC-SM3 manifest；"
        "代码有 SHA-256 pin 公钥目录/provider adapter 接口，但无生产信任部署或 KMS；Web/cloud 默认仍写 v1。"
        if key_management_smoke_passed else
        "SM2 API 与 SM4 封装已存在；Web 主流程未使用逐视频 DEK/接收者公钥分发。"
    )
    rotation_research_note = (
        "v1 重加密、v2 rewrap、单机 SQLite 队列、DB 内引用/manifest 锚点事务切换及缓存失效有原型；"
        "生产 KMS/HSM、分布式队列、外部备份证明和真实全局撤销未验收。"
        if rotation_control_smoke_passed else
        "v1 单视频重加密与 v2 file-backed 可恢复 batch rewrap 原型已实现；"
        "服务级任务/DB 切换、全局撤销及缓存和备份处置未实现。"
        if key_management_smoke_passed else
        "版本化派生和单视频迁移已实现；自动批量轮换、KMS 与全局撤销未实现。"
    )
    research = [
        {"area": "SM2/SM4 数字信封", "status": "PARTIAL", "note": envelope_note},
        {"area": "视频分块与选择性加密", "status": "PARTIAL",
         "note": "全视频字节分块加密已实现；编解码器感知的选择性加密未实现。"},
        {"area": "密钥派生与轮换", "status": "PARTIAL", "note": rotation_research_note},
        {"area": "人像特征保护与可撤销模板", "status": "PARTIAL",
         "note": "静态缓存加密已实现；可撤销、不可关联模板未实现。"},
        {"area": "密文安全索引与 ANN", "status": "NOT_IMPLEMENTED",
         "note": "当前为认证恢复后明文精确检索。"},
        {"area": "边缘-云协同推理", "status": "NOT_IMPLEMENTED",
         "note": "没有边缘设备部署与通信协议验收。"},
        {"area": "完整性与审计追溯", "status": "PARTIAL",
         "note": "分块认证、恢复 SHA-256、本地审计 SHA-256 链及快照校验已实现；无外部锚点，防删证明和完整率未测。"},
        {"area": "平台集成", "status": "PARTIAL",
         "note": "课程平台已实现；远程多租户、持续负载和规模化部署未测。"},
    ]
    return {
        "protocol": "facecheck-course-indicator-review-v1",
        "statuses": ["PASS", "FAIL", "PARTIAL", "NOT_MEASURED", "NOT_IMPLEMENTED"],
        "dataset": counts,
        "rows": rows,
        "research": research,
        "sources": sources,
        "scope_note": "本报告只读取已保存证据；没有运行正式识别实验或新性能基准。",
    }


def _source_digest(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SystemExit(f"cannot hash evidence {path}: {exc}") from exc
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, help="saved formal recognition metrics")
    parser.add_argument("--benchmark", type=Path, help="saved multi-round crypto benchmark")
    parser.add_argument("--manifest", type=Path, help="registered dataset manifest for scale counts")
    parser.add_argument("--cloud-validation", type=Path, help="saved cloud integration validation")
    parser.add_argument("--security-benchmark", type=Path,
                        help="saved error-key/tamper trials; kept separate from throughput runs")
    parser.add_argument("--key-management-evidence", type=Path,
                        help="saved small functional verification of the v2 key-envelope prototype")
    parser.add_argument("--rotation-control-evidence", type=Path,
                        help="saved small functional verification of SQLite rotation control")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_paths = {
        key: path for key, path in {
            "metrics": args.metrics,
            "benchmark": args.benchmark,
            "manifest": args.manifest,
            "cloud_validation": args.cloud_validation,
            "security_benchmark": args.security_benchmark,
            "key_management_smoke": args.key_management_evidence,
            "rotation_control_smoke": args.rotation_control_evidence,
        }.items() if path is not None
    }
    sources = {
        key: {
            "path": str(path),
            "sha256": _source_digest(path),
        }
        for key, path in input_paths.items()
    }
    payload = json.dumps(
        build_report(
            _load(args.metrics), _load(args.benchmark), _load(args.manifest),
            _load(args.cloud_validation), _load(args.security_benchmark), sources,
            _load(args.key_management_evidence), _load(args.rotation_control_evidence),
        ),
        ensure_ascii=False, indent=2,
    ) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
