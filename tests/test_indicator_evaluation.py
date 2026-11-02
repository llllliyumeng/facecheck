import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_indicators.py"
SPEC = importlib.util.spec_from_file_location("facecheck_indicator_evaluation", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


def test_current_benchmark_schema_and_scope_are_reported_without_overclaiming():
    metrics = {
        "plaintext": {
            "retrieval": {"query_count": 2, "top1_accuracy": {"value": .5}},
            "tar_far_test": {"tar": .5, "far": .0005},
        },
        "mode_consistency": {"consistent": True, "max_absolute_similarity_delta": 0.0},
    }
    benchmark = {
        "sm2": {"sign": {"median": 1200}, "verify": {"median": 600}},
        "sm4": {
            "envelope_sm4_hmac_sm3_encrypt": {"median": 100},
            "envelope_sm4_hmac_sm3_decrypt": {"median": 200},
        },
        "1080p": {"runs": [{"workers": 4, "encryption_ms": {"median": 10}}]},
        "resolution_matrix": [{"resolution": str(i)} for i in range(5)],
        "api_10_concurrent": {"success_rate": 1.0},
    }
    manifest = {"records": [
        {"identity_id": "id-1", "split": "gallery", "ytf_frame_count": 3},
        {"identity_id": "id-1", "split": "query"},
    ]}
    cloud = {"online_feature_cache": {
        "cold": {"http_seconds": 3.0}, "warm": {"http_seconds": .5},
    }}
    security = {"security_rejection": {
        "tamper_trials": 100, "tamper_detection_rate": 1.0,
        "unauthorized_trials": 100, "unauthorized_success_rate": 0.0,
    }}

    report = evaluator.build_report(metrics, benchmark, manifest, cloud, security)
    rows = {row["metric"]: row for row in report["rows"]}

    assert report["dataset"] == {
        "identities": 1, "gallery_videos": 1, "query_images": 1, "gallery_frames": 3,
    }
    assert rows["明文 1:N Top-1"]["value"] == "50.000%"
    assert rows["SM2 签名/验签吞吐"]["status"] == "PASS"
    assert rows["SM4 完整封装加/解密吞吐"]["status"] == "FAIL"
    assert rows["1080p 单路加密时延"]["status"] == "PARTIAL"
    assert rows["端到端上传/检索响应"]["status"] == "PARTIAL"
    assert rows["视频/特征篡改检出率"]["status"] == "PARTIAL"
    assert rows["未授权解密成功率"]["status"] == "PARTIAL"


def test_legacy_scalar_benchmark_schema_is_read_with_explicit_limitations():
    benchmark = {
        "sm2": {"sign_ops_per_second": 1100, "verify_ops_per_second": 510},
        "sm4": {"encrypt_mb_per_second": 40, "decrypt_mb_per_second": 45},
    }
    report = evaluator.build_report({}, benchmark)
    rows = {row["metric"]: row for row in report["rows"]}

    assert rows["SM2 签名/验签吞吐"]["status"] == "PASS"
    assert rows["SM4 完整封装加/解密吞吐"]["status"] == "FAIL"
    assert "旧格式未区分封装层级" in rows["SM4 完整封装加/解密吞吐"]["note"]


def test_key_management_smoke_test_runs_are_recognized_as_partial_evidence():
    evidence = {
        "status": "passed",
        "test_runs": [
            {"exit_code": 0, "tests_passed": 6},
            {"exit_code": 0, "tests_passed": 2},
        ],
    }

    report = evaluator.build_report({}, key_management_evidence=evidence)
    rows = {row["metric"]: row for row in report["rows"]}
    rotation = rows["密钥轮换后旧密钥失效"]
    rotation_research = next(
        row for row in report["research"] if row["area"] == "密钥派生与轮换"
    )

    assert rotation["status"] == "PARTIAL"
    assert "v2 per-video SM2 信封、单/批次 rewrap 原型" in rotation["value"]
    assert "file-backed journal" in rotation["note"]
    assert "file-backed 可恢复 batch rewrap 原型已实现" in rotation_research["note"]


def test_rotation_control_smoke_updates_scope_without_claiming_global_revocation():
    evidence = {
        "status": "passed",
        "execution": {"exit_code": 0, "tests_passed": 9},
    }

    report = evaluator.build_report(
        {}, sources={"rotation_control_smoke": {"path": "rotation-smoke.json"}},
        rotation_control_evidence=evidence,
    )
    rows = {row["metric"]: row for row in report["rows"]}
    rotation = rows["密钥轮换后旧密钥失效"]
    rotation_research = next(
        row for row in report["research"] if row["area"] == "密钥派生与轮换"
    )

    assert rotation["status"] == "PARTIAL"
    assert rotation["source"]["path"] == "rotation-smoke.json"
    assert "SQLite 单机队列/DB 引用切换" in rotation["value"]
    assert "旧密钥失效率 100% 尚未证明" in rotation["note"]
    assert "分布式队列" in rotation_research["note"]
