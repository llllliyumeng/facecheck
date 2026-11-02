#!/usr/bin/env python3
"""Repeatable crypto and video-path measurements (decimal MB)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import secrets
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


def _stats(values: list[float], unit: str) -> dict[str, float | str]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0, "unit": unit}


def _timed(fn: Callable[[], Any], rounds: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(rounds):
        started = time.perf_counter(); fn(); values.append(time.perf_counter() - started)
    return values


def _sm2(count: int, rounds: int, warmups: int) -> dict[str, Any]:
    from facecheck.crypto import sm2_sign, sm2_verify
    from gmssl import sm2

    curve = sm2.default_ecc_table
    private_value = secrets.randbelow(int(curve["n"], 16) - 1) + 1
    private_key = f"{private_value:064x}"
    public_key = sm2.CryptSM2(public_key="", private_key="")._kg(
        private_value, curve["g"]
    ).upper()
    message = b"FaceGuard indicator benchmark"
    random_values = [secrets.token_hex(32) for _ in range(count)]
    signatures: list[str] = []

    def sign() -> None:
        nonlocal signatures
        signatures = [sm2_sign(private_key, public_key, message, value) for value in random_values]

    def verify() -> None:
        assert sum(sm2_verify(public_key, message, value) for value in signatures) == count

    sign_times = _timed(sign, rounds, warmups); verify_times = _timed(verify, rounds, warmups)
    return {"count": count, "key_source": "random-process-local-test-key",
            "sign": _stats([count / value for value in sign_times], "ops/s"),
            "verify": _stats([count / value for value in verify_times], "ops/s"),
            "sign_seconds": _stats(sign_times, "seconds"), "verify_seconds": _stats(verify_times, "seconds"),
            "verified": count}


def _sm4(size_mib: int, rounds: int, warmups: int) -> dict[str, Any]:
    from facecheck.crypto import sm4_decrypt, sm4_encrypt
    from facecheck.crypto_backends import get_crypto_backend
    key, mac, plaintext = secrets.token_bytes(16), secrets.token_bytes(32), os.urandom(size_mib * 1024 * 1024)
    associated_data, backend, iv = b"indicator", get_crypto_backend(), secrets.token_bytes(16)
    primitive_ciphertext = backend.sm4_cbc_encrypt(key, iv, plaintext); envelope: bytes = b""

    def primitive_encrypt() -> None: backend.sm4_cbc_encrypt(key, iv, plaintext)
    def primitive_decrypt() -> None: backend.sm4_cbc_decrypt(key, iv, primitive_ciphertext)
    def hmac_sm3() -> None: backend.sm3_hmac(mac, plaintext)
    def envelope_encrypt() -> None:
        nonlocal envelope; envelope = sm4_encrypt(plaintext, key, mac, associated_data)
    def envelope_decrypt() -> None: assert sm4_decrypt(envelope, key, mac, associated_data) == plaintext

    timings = {"primitive_sm4_cbc_encrypt": _timed(primitive_encrypt, rounds, warmups),
               "primitive_sm4_cbc_decrypt": _timed(primitive_decrypt, rounds, warmups),
               "primitive_sm3_hmac": _timed(hmac_sm3, rounds, warmups),
               "envelope_sm4_hmac_sm3_encrypt": _timed(envelope_encrypt, rounds, warmups),
               "envelope_sm4_hmac_sm3_decrypt": _timed(envelope_decrypt, rounds, warmups)}
    size_mb = len(plaintext) / 1_000_000
    return {"size_mib": size_mib, "input_size_bytes": len(plaintext), "byte_unit": "MB_decimal",
            "backend": backend.name, **{name: _stats([size_mb / t for t in values], "MB/s_decimal")
            for name, values in timings.items()}, "round_trip": True}


def _write_1080p(path: Path) -> None:
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    if not writer.isOpened(): raise RuntimeError("OpenCV 1080p writer unavailable")
    frame = np.random.default_rng(7).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    try:
        for _ in range(2): writer.write(frame)
    finally: writer.release()


def _video_latency(rounds: int, warmups: int) -> dict[str, Any]:
    from facecheck.storage import store_encrypted_video
    key, mac = secrets.token_bytes(16), secrets.token_bytes(32)
    with tempfile.TemporaryDirectory(prefix="faceguard-indicator-") as root:
        source = Path(root) / "1080p.mp4"; _write_1080p(source); values = []
        for workers in (1, 4):
            def run() -> Any:
                return store_encrypted_video(source, Path(root) / f"storage-{workers}-{time.time_ns()}", key, mac,
                    chunk_size=1024 * 1024, key_version="indicator-v1", workers=workers)
            timings = _timed(run, rounds, warmups); source_bytes = source.stat().st_size
            values.append({"workers": workers, "source_bytes": source_bytes,
                "encryption_ms": _stats([t * 1000 for t in timings], "ms"),
                "mb_per_second": _stats([source_bytes / t / 1_000_000 for t in timings], "MB/s_decimal"),
                "chunk_count": run().manifest["chunk_count"]})
        return {"resolution": "1920x1080", "frames": 2, "runs": values}


def _resolution_matrix() -> list[dict[str, Any]]:
    import cv2
    import numpy as np
    from facecheck.storage import store_encrypted_video
    key, mac = secrets.token_bytes(16), secrets.token_bytes(32)
    rows = []
    with tempfile.TemporaryDirectory(prefix="faceguard-resolution-") as root:
        root_path = Path(root)
        for width, height in [(320, 240), (640, 480), (1280, 720), (1920, 1080), (2560, 1440)]:
            source = root_path / f"{width}x{height}.mp4"; writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))
            if not writer.isOpened(): rows.append({"resolution": f"{width}x{height}", "status": "writer_unavailable"}); continue
            writer.write(np.zeros((height, width, 3), dtype=np.uint8)); writer.release(); started = time.perf_counter()
            stored = store_encrypted_video(source, root_path / f"storage-{width}", key, mac, chunk_size=1024 * 1024, key_version="resolution-v1", workers=4)
            rows.append({"resolution": f"{width}x{height}", "status": "ok", "source_bytes": source.stat().st_size,
                         "encryption_ms": (time.perf_counter() - started) * 1000, "chunk_count": stored.manifest["chunk_count"]})
    return rows


def _api_concurrency(concurrency: int) -> dict[str, Any]:
    import concurrent.futures, cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from facecheck.main import create_app
    with tempfile.TemporaryDirectory(prefix="faceguard-api-indicator-") as root:
        root_path = Path(root); app = create_app(database_path=root_path / "db.sqlite3", storage_root=root_path / "storage",
            encryption_key=secrets.token_bytes(16), mac_key=secrets.token_bytes(32)); writer_path = root_path / "small.mp4"
        writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (16, 16)); writer.write(np.zeros((16, 16, 3), dtype=np.uint8)); writer.release(); payload = writer_path.read_bytes()
        def upload(_: int) -> int:
            with TestClient(app) as client: return client.post("/api/v1/videos", files={"file": ("sample.mp4", payload, "video/mp4")}).status_code
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool: statuses = list(pool.map(upload, range(concurrency)))
        return {"concurrency": concurrency, "requests": len(statuses), "successes": sum(status == 201 for status in statuses),
                "success_rate": sum(status == 201 for status in statuses) / len(statuses), "wall_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "python", "native"), default="auto"); parser.add_argument("--sm2-count", type=int, default=1000)
    parser.add_argument("--sm4-size-mib", type=int, default=16); parser.add_argument("--rounds", type=int, default=5); parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args(); os.environ["FACECHECK_CRYPTO_BACKEND"] = args.backend
    from facecheck.crypto_backends import get_crypto_backend
    get_crypto_backend.cache_clear(); backend = get_crypto_backend()
    report = {"environment": {"os": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
              "backend": backend.name, "backend_library": getattr(backend, "library_path", None),
              "gmssl_version": getattr(backend, "version", None),
              "sm4_native_impl": os.environ.get("FACECHECK_SM4_NATIVE_IMPL", "gmssl"),
              "rounds": args.rounds, "warmups": args.warmups},
              "sm2": _sm2(args.sm2_count, args.rounds, args.warmups), "sm4": _sm4(args.sm4_size_mib, args.rounds, args.warmups),
              "1080p": _video_latency(args.rounds, args.warmups), "resolution_matrix": _resolution_matrix(), "api_10_concurrent": _api_concurrency(10)}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["environment"]["process_peak_rss_bytes"] = int(peak_rss if platform.system() == "Darwin" else peak_rss * 1024)
    report["environment"]["memory_measurement"] = "process lifetime peak RSS; includes interpreter and imported libraries"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
