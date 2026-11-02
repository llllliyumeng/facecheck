#!/usr/bin/env python3
"""Compare SM4 envelope chunk sizes using process-local random test data."""
from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable


CHUNK_SIZES = (64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)


def _median_time(fn: Callable[[], object], rounds: int, warmups: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "stddev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-mib", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("FACECHECK_CRYPTO_BACKEND", "native")

    from facecheck.crypto import sm4_decrypt, sm4_encrypt
    from facecheck.crypto_backends import get_crypto_backend
    from facecheck.storage import store_encrypted_video

    get_crypto_backend.cache_clear()
    backend = get_crypto_backend()
    total = args.total_mib * 1024 * 1024
    if args.total_mib <= 0:
        raise SystemExit("--total-mib must be positive")
    plaintext = os.urandom(total)
    encryption_key, mac_key = secrets.token_bytes(16), secrets.token_bytes(32)
    rows = []

    with tempfile.TemporaryDirectory(prefix="facecheck-sm4-chunks-") as temp:
        root = Path(temp)
        source = root / "random-input.bin"
        source.write_bytes(plaintext)
        for chunk_size in CHUNK_SIZES:
            chunks = [plaintext[offset : offset + chunk_size]
                      for offset in range(0, total, chunk_size)]
            ivs = [secrets.token_bytes(16) for _ in chunks]
            aad = [b"chunk-index=" + index.to_bytes(4, "big") for index in range(len(chunks))]
            ciphertexts = [backend.sm4_cbc_encrypt(encryption_key, iv, part)
                           for iv, part in zip(ivs, chunks)]
            envelopes = [sm4_encrypt(part, encryption_key, mac_key, context)
                         for part, context in zip(chunks, aad)]
            restored = b"".join(
                sm4_decrypt(envelope, encryption_key, mac_key, context)
                for envelope, context in zip(envelopes, aad)
            )
            if restored != plaintext:
                raise RuntimeError(f"chunk round trip failed at {chunk_size} bytes")

            def primitive_encrypt() -> None:
                for iv, part in zip(ivs, chunks):
                    backend.sm4_cbc_encrypt(encryption_key, iv, part)

            def primitive_decrypt() -> None:
                for iv, part in zip(ivs, ciphertexts):
                    backend.sm4_cbc_decrypt(encryption_key, iv, part)

            def hmac_parts() -> None:
                for part in chunks:
                    backend.sm3_hmac(mac_key, part)

            def envelope_encrypt() -> None:
                for part, context in zip(chunks, aad):
                    sm4_encrypt(part, encryption_key, mac_key, context)

            def envelope_decrypt() -> None:
                for envelope, context in zip(envelopes, aad):
                    sm4_decrypt(envelope, encryption_key, mac_key, context)

            primitive_enc = _median_time(primitive_encrypt, args.rounds, args.warmups)
            primitive_dec = _median_time(primitive_decrypt, args.rounds, args.warmups)
            hmac_time = _median_time(hmac_parts, args.rounds, args.warmups)
            envelope_enc = _median_time(envelope_encrypt, args.rounds, args.warmups)
            envelope_dec = _median_time(envelope_decrypt, args.rounds, args.warmups)

            storage_runs = {}
            for workers in (1, args.workers):
                def storage_encrypt() -> None:
                    destination = root / f"storage-{chunk_size}-{workers}-{time.time_ns()}"
                    store_encrypted_video(
                        source,
                        destination,
                        encryption_key,
                        mac_key,
                        chunk_size=chunk_size,
                        key_version="benchmark-v1",
                        workers=workers,
                    )

                storage_time = _median_time(storage_encrypt, args.rounds, args.warmups)
                storage_runs[str(workers)] = {
                    **storage_time,
                    "throughput_MB_s": total / 1_000_000 / storage_time["median_seconds"],
                }

            rows.append({
                "chunk_size_bytes": chunk_size,
                "chunk_count": len(chunks),
                "primitive_cbc_encrypt": {
                    **primitive_enc,
                    "throughput_MB_s": total / 1_000_000 / primitive_enc["median_seconds"],
                },
                "primitive_cbc_decrypt": {
                    **primitive_dec,
                    "throughput_MB_s": total / 1_000_000 / primitive_dec["median_seconds"],
                },
                "hmac_sm3_per_chunk": {
                    **hmac_time,
                    "throughput_MB_s": total / 1_000_000 / hmac_time["median_seconds"],
                },
                "envelope_encrypt": {
                    **envelope_enc,
                    "throughput_MB_s": total / 1_000_000 / envelope_enc["median_seconds"],
                },
                "envelope_decrypt": {
                    **envelope_dec,
                    "throughput_MB_s": total / 1_000_000 / envelope_dec["median_seconds"],
                },
                "storage_encrypt": storage_runs,
            })

    report = {
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "backend": backend.name,
            "backend_library": getattr(backend, "library_path", None),
            "total_input_bytes": total,
            "byte_unit": "MB_decimal",
            "rounds": args.rounds,
            "warmups": args.warmups,
            "workers": args.workers,
            "data_source": "process-local cryptographically random test data",
        },
        "chunk_sizes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
