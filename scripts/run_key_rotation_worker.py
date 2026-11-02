#!/usr/bin/env python3
"""Run persisted video-key rotation items using an injected provider adapter.

The adapter factory is selected by trusted process configuration, never by an
HTTP request. Production deployments must return providers backed by their KMS
or HSM; this script does not load private key material from command-line args.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from facecheck.feature_store import FeatureStore
from facecheck.key_management import load_video_key_provider_bundle
from facecheck.rotation_control import RotationControlRepository, VideoKeyRotationWorker
from facecheck.security import secret_file


def _load_provider_bundle(spec: str):
    try:
        bundle = load_video_key_provider_bundle(spec)
    except Exception as exc:
        raise SystemExit(f"video key provider adapter is unavailable: {type(exc).__name__}") from None
    if bundle is None:
        raise SystemExit("video key provider adapter is not configured")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument(
        "--recover-interrupted", action="store_true",
        help="requeue running items after confirming the previous worker has stopped",
    )
    parser.add_argument(
        "--revoke-old-key", action="store_true",
        help="request KMS/HSM revocation only if inventory gates are clear",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.job_id):
        raise SystemExit("invalid rotation job ID")
    if not 1 <= args.max_items <= 1000:
        raise SystemExit("--max-items must be between 1 and 1000")
    factory_spec = os.getenv("FACECHECK_VIDEO_KEY_PROVIDER_FACTORY", "")
    if not factory_spec:
        raise SystemExit("configure a trusted KMS/HSM provider factory before running the worker")

    data_root = Path(os.getenv("FACECHECK_DATA_ROOT", "data"))
    database = Path(os.getenv("FACECHECK_DATABASE", str(data_root / "facecheck.sqlite3")))
    storage_root = Path(os.getenv("FACECHECK_STORAGE_ROOT", str(data_root / "storage")))
    old_encryption_key = secret_file("FACECHECK_SM4_KEY")
    old_mac_key = secret_file("FACECHECK_MAC_KEY")
    provider_bundle = _load_provider_bundle(factory_spec)
    target_provider = provider_bundle["target_provider"]
    resolver = provider_bundle["source_provider_resolver"]
    feature_store = None
    use_cache = os.getenv("FACECHECK_FEATURE_CACHE", "0") == "1"
    cache_root = storage_root.parent / "protected-features"
    if (use_cache or cache_root.exists()) and old_encryption_key and old_mac_key:
        feature_store = FeatureStore(
            cache_root, old_encryption_key, old_mac_key,
        )
    repository = RotationControlRepository(database)
    if args.recover_interrupted:
        recovered = repository.recover_interrupted(
            args.job_id, worker_stopped_confirmed=True,
        )
        print(json.dumps({"job_id": args.job_id, "recovered_items": recovered}, sort_keys=True))
    worker = VideoKeyRotationWorker(
        repository, storage_root, target_provider=target_provider,
        source_provider_resolver=resolver,
        legacy_encryption_key=old_encryption_key,
        legacy_mac_key=old_mac_key,
        feature_store=feature_store,
    )

    processed = 0
    while processed < args.max_items:
        result = worker.process_next(args.job_id)
        if result is None:
            break
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        processed += 1
    cache_results = worker.process_pending_cache_artifacts(args.job_id)
    if cache_results:
        print(json.dumps({"cache_cleanup": cache_results}, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"job_id": args.job_id, "processed_items": processed,
                      "revocation_readiness": repository.revocation_readiness(args.job_id)},
                     ensure_ascii=False, sort_keys=True))
    if args.revoke_old_key:
        revoke = provider_bundle.get("revoke_key")
        if not callable(revoke):
            raise SystemExit("provider adapter does not implement KMS/HSM key revocation")
        result = repository.revoke_old_key(args.job_id, revoke)
        print(json.dumps({"key_revocation": result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
