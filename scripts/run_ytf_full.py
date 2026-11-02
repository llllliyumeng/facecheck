"""Run the approved full YTF protocol with private execution provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import resource
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import cryptography
import numpy as np

from facecheck.face_detection import FaceDetectionConfig
from facecheck.recognition import DEFAULT_RECOGNITION_MODEL, recognition_config
from facecheck.storage import restore_encrypted_video, store_encrypted_video
from facecheck.ytf_qg_protocol import split_identities
from facecheck.ytf_runner import (
    OpenCVFeatureBackend, RunnerConfig, _load_inputs, _sha256, run_ytf_runner,
)


def read_keys(path: Path) -> tuple[bytes, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("existing experiment key file must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("experiment key file must not be accessible by group or others")
    lengths = {"FACECHECK_EXPERIMENT_ENCRYPTION_KEY": 16,
               "FACECHECK_EXPERIMENT_MAC_KEY": 32}
    keys = {}
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in lengths or name in keys:
            raise ValueError("unexpected experiment key file structure")
        try:
            decoded = bytes.fromhex(value.strip())
        except ValueError:
            raise ValueError("experiment key is not hexadecimal") from None
        if len(decoded) != lengths[name]:
            raise ValueError("experiment key has an invalid length")
        keys[name] = decoded
    if keys.keys() != lengths.keys():
        raise ValueError("experiment key file is missing required entries")
    return keys["FACECHECK_EXPERIMENT_ENCRYPTION_KEY"], keys["FACECHECK_EXPERIMENT_MAC_KEY"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recognition-model", choices=("arcface", "sface"), default=DEFAULT_RECOGNITION_MODEL)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    execution = output.with_name(output.name + ".execution")
    if any(path.exists() or path.is_symlink() for path in (output, execution)):
        raise ValueError("result or execution directory already exists; refusing overwrite")
    manifest = root / "data/ytf/experiment_full.json"
    split_path = root / "data/ytf/query_gallery_identity_split.json"
    records, assignment, _, split_json = _load_inputs(manifest, split_path)
    if len(records) != 2008 or Counter(assignment.values()) != {
            "threshold_selection": 502, "test": 502}:
        raise ValueError("full experiment requires 1004 identities and 2008 media records")
    if assignment != split_identities(assignment, seed=split_json["seed"]):
        raise ValueError("identity assignments differ from their deterministic seed")
    encryption_key, mac_key = read_keys(root / ".env.ytf-experiment")
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise ValueError("less than 10 GiB free disk space")
    code_paths = sorted((root / "src/facecheck").glob("*.py")) + [Path(__file__).resolve()]
    code_hashes = {str(p.relative_to(root)): _sha256(p) for p in code_paths}
    provenance = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "opencv": cv2.__version__,
        "cryptography": cryptography.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": code_hashes, "manifest_sha256": _sha256(manifest),
        "identity_split_sha256": _sha256(split_path),
        "output": str(output), "key_source": ".env.ytf-experiment (read only)",
        "protocol": "two identity-disjoint partitions; 502 galleries per query",
    }
    execution.mkdir(mode=0o700)
    (execution / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(execution / "runner.log")],
    )
    logging.captureWarnings(True)
    counts = Counter()
    started = time.perf_counter()

    class ProgressBackend(OpenCVFeatureBackend):
        def extract(self, record):
            result = super().extract(record)
            mode = record["_experiment_mode"]
            counts[mode + "_media"] += 1
            counts[mode + "_frames"] += result.sampled_frame_count
            counts[mode + "_features"] += len(result.features)
            counts[mode + "_feature_failures"] += result.feature_extraction_failure_count
            if counts[mode + "_media"] % 25 == 0 or counts[mode + "_media"] == 2008:
                logging.info("mode=%s media=%s/2008 frames=%s features=%s feature_failures=%s elapsed=%.1fs",
                             mode, counts[mode + "_media"], counts[mode + "_frames"],
                             counts[mode + "_features"], counts[mode + "_feature_failures"],
                             time.perf_counter() - started)
            return result

    def encrypt(*positional, **kwargs):
        stored = store_encrypted_video(*positional, **kwargs)
        counts["encrypted_videos"] += 1
        counts["encrypted_chunks"] += stored.manifest["chunk_count"]
        return stored

    def restore(stored_directory, destination, *keys, **kwargs):
        result = restore_encrypted_video(stored_directory, destination, *keys, **kwargs)
        metadata = json.loads((stored_directory / "manifest.json").read_text())
        counts["authenticated_restored_videos"] += 1
        counts["authenticated_restored_chunks"] += metadata["chunk_count"]
        return result

    logging.info("Starting approved full run: output=%s; 1004 identities; 502+502 partitions", output)
    status = "failed"
    try:
        config = RunnerConfig(
            manifest, split_path, output,
            FaceDetectionConfig(root / "models/face_detection_yunet_2023mar.onnx"),
            recognition_config(args.recognition_model),
        )
        metrics = run_ytf_runner(config, encryption_key=encryption_key, mac_key=mac_key,
                                 backend_factory=ProgressBackend, encrypt=encrypt, restore=restore)
        status = "completed"
        logging.info("Published full result bundle: %s; consistency=%s", output,
                     metrics["mode_consistency"]["consistent"])
    except Exception:
        logging.exception("Formal experiment failed; result bundle was not published")
        raise
    finally:
        summary = {
            "status": status, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started, "counts": dict(counts),
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_unchanged": all(_sha256(root / name) == digest for name, digest in code_hashes.items()),
        }
        (execution / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logging.info("Execution finished: status=%s wall=%.1fs counts=%s",
                     status, summary["wall_seconds"], dict(counts))


if __name__ == "__main__":
    main()
