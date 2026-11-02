#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
KEY_FILE="$ROOT/.env.ytf-experiment"
OUTPUT=${FACECHECK_YTF_OUTPUT:-"$ROOT/data/ytf/experiments/smoke_20_run_01"}
SCORE_THRESHOLD=${FACECHECK_YTF_SCORE_THRESHOLD:-0.9}
SCORE_AGGREGATION=${FACECHECK_YTF_SCORE_AGGREGATION:-max}
SCORE_AGGREGATION_TOP_N=${FACECHECK_YTF_SCORE_AGGREGATION_TOP_N:-3}
QUERY_USE_ANNOTATION_BOX=${FACECHECK_YTF_QUERY_USE_ANNOTATION_BOX:-0}
case "$OUTPUT" in
    /*) ;;
    *) OUTPUT="$ROOT/$OUTPUT" ;;
esac

if [ ! -x "$PYTHON" ]; then
    echo "Missing project virtualenv Python: $PYTHON" >&2
    exit 1
fi

if [ -e "$OUTPUT" ] || [ -L "$OUTPUT" ]; then
    echo "Refusing to overwrite existing experiment output: $OUTPUT" >&2
    exit 1
fi

case "$SCORE_THRESHOLD" in
    ''|*[!0-9.]*|.*|*.*.*) echo "Invalid FACECHECK_YTF_SCORE_THRESHOLD" >&2; exit 1 ;;
esac
case "$SCORE_AGGREGATION" in
    max|top_k_mean) ;;
    *) echo "Invalid FACECHECK_YTF_SCORE_AGGREGATION" >&2; exit 1 ;;
esac
case "$SCORE_AGGREGATION_TOP_N" in
    ''|*[!0-9]*) echo "Invalid FACECHECK_YTF_SCORE_AGGREGATION_TOP_N" >&2; exit 1 ;;
esac
case "$QUERY_USE_ANNOTATION_BOX" in
    0|1) ;;
    *) echo "Invalid FACECHECK_YTF_QUERY_USE_ANNOTATION_BOX (expected 0 or 1)" >&2; exit 1 ;;
esac

if ! git check-ignore -q .env.ytf-experiment; then
    echo ".env.ytf-experiment is not ignored by Git; refusing to create local keys" >&2
    exit 1
fi

# Validate or create the local key file without printing key material.
"$PYTHON" - "$KEY_FILE" <<'PY'
import os
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "FACECHECK_EXPERIMENT_ENCRYPTION_KEY": 16,
    "FACECHECK_EXPERIMENT_MAC_KEY": None,
}

def parse_key_file(text):
    values = {}
    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid key file line {number}")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name not in expected or name in values:
            raise ValueError(f"unexpected or duplicate key name on line {number}")
        if not value or len(value) % 2:
            raise ValueError(f"key value on line {number} is not non-empty even-length hex")
        try:
            decoded = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"key value on line {number} is not hexadecimal") from exc
        required_length = expected[name]
        if required_length is not None and len(decoded) != required_length:
            raise ValueError(f"{name} must decode to exactly {required_length} bytes")
        values[name] = value
    if set(values) != set(expected):
        raise ValueError("key file must contain exactly the two required key variables")
    return values

if path.is_symlink():
    raise SystemExit("Refusing to use a symbolic link as the local experiment key file")
if path.exists():
    if not path.is_file():
        raise SystemExit("Existing local experiment key path is not a regular file")
    try:
        parse_key_file(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"Existing local experiment key file is invalid; it was not changed: {exc}") from exc
else:
    values = {
        "FACECHECK_EXPERIMENT_ENCRYPTION_KEY": secrets.token_hex(16),
        "FACECHECK_EXPERIMENT_MAC_KEY": secrets.token_hex(32),
    }
    content = "".join(f"{name}={value}\n" for name, value in values.items())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        # A concurrent creator won the race. Validate and reuse its file.
        try:
            parse_key_file(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise SystemExit(f"Concurrent local experiment key file is invalid; it was not changed: {exc}") from exc
    else:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        parse_key_file(path.read_text(encoding="ascii"))

os.chmod(path, 0o600)
print("Local YTF experiment key file ready; key values were not displayed.")
PY

# Load keys into this Python process only; do not source shell code from the key file.
exec "$PYTHON" - "$ROOT" "$KEY_FILE" "$OUTPUT" "$SCORE_THRESHOLD" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
key_file = Path(sys.argv[2])
output = Path(sys.argv[3])
if not output.is_absolute():
    output = root / output
values = {}
for raw_line in key_file.read_text(encoding="ascii").splitlines():
    line = raw_line.strip()
    if line and not line.startswith("#"):
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
os.environ["FACECHECK_EXPERIMENT_ENCRYPTION_KEY"] = values["FACECHECK_EXPERIMENT_ENCRYPTION_KEY"]
os.environ["FACECHECK_EXPERIMENT_MAC_KEY"] = values["FACECHECK_EXPERIMENT_MAC_KEY"]
sys.path.insert(0, str(root / "src"))
sys.argv = [
    "facecheck.ytf_runner",
    "--manifest", str(root / "data/ytf/smoke_20/experiment_smoke_20.json"),
    "--identity-split", str(root / "data/ytf/smoke_20/identity_split_smoke_20.json"),
    "--output", str(output),
    "--yunet-model", str(root / "models/face_detection_yunet_2023mar.onnx"),
    "--sface-model", str(root / "models/face_recognition_sface_2021dec.onnx"),
    "--score-threshold", str(sys.argv[4]),
    "--score-aggregation", os.environ.get("FACECHECK_YTF_SCORE_AGGREGATION", "max"),
    "--score-aggregation-top-n", os.environ.get("FACECHECK_YTF_SCORE_AGGREGATION_TOP_N", "3"),
]
if os.environ.get("FACECHECK_YTF_QUERY_USE_ANNOTATION_BOX", "0") == "1":
    sys.argv.append("--query-use-annotation-box")
from facecheck.ytf_runner import main
main()
PY
