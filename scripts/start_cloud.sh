#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 3 ]]; then
  echo 'Usage: scripts/start_cloud.sh PRIVATE_ENV_FILE TLS_CERT_FILE TLS_KEY_FILE' >&2
  exit 2
fi
for task_file in "$@"; do
  [[ -r "$task_file" ]] || { echo 'A required private configuration/certificate file is unreadable' >&2; exit 2; }
done
# uvicorn --env-file does not override existing variables: refuse ambiguous legacy keys.
if [[ -n "${FACECHECK_SM4_KEY:-}" || -n "${FACECHECK_MAC_KEY:-}" ]]; then
  echo 'Unset legacy FACECHECK_SM4_KEY / FACECHECK_MAC_KEY before using private key files' >&2
  exit 2
fi
TASK_ENV_FILE="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
TASK_CERT_FILE="$(cd -- "$(dirname -- "$2")" && pwd)/$(basename -- "$2")"
TASK_TLS_KEY_FILE="$(cd -- "$(dirname -- "$3")" && pwd)/$(basename -- "$3")"
cd "$TASK_ROOT"
# Force the secure profile even if a caller mistakenly supplies a local environment file.
export FACECHECK_DEPLOYMENT=cloud
exec .venv/bin/python -m uvicorn facecheck.main:app --env-file "$TASK_ENV_FILE" \
  --host "${FACECHECK_BIND_HOST:-0.0.0.0}" --port "${PORT:-8443}" \
  --ssl-certfile "$TASK_CERT_FILE" --ssl-keyfile "$TASK_TLS_KEY_FILE" \
  --no-proxy-headers --workers 1 --limit-concurrency 64 --timeout-keep-alive 5
