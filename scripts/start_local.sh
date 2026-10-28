#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${FACECHECK_DATA_ROOT:=$PROJECT_ROOT/data}"
: "${FACECHECK_YUNET_MODEL:=$PROJECT_ROOT/models/face_detection_yunet_2023mar.onnx}"
: "${FACECHECK_SFACE_MODEL:=$PROJECT_ROOT/models/face_recognition_sface_2021dec.onnx}"
export FACECHECK_DATA_ROOT FACECHECK_YUNET_MODEL FACECHECK_SFACE_MODEL

if [[ -z "${FACECHECK_SM4_KEY:-}" ]]; then
  export FACECHECK_SM4_KEY="0123456789abcdef"
  echo "提示：未设置 FACECHECK_SM4_KEY，使用本机课程演示密钥；不要用于生产环境。" >&2
fi
if [[ -z "${FACECHECK_MAC_KEY:-}" ]]; then
  export FACECHECK_MAC_KEY="local-demo-mac-key"
  echo "提示：未设置 FACECHECK_MAC_KEY，使用本机课程演示密钥；不要用于生产环境。" >&2
fi

if [[ ! -f "$FACECHECK_YUNET_MODEL" ]]; then
  echo "错误：YuNet 模型不存在：$FACECHECK_YUNET_MODEL" >&2
  echo "请将模型放入 models/，或设置 FACECHECK_YUNET_MODEL。" >&2
  exit 1
fi
if [[ ! -f "$FACECHECK_SFACE_MODEL" ]]; then
  echo "错误：SFace 模型不存在：$FACECHECK_SFACE_MODEL" >&2
  echo "请将模型放入 models/，或设置 FACECHECK_SFACE_MODEL。" >&2
  exit 1
fi

mkdir -p "$FACECHECK_DATA_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "错误：项目虚拟环境不存在，请先执行：uv sync --python python3" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "错误：找不到 uv，请先安装 uv 或将其加入 PATH。" >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  if curl --silent --show-error --max-time 3 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "Facecheck 已经运行：http://${HOST}:${PORT}/"
    echo "如果要重启，请先停止原服务，或使用其他端口：PORT=8001 ./scripts/start_local.sh"
    exit 0
  fi
  echo "错误：${HOST}:${PORT} 已被其他进程占用，无法启动 Facecheck。" >&2
  echo "请停止占用该端口的进程，或使用其他端口：PORT=8001 ./scripts/start_local.sh" >&2
  exit 1
fi

echo "Facecheck 正在启动：http://${HOST}:${PORT}/"
echo "按 Ctrl+C 停止服务。"
exec uv run uvicorn facecheck.main:app --host "$HOST" --port "$PORT"
