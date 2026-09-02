#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REAL_WORLD_INFERENCE_CONFIG:-${SCRIPT_DIR}/config/mobile_transfer_lingbot_new40_tcp23.json}"

if [[ "$#" -gt 1 ]] || { [[ "$#" -eq 1 ]] && [[ "$1" != "--check" ]]; }; then
  echo "Usage: bash real_world_inference/start_server.sh [--check]" >&2
  exit 2
fi

PYTHON_BIN="${LINGBOT_INFERENCE_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]] && [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] No active Conda environment. Run: conda activate lingbotvla" >&2
  echo "[ERROR] Or set LINGBOT_INFERENCE_PYTHON explicitly." >&2
  exit 2
fi
if [[ -z "${PYTHON_BIN}" ]] || [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Active-environment Python is not executable: ${PYTHON_BIN:-<empty>}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[LingBot inference] config=${CONFIG_FILE}"
echo "[LingBot inference] python=${PYTHON_BIN}"
echo "[LingBot inference] conda_prefix=${CONDA_PREFIX:-<not-active>}"
echo "[LingBot inference] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherited>}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/server.py" --config "${CONFIG_FILE}" --check >/dev/null

if [[ "${1:-}" == "--check" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/server.py" \
    "${SCRIPT_DIR}/inference_service.py" \
    "${SCRIPT_DIR}/policy_adapter.py" \
    "${SCRIPT_DIR}/pose_transforms.py"
  echo "[LingBot inference] check passed"
  exit 0
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/server.py" --config "${CONFIG_FILE}"
