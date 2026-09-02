#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lingbotvla" ]]; then
  echo "Expected the lingbotvla Conda environment, got: ${CONDA_DEFAULT_ENV:-<none>}" >&2
  echo "Run: conda activate lingbotvla" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]] || [[ "$(python -c 'import sys; print(sys.prefix)')" != "$CONDA_PREFIX" ]]; then
  echo "The active python does not belong to CONDA_PREFIX=$CONDA_PREFIX" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#CUDA_DEVICE_LIST[@]}" -ne 4 ]]; then
  echo "This config requires four GPUs (global batch 4, micro batch 1); got $CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-62510}"
LOG_DIR="${LOG_DIR:-/data0/xieweize/checkpoints/lingbotvla_robust_meanstd/logs}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/nero_mobile_xyzquat_expert_only_robust_norm_$RUN_ID.log}"
mkdir -p "$(dirname "$TRAIN_LOG")"

echo "Training log: $TRAIN_LOG"
echo "TensorBoard logs: /data0/xieweize/checkpoints/lingbotvla_robust_meanstd/runs"

python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  tasks/vla/train_lingbotvla.py \
  configs/vla/real_robot/nero_mobile_xyzquat_expert_only_robust_norm.yaml \
  "$@" 2>&1 | tee "$TRAIN_LOG"
