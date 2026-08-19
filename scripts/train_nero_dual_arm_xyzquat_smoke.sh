#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-lingbotvla}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]]; then
  echo "Expected Conda environment '$EXPECTED_CONDA_ENV', got '${CONDA_DEFAULT_ENV:-<none>}'" >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]] || [[ "$(python -c 'import sys; print(sys.prefix)')" != "$CONDA_PREFIX" ]]; then
  echo "The active python does not belong to CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi

required_paths=(
  weights/lingbot-vla-v2-6b
  weights/Qwen3-VL-4B-Instruct
  weights/moge-2-vitb-normal/model.pt
  weights/lingbot-vla-v2-6b/depth/model.pt
  weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
  weights/lingbot-vla-v2-6b/dino_video/config.yaml
  data/nero_dual_arm_xyzquat_lerobot_v3
  assets/norm_stats/nero_dual_arm_xyzquat.json
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required smoke-test input: $REPO_ROOT/$path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#CUDA_DEVICE_LIST[@]}" -ne 1 ]]; then
  echo "Nero smoke config requires one GPU; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1

OUTPUT_DIR="${OUTPUT_DIR:-outputs/nero_dual_arm_xyzquat_smoke}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
TRAIN_LOG="${TRAIN_LOG:-$OUTPUT_DIR/logs/nero_dual_arm_xyzquat_smoke_$RUN_ID.log}"
mkdir -p "$(dirname "$TRAIN_LOG")"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-62520}" \
  tasks/vla/train_lingbotvla.py \
  configs/vla/real_robot/nero_dual_arm_xyzquat_expert_only_smoke.yaml \
  --train.output_dir "$OUTPUT_DIR" \
  "$@" 2>&1 | tee "$TRAIN_LOG"
