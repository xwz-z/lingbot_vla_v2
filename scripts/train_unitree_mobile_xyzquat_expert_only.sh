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
  data/unitree_mobile_lingbot_v3
  assets/norm_stats/unitree_mobile_xyzquat_robust_meanstd.json
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training input: $REPO_ROOT/$path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#CUDA_DEVICE_LIST[@]}" -ne 4 ]]; then
  echo "Unitree config requires four GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/unitree_mobile_xyzquat_expert_only}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
TRAIN_LOG="${TRAIN_LOG:-$OUTPUT_DIR/logs/unitree_mobile_xyzquat_expert_only_$RUN_ID.log}"
mkdir -p "$(dirname "$TRAIN_LOG")"

echo "Training log: $TRAIN_LOG"
echo "TensorBoard logs: $OUTPUT_DIR/runs"

python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  tasks/vla/train_lingbotvla.py \
  configs/vla/real_robot/unitree_mobile_xyzquat_expert_only.yaml \
  --train.output_dir "$OUTPUT_DIR" \
  "$@" 2>&1 | tee "$TRAIN_LOG"
