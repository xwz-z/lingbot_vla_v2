# Unitree Mobile training reproduction

This repository extends upstream LingBot-VLA-v2 commit `2838c1862bbec1ea47942fb61512130f635eb595` with one production training path and one smoke path:

- **Unitree Mobile**: three cameras, relative dual-arm TCP `xyz+quaternion`, two grippers, and `vx/wz/height` base commands. This is the tested 40,000-step Expert-only path.
- **Nero Dual Arm smoke**: one camera, dual-arm TCP `xyz+quaternion`, and two grippers. This is intentionally limited to one training step.

Evaluation and robot-side inference are outside this minimal training reproduction.

## Implementation tree

```text
lingbot-vla-v2/
├── assets/norm_stats/
│   ├── unitree_mobile_xyzquat.json
│   ├── unitree_mobile_xyzquat_robust_meanstd.json
│   └── nero_dual_arm_xyzquat.json
├── configs/
│   ├── robot_configs/
│   │   ├── unitree_mobile_xyzquat.yaml
│   │   └── nero_dual_arm_xyzquat.yaml
│   └── vla/real_robot/
│       ├── unitree_mobile_xyzquat_expert_only.yaml
│       └── nero_dual_arm_xyzquat_expert_only_smoke.yaml
├── lingbotvla/
│   ├── data/vla_data/base_dataset.py
│   ├── models/vla/action_loss_utils.py
│   ├── models/vla/lingbot_vla/modeling_lingbot_vla_v2.py
│   └── utils/
│       ├── model_utils.py
│       └── normalize.py
├── scripts/
│   ├── compute_norm_stats.py
│   ├── convert_unitree_mobile_to_lingbot_v3.py
│   ├── convert_nero_fk_to_xyzquat.py
│   ├── build_robust_meanstd_stats.py
│   ├── validate_unitree_mobile_norm_stats.py
│   ├── train_unitree_mobile_xyzquat_expert_only.sh
│   └── train_nero_dual_arm_xyzquat_smoke.sh
├── tasks/vla/train_lingbotvla.py
├── tests/
│   ├── test_action_padding.py
│   ├── test_expert_only.py
│   └── test_normalize.py
└── tools/create_train_env.repro.sh
```

## What changed from upstream

1. `norm_stats_file` can override the robot mapping's default statistics file.
2. running statistics accumulate in float64.
3. normalization excludes padded episode-tail actions and supports `num_workers=0`.
4. action loss excludes padded timesteps as well as inactive action dimensions.
5. Expert-only startup auditing verifies that Qwen-VL is frozen and the action expert/projections are trainable.
6. Unitree source data are converted from `xyz+Rot6D` to continuous normalized `xyz+quaternion(xyzw)` features.
7. sparse base controls use a range-capped mean/std scale validated over every valid 50-step chunk.

## Environment

Create or resume the training environment. A compatible Python 3.12 FlashAttention wheel may be supplied explicitly:

```bash
bash tools/create_train_env.repro.sh \
  --env-name lingbotvla \
  --resume \
  --flash-attn-wheel /path/to/flash_attn-2.8.3+cu12torch2.8-cp312-linux_x86_64.whl
conda activate lingbotvla
```

Verify that the active interpreter belongs to the environment:

```bash
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda)'
```

## Repository-local external paths

Training YAML files intentionally use repository-relative paths. Link external weights and datasets without copying them into Git:

```bash
ln -s /path/to/lingbot-vla-v2-weights weights
mkdir -p data outputs
ln -s /path/to/unitree_mobile_lingbot_v3 data/unitree_mobile_lingbot_v3
ln -s /path/to/nero_dual_arm_xyzquat_lerobot_v3 data/nero_dual_arm_xyzquat_lerobot_v3
```

The weight directory must contain:

```text
weights/
├── lingbot-vla-v2-6b/
│   ├── depth/model.pt
│   └── dino_video/
│       ├── teacher_step_10000.pth
│       └── config.yaml
├── Qwen3-VL-4B-Instruct/
└── moge-2-vitb-normal/model.pt
```

## Unitree dataset conversion

The source exporter stores a 26-D state and 23-D action using dual-arm `xyz+Rot6D`. Convert it once:

```bash
python scripts/convert_unitree_mobile_to_lingbot_v3.py \
  /path/to/mobile_transfer_lerobot_clean \
  data/unitree_mobile_lingbot_v3 \
  --video-mode copy
```

The converted dataset exposes:

- `observation.state.end.position` and `action.end.position`: 14-D dual-arm `xyz+quaternion(xyzw)`.
- `observation.state.effector.position` and `action.effector.position`: two grippers.
- `observation.state.base.position` and `action.base.position`: `vx`, `wz`, and height.
- `cam0`, `cam1`, and `cam2`: head, left wrist, and right wrist cameras.

Raw TCP actions remain absolute in the dataset. The robot config converts them to world-frame relative translation and `q_state^-1 * q_action` during training.

## Recompute and validate normalization

The repository includes the exact tested statistics. To regenerate them from converted data:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh \
  scripts/compute_norm_stats.py \
  configs/vla/real_robot/unitree_mobile_xyzquat_expert_only.yaml \
  --data.train_path data/unitree_mobile_lingbot_v3 \
  --data.norm_path /tmp/unitree_mobile_xyzquat.json \
  --data.data_ratio_for_norm_compute 1 \
  --train.global_batch_size 1

python scripts/build_robust_meanstd_stats.py \
  /tmp/unitree_mobile_xyzquat.json \
  /tmp/unitree_mobile_xyzquat_robust_meanstd.json \
  --max-abs 3

python scripts/validate_unitree_mobile_norm_stats.py \
  data/unitree_mobile_lingbot_v3 \
  /tmp/unitree_mobile_xyzquat.json \
  /tmp/unitree_mobile_xyzquat_robust_meanstd.json \
  --chunk-size 50 \
  --output outputs/unitree_norm_validation.json
```

To train with regenerated statistics, pass:

```bash
--data.norm_stats_file /tmp/unitree_mobile_xyzquat_robust_meanstd.json
```

## Unitree four-GPU training

```bash
conda activate lingbotvla
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/train_unitree_mobile_xyzquat_expert_only.sh
```

Canonical settings:

- micro batch size: 1 per GPU
- global batch size: 4
- learning rate: `5e-5`
- maximum steps: 40,000
- checkpoint interval: 10,000 steps
- output: `outputs/unitree_mobile_xyzquat_expert_only`
- TensorBoard: `outputs/unitree_mobile_xyzquat_expert_only/runs`

Extra parser overrides may be appended to the command, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OUTPUT_DIR=outputs/unitree_ablation \
  bash scripts/train_unitree_mobile_xyzquat_expert_only.sh \
  --train.max_steps 100
```

## Nero one-step smoke

If the Nero source still contains FK-derived representations, convert it first:

```bash
python scripts/convert_nero_fk_to_xyzquat.py \
  /path/to/nero_pick_place_lerobot_v3 \
  data/nero_dual_arm_xyzquat_lerobot_v3
```

Run the single-GPU, single-step smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_nero_dual_arm_xyzquat_smoke.sh
```

This validates loading, preprocessing, model construction, forward/backward, and optimizer wiring. It is not a production Nero checkpoint recipe.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover padding exclusion, Expert-only trainable boundaries, and normalization precision.
