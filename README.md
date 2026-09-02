<h1 align="center">LingBot-VLA 2.0: From Foundation to Application</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2607.06403"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-vla-v2"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-vla-v2"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VLA-V2"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

## Overview

**LingBot-VLA 2.0** is a practical Vision-Language-Action foundation model designed to move from large-scale pre-training toward reliable real-world robot applications.

Compared with LingBot-VLA 1.0, LingBot-VLA 2.0 improves three core capabilities:

- **Generalization across tasks and embodiments**: a redesigned data pipeline curates around **60,000 hours** of pre-training data, including **50,000 hours** of robot trajectories across **20 robot configurations** and **10,000 hours** of egocentric human videos.
- **Expanded action space**: the unified representation supports arms, end-effectors, grippers, dexterous hands, waist, head, and mobile-base signals instead of only standard dual-arm manipulation.
- **Predictive dynamics modeling**: future prediction is used as a proxy task, with DINO-Video providing semantic temporal priors and LingBot-Depth providing geometric cues.

<p align="center">
  <img src="assets/lingbot_vla2_framework.png" width="86%">
</p>

## Nero / Unitree Real-Robot Demo

This fork contains the implementation used to run a LingBot-VLA 2.0 real-robot
demo on a Nero/Unitree mobile dual-arm robot. The tested deployment profile is
one NVIDIA GeForce RTX 4090, BF16 inference, a 50-action output chunk, and 10
diffusion/denoising steps.

### Runtime boundary

The demo uses the self-contained service in this repository:

```text
Unitree client (mobile_tcp23)
  -> real_world_inference HTTP service
  -> mobile_tcp23 / LingBot representation adapter
  -> deploy.LingbotVLAv2Server
  -> LingBot-VLA 2.0 checkpoint
```

Despite its directory name, `real_world_inference/` is a standalone inference
implementation in this LingBot-VLA repository. It directly wraps the official
[`deploy/lingbot_vla_v2_policy.py`](deploy/lingbot_vla_v2_policy.py) policy and
does not depend on the separately maintained generic `real_world_inference`
framework. This keeps the Demo repository independently runnable while retaining
the same robot-side `mobile_tcp23` contract.

### Robot adaptation

The robot client sends three RGB camera streams (`head_fpv`, `left_hand`, and
`right_hand`) together with `mobile_state[26]`. The state contains two
`xyz+Rot6D+gripper` arm states, map-frame `x/y/yaw`, and base
`vx/wz/height`. The adapter:

- converts both TCP rotations from Rot6D to quaternion `xyzw` and preserves
  quaternion sign continuity between requests;
- maps the three cameras to LingBot `cam0/cam1/cam2` inputs;
- intentionally leaves map-frame `x/y/yaw` outside the model input and maps
  `vx/wz/height` to the mobile-base feature;
- converts LingBot's dual-arm quaternion actions back to Rot6D; and
- returns 20 arm/gripper dimensions in `actions` plus
  `[vx, vy=0, wz, height]` in `base_action`.

The model-side robot mapping is defined in
[`configs/robot_configs/nero_mobile_xyzquat.yaml`](configs/robot_configs/nero_mobile_xyzquat.yaml).
It uses relative TCP targets: world-frame translation deltas and
`q_state^-1 * q_action` rotations. Gripper and base targets remain absolute.

### Demo implementation files

| Responsibility | Files |
| --- | --- |
| Unitree-compatible HTTP endpoints and wire contract | [`real_world_inference/inference_service.py`](real_world_inference/inference_service.py), [`real_world_inference/server.py`](real_world_inference/server.py) |
| State, image, pose, and action adaptation | [`real_world_inference/policy_adapter.py`](real_world_inference/policy_adapter.py), [`real_world_inference/pose_transforms.py`](real_world_inference/pose_transforms.py) |
| Demo runtime configuration | [`real_world_inference/config/mobile_transfer_lingbot_new40_tcp23.json`](real_world_inference/config/mobile_transfer_lingbot_new40_tcp23.json) |
| LingBot policy loading and native inference | [`deploy/lingbot_vla_v2_policy.py`](deploy/lingbot_vla_v2_policy.py) |
| Robot feature mapping and normalization | [`configs/robot_configs/nero_mobile_xyzquat.yaml`](configs/robot_configs/nero_mobile_xyzquat.yaml), [`assets/norm_stats/nero_mobile_xyzquat_robust_meanstd.json`](assets/norm_stats/nero_mobile_xyzquat_robust_meanstd.json) |
| Expert-only post-training configuration | [`configs/vla/real_robot/nero_mobile_xyzquat_expert_only_robust_norm.yaml`](configs/vla/real_robot/nero_mobile_xyzquat_expert_only_robust_norm.yaml) |
| Dataset conversion and statistics preparation | [`scripts/convert_mobile_transfer_to_lingbot_v3.py`](scripts/convert_mobile_transfer_to_lingbot_v3.py), [`scripts/build_robust_meanstd_stats.py`](scripts/build_robust_meanstd_stats.py), [`scripts/validate_nero_mobile_norm_stats.py`](scripts/validate_nero_mobile_norm_stats.py) |
| Server launcher and protocol tests | [`real_world_inference/start_server.sh`](real_world_inference/start_server.sh), [`tests/test_unitree_lingbot_inference.py`](tests/test_unitree_lingbot_inference.py) |

The post-training path freezes the Qwen/VLM backbone and trains the action
expert plus the state/action projections. The local changes also allow an
explicit normalization-statistics file and exclude padded action timesteps from
the training loss, which are required for consistent 50-step mobile-action
chunks.

### RTX 4090 launch

First edit the following fields in
[`real_world_inference/config/mobile_transfer_lingbot_new40_tcp23.json`](real_world_inference/config/mobile_transfer_lingbot_new40_tcp23.json):

- `policy.checkpoint_dir`: the post-training `hf_ckpt` directory;
- `policy.norm_stats_path`: the matching normalization JSON;
- `policy.task_prompt`: the instruction used when the client does not send one.

Then launch the standalone server on GPU 0:

```bash
cd /path/to/lingbot-vla-v2
conda activate lingbotvla

export CUDA_VISIBLE_DEVICES=0
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export LINGBOT_INFERENCE_PYTHON="$CONDA_PREFIX/bin/python"
export REAL_WORLD_INFERENCE_CONFIG=real_world_inference/config/mobile_transfer_lingbot_new40_tcp23.json

# Validate paths, protocol configuration, and Python modules without loading the model.
bash real_world_inference/start_server.sh --check

# Start HTTP inference on the host and port declared in the JSON (default: 0.0.0.0:8027).
bash real_world_inference/start_server.sh
```

The client can use `GET /health`, `POST /handshake`, and `POST /infer`. The
request may be JSON with base64 images or multipart form data with JPEG image
parts.

### Diffusion steps and RTC status

The only inference-side optimization used for this Demo is controlling the
number of diffusion/denoising steps. In real-robot testing, 50 denoising steps
gave mediocre results, while 10 steps remained usable and provided more latency
headroom for future real-time chunking (RTC). The current model default is
therefore `num_steps = 10` in
[`configuration_lingbot_vla.py`](lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py).

`num_steps = 10` is the number of denoising iterations; it is independent of
`target_chunk_size = 50`, which is the number of actions returned per inference
request. RTC correction/reuse of a previously executing action chunk is not
implemented in this Demo yet; the current service performs direct chunk
inference and keeps the protocol boundary ready for that follow-up work.

## News

- **[2026-07-08]** LingBot-VLA 2.0 technical report and pre-trained weights are prepared.

## Installation

Requirements:

- Miniconda or Anaconda
- Python 3.12
- PyTorch 2.8.0

Before running the setup script, make sure Conda is initialized in your shell and `conda activate` works.

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2

bash tools/create_train_env.sh
```

By default, the script installs `flash-attn==2.8.3` from pip. If you already have a matching local wheel, pass it explicitly:

```bash
bash tools/create_train_env.sh \
  --flash-attn-wheel /path/to/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

You can also choose the environment name or force a rebuild:

```bash
bash tools/create_train_env.sh \
  --env-name lingbotvla \
  --recreate
```

## Model Download

We release **LingBot-VLA 2.0** pre-trained weights as a native-depth model.

| Model Name | Hugging Face | ModelScope | Description |
| :--- | :---: | :---: | :---: |
| LingBot-VLA 2.0 | [lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | [lingbot-vla-v2-6b](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b) | Native Depth |

To train LingBot-VLA 2.0 with this codebase, weights from [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal), [LingBot-Depth](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/depth), and [DINO-VIDEO](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/dino_video) teacher checkpoint/config are also required. See [Training_Config.md](configs/vla/Training_Config.md).

```bash
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir lingbot-vla
```

## Pre-Training Data

LingBot-VLA 2.0 uses a large, heterogeneous pre-training corpus that covers single-arm, dual-arm, half-humanoid, humanoid, and egocentric sources.

<p align="center">
  <img src="assets/lingbot_vla2_data_demo.png" width="100%">
</p>

The raw pool is filtered into high-quality robotic and egocentric streams. The robotic side removes video-state misalignment, blurry or occluded videos, multi-view misalignment, abnormal velocity/acceleration/jerk, and static-signal episodes. The egocentric side keeps manipulation-centric videos, reconstructs and standardizes hand trajectories, and filters unstable camera or hand-motion estimates.

<p align="center">
  <img src="assets/lingbot_vla2_data_process.png" width="70%">
</p>

## Model Design

### Unified Action Representation

LingBot-VLA 2.0 maps heterogeneous embodiments into a 55-dimensional canonical state/action vector:

- 14 dimensions for arm joint position
- 14 dimensions for end-effector pose
- 2 dimensions for gripper position
- 12 dimensions for hand joint position
- 4 dimensions for waist position
- 2 dimensions for head position
- 3 dimensions for mobility signal
- 4 reserved dimensions

<p align="center">
  <img src="assets/lingbot_vla2_data_dimension.png" width="100%">
</p>

### MoE Action Expert

To improve cross-embodiment scaling, LingBot-VLA 2.0 uses sparse MoE layers inside the action expert. Fine-grained expert segmentation and shared expert isolation allow universal priors and specialized embodiment/task patterns to coexist under the same active compute budget.

<p align="center">
  <img src="assets/lingbot_vla2_loss_mse_comparison.png" width="90%">
</p>

### Dual-Query Distillation

LingBot-VLA 2.0 appends current and future perceptual queries to the visual/text tokens. These queries are distilled from LingBot-Depth and DINO-Video, encouraging causal inference to capture both current scene geometry and future scene evolution.

<p align="center">
  <img src="assets/lingbot_vla2_vis_distillation.png" width="100%">
</p>

## Post-Training Example

### Data Preparation

Post-training requires three preparation steps. For a complete guide on customizing your own dataset, see the [Custom Data Guide](lingbotvla/data/vla_data/README.md).

| Step | Description | Output |
|------|-------------|--------|
| 1. Prepare LeRobot Dataset | Prepare a LeRobot v2.1 or v3.0 dataset directory | LeRobot dataset directory |
| 2. Prepare Robot Config | Define feature mapping from raw states/actions/images to the unified feature space | `configs/robot_configs/<data_name>.yaml` |
| 3. Compute Norm Statistics | Calculate normalization statistics over your dataset | `assets/norm_stats/<name>.json` |

Below we use **RoboTwin 2.0** 50 tasks, trained with clean and randomized data together, as an example.

- **Step 1 - RoboTwin Data**: Follow [RoboTwin2.0 Preparation](experiment/robotwin/README.md) to download and prepare the dataset.
- **Step 2 - Robot Config**: See [configs/robot_configs/robotwin.yaml](configs/robot_configs/robotwin.yaml) for the RoboTwin feature mapping.
- **Step 3 - Normalization**: Pre-computed stats are provided at `assets/norm_stats/robotwin.json`. To recompute for a custom task subset, see the [Custom Data Guide](lingbotvla/data/vla_data/README.md).

### Training

We provide a post-training example of **LingBot-VLA 2.0** on RoboTwin 2.0 50 tasks with clean and randomized data:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.train_path assets/training_data/robotwin.txt \
  --data.data_name multi \
  --train.output_dir output/
```

The post-training config uses sequence-wise auxiliary loss (`sequence_wise_mode: "per_sequence"`, `sequence_wise_loss_coeff: 1e-3`) together with z-loss (`router_z_loss_coeff: 1e-4`) for MoE routing. These terms can be adjusted or disabled depending on the downstream task. To use a loss-free routing setup, comment out the sequence-wise auxiliary loss and z-loss options, and set `bias_update_speed: 0.00025`.
The post-training config also enables the Muon optimizer. Muon can produce a better-converged loss, but it increases training time. To use the default AdamW optimizer instead, comment out `optimizer: muon`.

For real-world scenarios, see the native-depth training configuration [real_robot.yaml](configs/vla/real_robot/real_robot.yaml). For detailed explanations of batch size, gradient accumulation, checkpointing, depth/video distillation, MoE, and optimizer settings, see [Training_Config.md](configs/vla/Training_Config.md).

## Evaluation and Deployment

### Open-Loop Evaluation

```bash
export QWEN3_PATH=Qwen/Qwen3-VL-4B-Instruct
python scripts/open_loop_eval.py \
  --model_path path_to_posttraining_ckpt \
  --robo_name robotwin \
  --data_path path_to_validation_data \
  --use_length 50
```

`--robo_name` is required for open-loop evaluation. It selects the robot config from `configs/robot_configs/{robo_name}.yaml`, for example `--robo_name robotwin` uses `configs/robot_configs/robotwin.yaml`.



### RoboTwin Deployment

After making the RoboTwin simulation dependencies compatible with the model inference dependencies in a single environment, we provide a one-command evaluation script for all 50 RoboTwin 2.0 tasks:
```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct/ \
EVAL_WORKDIR=/path/to/Robotwin_code/ \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/your/post_training_checkpoint \
  --output_base /path/to/your/eval_output \
  --num_per_gpu 2
```
`num_per_gpu` specifies how many tasks can be evaluated concurrently on each GPU. Tune it according to your available GPU memory and the communication load your machine can handle.

### Real-Robot Deployment

```bash
export QWEN3VL_PATH=path_to_Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path path_to_posttraining_ckpt \
  --use_compile \
  --use_length 25 \
  --port port
```

Using `deploy.lingbot_vla_v2_policy`, one inference call on an NVIDIA GeForce RTX 4090D takes about **130 ms** with **10 denoising steps**.

## Performance

LingBot-VLA 2.0 is evaluated in a generalist setting on GM-100 bimanual manipulation and long-horizon mobile manipulation. Metrics are reported as progress score / success rate where applicable.

### GM-100 Bimanual Manipulation

| Platform | GR00T N1.7 | π<sub>0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
| :--- | ---: | ---: | ---: | ---: |
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / **15.6** | **34.6 / 15.6** |

<p align="center">
  <img src="assets/lingbot_vla2_gm100_ablation_barplot.png" width="75%">
</p>

### Long-Horizon Mobile Manipulation

| Embodiment | Task | Setting | LingBot-VLA 2.0 | π<sub>0.5 |
| :--- | :--- | :--- | ---: | ---: |
| Astribot S1 | Refrigerator sorting | In-domain | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | Refrigerator sorting | Out-of-domain | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | Stove cleaning | In-domain | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | Stove cleaning | Out-of-domain | **67.5 / 40.0** | 62.5 / 33.3 |

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{lingbotvla2,
      title={From Foundation to Application: Improving VLA Models in Practice}, 
      author={Wei Wu and Fangjing Wang and Fan Lu and He Sun and Shi Liu and Yunnan Wang and Yibin Yan and Yong Wang and Shuailei Ma and Xinyang Wang and Yibin Liu and Shuai Yang and Tianxiang Zhou and Kejia Zhang and Lei Zhou and Cheng Su and Nan Xue and Bin Tan and Han Zhang and Youchao Zhang and Fei Liao and Xing Zhu and Yujun Shen and Kecheng Zheng},
      journal={arXiv preprint arXiv:2607.06403},
      year={2026}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Acknowledgement

We sincerely thank the developers of [VeOmni](https://arxiv.org/abs/2508.02317) and [LeRobot](https://github.com/huggingface/lerobot). This project benefits from their contributions to the open-source community.

## Local Unitree training reproduction

This fork adds a minimal, tested Unitree Mobile Expert-only training path and a one-step Nero Dual Arm smoke path. Start with [docs/TRAINING_REPRODUCTION.md](docs/TRAINING_REPRODUCTION.md) for the implementation tree, external path layout, dataset conversion, normalization validation, training commands, and tests.
