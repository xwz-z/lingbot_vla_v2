from __future__ import annotations

"""Unitree mobile_tcp23 to LingBot VLA V2 policy adapter."""

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

try:
    from .pose_transforms import rot6d_to_xyzw, xyzw_to_rot6d
except ImportError:  # Direct script execution.
    from pose_transforms import rot6d_to_xyzw, xyzw_to_rot6d


MOBILE_STATE_DIM = 26
WIRE_ARM_ACTION_DIM = 20
VIRTUAL_MODEL_ACTION_DIM = 23
LINGBOT_LOGICAL_ACTION_DIM = 19


@dataclass(frozen=True)
class PolicyInputs:
    mobile_state26: np.ndarray
    images: Mapping[str, np.ndarray]
    prompt: str


def validate_mobile_state26(values: object) -> np.ndarray:
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (MOBILE_STATE_DIM,):
        raise ValueError(f"mobile_state expected shape (26,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("mobile_state contains NaN or infinity")
    # Conversion validates both Rot6D blocks, including collinearity.
    rot6d_to_xyzw(state[3:9])
    rot6d_to_xyzw(state[13:19])
    return state


def mobile_state26_to_lingbot(state: object) -> dict[str, np.ndarray]:
    state = validate_mobile_state26(state)
    left_pose = np.concatenate((state[0:3], rot6d_to_xyzw(state[3:9])))
    right_pose = np.concatenate((state[10:13], rot6d_to_xyzw(state[13:19])))
    return {
        "observation.state.end.position": np.concatenate((left_pose, right_pose)).astype(np.float32),
        "observation.state.effector.position": state[[9, 19]].astype(np.float32),
        # Dataset conversion intentionally drops map x/y/yaw at [20:23].
        "observation.state.base.position": state[23:26].astype(np.float32),
    }


def _as_action_chunk(value: object, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] == 0:
        raise ValueError(f"{name} expected non-empty shape (T,{width}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def lingbot_actions_to_mobile23(result: Mapping[str, object]) -> np.ndarray:
    required = (
        "action.end.position",
        "action.effector.position",
        "action.base.position",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"LingBot policy result missing keys: {missing}")
    end = _as_action_chunk(result[required[0]], 14, required[0])
    effector = _as_action_chunk(result[required[1]], 2, required[1])
    base = _as_action_chunk(result[required[2]], 3, required[2])
    if len(end) != len(effector) or len(end) != len(base):
        raise ValueError(
            "LingBot action chunks have different horizons: "
            f"end={len(end)}, effector={len(effector)}, base={len(base)}"
        )

    left = np.column_stack((end[:, 0:3], xyzw_to_rot6d(end[:, 3:7]), effector[:, 0]))
    right = np.column_stack((end[:, 7:10], xyzw_to_rot6d(end[:, 10:14]), effector[:, 1]))
    actions23 = np.column_stack((left, right, base)).astype(np.float32)
    if actions23.shape != (len(end), VIRTUAL_MODEL_ACTION_DIM):
        raise AssertionError(f"internal action mapping produced unexpected shape {actions23.shape}")
    return actions23


class LingBotMobileTCP23Adapter:
    """Thin representation adapter around deploy.LingbotVLAv2Server.

    ``policy`` can be injected for tests. Production construction imports the
    heavy LingBot/CUDA stack lazily so config checks and protocol tests stay CPU-only.
    """

    def __init__(self, policy_cfg: dict, policy=None):
        self.policy_cfg = policy_cfg
        self.target_chunk_size = int(policy_cfg.get("target_chunk_size", 10))
        if self.target_chunk_size <= 0:
            raise ValueError("policy.target_chunk_size must be positive")
        self._previous_quaternions: list[np.ndarray | None] = [None, None]
        self.policy = policy if policy is not None else self._load_policy()

    def reset_pose_history(self) -> None:
        self._previous_quaternions = [None, None]

    def _make_quaternions_continuous(self, observation: dict[str, np.ndarray]) -> None:
        poses = observation["observation.state.end.position"]
        for arm_index, start in enumerate((3, 10)):
            quaternion = poses[start : start + 4]
            previous = self._previous_quaternions[arm_index]
            if previous is not None and float(np.dot(previous, quaternion)) < 0.0:
                quaternion *= -1.0
            self._previous_quaternions[arm_index] = quaternion.copy()

    def _load_policy(self):
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        policy = LingbotVLAv2Server(
            path_to_pi_model=self.policy_cfg["checkpoint_dir"],
            robot_norm_path=self.policy_cfg["norm_stats_path"],
            use_length=self.target_chunk_size,
            chunk_ret=True,
            use_bf16=bool(self.policy_cfg.get("use_bf16", True)),
            use_fp32=bool(self.policy_cfg.get("use_fp32", False)),
            use_compile=bool(self.policy_cfg.get("use_compile", False)),
        )
        policy.reset(self.policy_cfg.get("robot_name", "nero_mobile_xyzquat"))
        return policy

    def build_observation(self, inputs: PolicyInputs) -> dict:
        observation = mobile_state26_to_lingbot(inputs.mobile_state26)
        self._make_quaternions_continuous(observation)
        expected_images = {
            "observation.images.cam0",
            "observation.images.cam1",
            "observation.images.cam2",
        }
        if set(inputs.images) != expected_images:
            raise ValueError(
                f"decoded LingBot image keys must be {sorted(expected_images)}, "
                f"got {sorted(inputs.images)}"
            )
        for key, image in inputs.images.items():
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"{key} expected HWC RGB image, got {array.shape}")
            observation[key] = array
        prompt = str(inputs.prompt).strip()
        if not prompt:
            raise ValueError("prompt/task must not be empty")
        observation["task"] = prompt
        return observation

    def infer_actions23(self, inputs: PolicyInputs) -> tuple[np.ndarray, float]:
        observation = self.build_observation(inputs)
        started = time.perf_counter()
        result = self.policy.infer(observation)
        infer_ms = (time.perf_counter() - started) * 1000.0
        actions23 = lingbot_actions_to_mobile23(result)
        if len(actions23) < self.target_chunk_size:
            raise ValueError(
                f"LingBot returned {len(actions23)} actions, fewer than requested "
                f"target_chunk_size={self.target_chunk_size}"
            )
        return actions23[: self.target_chunk_size], infer_ms
