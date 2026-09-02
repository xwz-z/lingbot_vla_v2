#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .inference_service import LingBotInferenceService
except ImportError:  # Direct script execution.
    from inference_service import LingBotInferenceService


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def load_config(path: str | None) -> dict:
    config_name = path or os.environ.get("REAL_WORLD_INFERENCE_CONFIG")
    if not config_name:
        config_name = str(SCRIPT_DIR / "config" / "mobile_transfer_lingbot_new40_tcp23.json")
    config_path = Path(config_name).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict, check_paths: bool = True) -> None:
    server = config.get("server")
    policy = config.get("policy")
    if not isinstance(server, dict) or not isinstance(policy, dict):
        raise ValueError("config requires server and policy objects")
    if policy.get("inference_format") != "mobile_tcp23":
        raise ValueError("policy.inference_format must be 'mobile_tcp23'")
    if int(policy.get("action_dim", 0)) != 23:
        raise ValueError("policy.action_dim must be 23 for wire-compatible mobile_tcp23")
    if int(policy.get("target_chunk_size", 0)) <= 0:
        raise ValueError("policy.target_chunk_size must be positive")
    if bool(policy.get("use_bf16", True)) == bool(policy.get("use_fp32", False)):
        raise ValueError("exactly one of policy.use_bf16 and policy.use_fp32 must be true")
    for key in ("checkpoint_dir", "norm_stats_path", "robot_name", "task_prompt"):
        if not str(policy.get(key, "")).strip():
            raise ValueError(f"missing policy.{key}")
    image_inputs = policy.get("image_inputs")
    expected = {"head_fpv", "left_hand", "right_hand"}
    if not isinstance(image_inputs, dict) or set(image_inputs) != expected:
        raise ValueError(f"policy.image_inputs keys must be {sorted(expected)}")
    int(server.get("port", 8027))
    if check_paths:
        for key in ("checkpoint_dir", "norm_stats_path"):
            path = Path(policy[key]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"policy.{key} does not exist: {path}")
        robot_config = PROJECT_DIR / "configs" / "robot_configs" / f"{policy['robot_name']}.yaml"
        if not robot_config.exists():
            raise FileNotFoundError(f"robot config does not exist: {robot_config}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree-compatible LingBot VLA V2 HTTP server")
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-runtime", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    validate_config(config)
    if arguments.print_runtime:
        print(config["server"].get("host", "0.0.0.0"))
        print(int(config["server"].get("port", 8027)))
        print(config["policy"]["inference_format"])
        print(int(config["policy"]["action_dim"]))
        return
    if arguments.check:
        print("[LingBot inference] configuration check passed")
        print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
        return
    os.chdir(PROJECT_DIR)
    host = arguments.host or config["server"].get("host", "0.0.0.0")
    port = arguments.port or int(config["server"].get("port", 8027))
    LingBotInferenceService(config).serve_http(host, port)


if __name__ == "__main__":
    main()
