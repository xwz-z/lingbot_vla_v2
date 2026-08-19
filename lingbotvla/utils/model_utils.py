# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch.nn as nn

from . import logging


logger = logging.get_logger(__name__)


_EXPERT_ONLY_VLM_PATH = "model.qwenvl_with_expert.qwenvl"
_EXPERT_ONLY_ACTION_EXPERT_PATH = "model.qwenvl_with_expert.qwen_expert"
_EXPERT_ONLY_PROJECTION_PATHS = (
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    module = model
    for name in path.split("."):
        if not hasattr(module, name):
            raise RuntimeError(f"Expert-only audit expected model submodule {path!r}, but it was not found.")
        module = getattr(module, name)
    if not isinstance(module, nn.Module):
        raise RuntimeError(f"Expert-only audit expected {path!r} to be a torch module.")
    return module


def _parameter_count(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def audit_expert_only_parameters(model: nn.Module):
    """Return parameter counts for the native LingBot expert-only training boundary."""
    vlm = _get_submodule(model, _EXPERT_ONLY_VLM_PATH)
    action_expert = _get_submodule(model, _EXPERT_ONLY_ACTION_EXPERT_PATH)
    projection_modules = [_get_submodule(model, path) for path in _EXPERT_ONLY_PROJECTION_PATHS]

    total = _parameter_count(model)
    trainable = _parameter_count(model, trainable_only=True)
    vlm_total = _parameter_count(vlm)
    vlm_trainable = _parameter_count(vlm, trainable_only=True)
    action_expert_total = _parameter_count(action_expert)
    action_expert_trainable = _parameter_count(action_expert, trainable_only=True)
    projection_trainable = sum(_parameter_count(module, trainable_only=True) for module in projection_modules)

    return {
        "total": total,
        "trainable": trainable,
        "vlm_total": vlm_total,
        "vlm_trainable": vlm_trainable,
        "action_expert_total": action_expert_total,
        "action_expert_trainable": action_expert_trainable,
        "projection_trainable": projection_trainable,
        "auxiliary_trainable": trainable - action_expert_trainable - projection_trainable,
    }


def validate_expert_only_parameters(model: nn.Module):
    """Fail fast if the model does not match the intended expert-only trainable boundary."""
    vlm = _get_submodule(model, _EXPERT_ONLY_VLM_PATH)
    action_expert = _get_submodule(model, _EXPERT_ONLY_ACTION_EXPERT_PATH)
    projection_modules = [_get_submodule(model, path) for path in _EXPERT_ONLY_PROJECTION_PATHS]

    leaked_vlm_parameters = [name for name, parameter in vlm.named_parameters() if parameter.requires_grad]
    if leaked_vlm_parameters:
        preview = ", ".join(leaked_vlm_parameters[:5])
        raise RuntimeError(
            "Expert-only training requires the complete Qwen-VL backbone (including vision) to be frozen; "
            f"found {len(leaked_vlm_parameters)} trainable parameter(s), for example: {preview}."
        )

    if _parameter_count(action_expert, trainable_only=True) == 0:
        raise RuntimeError("Expert-only training found no trainable parameters in the action expert.")

    frozen_projection_paths = [
        path
        for path, module in zip(_EXPERT_ONLY_PROJECTION_PATHS, projection_modules)
        if _parameter_count(module, trainable_only=True) == 0
    ]
    if frozen_projection_paths:
        raise RuntimeError(
            "Expert-only training requires the action/state projection modules to remain trainable; "
            f"fully frozen modules: {', '.join(frozen_projection_paths)}."
        )

    stats = audit_expert_only_parameters(model)
    if stats["trainable"] == 0:
        raise RuntimeError("Expert-only training found no trainable model parameters.")
    return stats


def format_expert_only_parameter_stats(stats) -> str:
    """Format the startup audit without dumping every parameter name."""
    trainable_ratio = 100.0 * stats["trainable"] / max(1, stats["total"])
    return "\n".join(
        [
            "Expert-only parameter audit:",
            f"  total parameters:             {stats['total']:,}",
            f"  trainable parameters:         {stats['trainable']:,} ({trainable_ratio:.2f}%)",
            f"  frozen Qwen-VL parameters:    {stats['vlm_total'] - stats['vlm_trainable']:,}",
            f"  trainable action expert:      {stats['action_expert_trainable']:,}",
            f"  trainable action projections: {stats['projection_trainable']:,}",
            f"  trainable auxiliary modules:  {stats['auxiliary_trainable']:,}",
        ]
    )


def pretty_print_trainable_parameters(model: nn.Module):
    trainable_parameters = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_parameters.append(n)

    printable_results = {}
    for p in trainable_parameters:
        param_split = p.split(".")
        param_name = ""
        digit_index = 0
        layer_index_list = []
        for split_item in param_split:
            if split_item.isdigit():
                param_name += f"<{digit_index}>."
                layer_index_list.append(int(split_item))
                digit_index += 1
            else:
                param_name += f"{split_item}."
        param_name = param_name[:-1]

        if param_name not in printable_results:
            printable_results[param_name] = []
        printable_results[param_name].append(layer_index_list)

    train_param_info = "\n**** trainable parameters ****"
    for param_key in printable_results.keys():
        layer_idxs = np.array(printable_results[param_key])
        if layer_idxs.shape[-1] == 0:
            train_param_info += "\n" + param_key
            continue
        layer_min = layer_idxs.min(axis=0)
        layer_max = layer_idxs.max(axis=0)
        print_pattern = param_key
        for index in range(len(layer_min)):
            if layer_min[index] == layer_max[index]:
                print_pattern = print_pattern.replace(f"<{index}>", f"[{layer_min[index]}]")
            else:
                print_pattern = print_pattern.replace(f"<{index}>", f"[{layer_min[index]}-{layer_max[index]}]")
        train_param_info += "\n" + print_pattern
    train_param_info += "\n**** trainable parameters ****"
    logger.info_rank0(train_param_info)
