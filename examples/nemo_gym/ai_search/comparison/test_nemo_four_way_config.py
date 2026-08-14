# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved-config tests for the strict NeMo four-way Search-R1 launcher."""

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config_with_inheritance,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


RECIPE = Path(__file__).parents[1] / "grpo_qwen2_5_7b_search_r1_four_way.yaml"


def _resolved(overrides: list[str] | None = None) -> dict:
    register_omegaconf_resolvers()
    config = load_config_with_inheritance(RECIPE)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def test_campaign_config_preserves_strict_optimizer_and_order() -> None:
    config = _resolved()
    assert config["data"]["shuffle"] is False
    assert config["grpo"]["max_num_steps"] == 500
    assert config["grpo"]["num_prompts_per_step"] == 512
    assert config["grpo"]["num_generations_per_prompt"] == 5
    assert config["policy"]["train_global_batch_size"] == 256
    assert config["policy"]["train_micro_batch_size"] == 8
    assert config["policy"]["logprob_batch_size"] == 16
    assert config["policy"]["optimizer"] == {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 1.0e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
    }
    assert config["policy"]["scheduler"][0]["kwargs"]["total_iters"] == 142
    assert config["policy"]["generation"]["val_temperature"] == 0.0
    assert config["policy"]["dtensor_cfg"]["offload_optimizer_during_refit"] is False
    assert config["policy"]["dtensor_cfg"]["offload_model_during_refit"] is False
    generation = config["policy"]["generation"]
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["compilation_config"] == {
        "backend": "eager",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": list(range(1, 11)),
    }


def test_performance_overrides_resolve_to_one_update_per_outer_step() -> None:
    config = _resolved(
        [
            "grpo.max_num_steps=4",
            "grpo.num_prompts_per_step=8",
            "grpo.num_generations_per_prompt=5",
            "policy.train_global_batch_size=40",
            "policy.train_micro_batch_size=5",
            "policy.logprob_batch_size=1",
            "grpo.val_at_start=false",
            "grpo.val_at_end=false",
            "checkpointing.enabled=false",
        ]
    )
    assert config["grpo"]["max_num_steps"] == 4
    trajectories = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
    )
    assert trajectories == 40
    assert trajectories // config["policy"]["train_global_batch_size"] == 1
    assert config["policy"]["train_micro_batch_size"] == 5
    assert config["policy"]["logprob_batch_size"] == 1
    assert config["grpo"]["val_at_start"] is False
    assert config["grpo"]["val_at_end"] is False
    assert config["checkpointing"]["enabled"] is False


def test_observed_vllm_metric_overrides_resolve() -> None:
    register_omegaconf_resolvers()
    config = load_config_with_inheritance(RECIPE)
    config = parse_hydra_overrides(
        config,
        [
            "++policy.generation.vllm_cfg.enable_vllm_metrics_logger=true",
            "++policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5",
        ],
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    vllm_config = resolved["policy"]["generation"]["vllm_cfg"]
    assert vllm_config["enable_vllm_metrics_logger"] is True
    assert vllm_config["vllm_metrics_logger_interval"] == 0.5
