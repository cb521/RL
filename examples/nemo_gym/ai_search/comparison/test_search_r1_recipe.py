# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the strict Search-R1 training recipe."""

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)


def test_search_r1_recipe_instantiates_master_config() -> None:
    repo_root = Path(__file__).parents[4]
    recipe = repo_root / "examples/nemo_gym/ai_search/grpo_qwen2_5_7b_search_r1.yaml"

    register_omegaconf_resolvers()
    raw_config = load_config_with_inheritance(recipe)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    config = MasterConfig(**resolved_config)
    assert config.policy["tokenizer"]["chat_template"] is None
    assert config.policy["generation"]["vllm_cfg"]["enforce_eager"] is True
    assert config.policy["generation"]["vllm_cfg"]["use_torch_logprobs"] is True


def test_search_r1_resources_allow_remote_retriever(monkeypatch) -> None:
    repo_root = Path(__file__).parents[4]
    resources_config = (
        repo_root
        / "examples/nemo_gym/ai_search/resources_servers/ai_search/configs"
        / "ai_search_search_r1.yaml"
    )
    retriever_url = "http://retriever.example:8000/retrieve"

    monkeypatch.setenv("AI_SEARCH_RETRIEVER_URL", retriever_url)
    raw_config = OmegaConf.load(resources_config)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    assert (
        resolved_config["ai_search"]["resources_servers"]["ai_search"]["search"][
            "http_url"
        ]
        == retriever_url
    )


def test_search_r1_agent_allows_local_tokenizer(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[4]
    resources_config = (
        repo_root
        / "examples/nemo_gym/ai_search/resources_servers/ai_search/configs"
        / "ai_search_search_r1.yaml"
    )
    tokenizer_path = tmp_path / "model-snapshot"

    monkeypatch.setenv("SEARCH_R1_MODEL_PATH", str(tokenizer_path))
    raw_config = OmegaConf.load(resources_config)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    agent_config = resolved_config["ai_search_search_r1_agent"][
        "responses_api_agents"
    ]["search_r1_agent"]
    assert agent_config["tokenizer_name"] == str(tokenizer_path)


def test_observed_recipe_enables_local_capable_loggers() -> None:
    repo_root = Path(__file__).parents[4]
    recipe = (
        repo_root
        / "examples/nemo_gym/ai_search/grpo_qwen2_5_7b_search_r1_observed.yaml"
    )

    raw_config = load_config_with_inheritance(recipe)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    logger = resolved_config["logger"]
    assert logger["wandb_enabled"] is False
    assert logger["swanlab_enabled"] is True
    assert logger["tensorboard_enabled"] is True
    assert logger["monitor_gpus"] is True
