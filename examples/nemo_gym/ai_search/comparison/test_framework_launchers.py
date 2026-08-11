# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the aligned current-veRL and slime launchers."""

import subprocess
from pathlib import Path

import yaml


COMPARISON_DIR = Path(__file__).parent
CURRENT_VERL_LAUNCHER = COMPARISON_DIR / "run_current_verl_search_r1.sh"
SLIME_LAUNCHER = COMPARISON_DIR / "run_slime_search_r1.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_framework_launchers_have_valid_bash_syntax() -> None:
    for launcher in (CURRENT_VERL_LAUNCHER, SLIME_LAUNCHER):
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_current_verl_launcher_freezes_aligned_protocol() -> None:
    source = _read(CURRENT_VERL_LAUNCHER)

    required_fragments = (
        "expected_verl_commit=5cfb74fa04c7f6e5d98260b8f05157c6a9402695",
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39",
        "expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461",
        "prompts_per_step=8\n    total_steps=4",
        "prompts_per_step=512\n    total_steps=500",
        "actor_rollout_ref.rollout.n=5",
        "data.max_prompt_length=2048",
        "data.max_response_length=4096",
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_text",
        "reward.custom_reward_function.name=verl_compute_score",
        "trainer.total_training_steps=${total_steps}",
        "SWANLAB_MODE=local",
        "['console','swanlab','tensorboard']",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source


def test_slime_launcher_freezes_aligned_protocol() -> None:
    source = _read(SLIME_LAUNCHER)

    required_fragments = (
        "expected_slime_commit=a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e",
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39",
        "expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461",
        "prompts_per_step=8\n    num_rollout=4\n    global_batch_size=40",
        "prompts_per_step=512\n    num_rollout=500\n    global_batch_size=256",
        "--actor-num-gpus-per-node 4",
        "--rollout-num-gpus 4",
        "--colocate",
        "--n-samples-per-prompt 5",
        "--rollout-max-prompt-len 2048",
        "--rollout-max-response-len 500",
        "--rollout-max-context-len 6144",
        "--kl-loss-coef 0.001",
        "--eps-clip 0.2",
        "--custom-generate-function-path slime_search_r1_adapter.generate",
        "--custom-rm-path framework_eval_adapters.slime_reward",
        "--save-debug-rollout-data",
        "--use-tensorboard",
        'printf \'swanlab_source=post-run-tensorboard-conversion\\n\'',
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source
    assert "pkill" not in source


def test_slime_eval_template_uses_runtime_common_dataset() -> None:
    config = yaml.safe_load(
        (COMPARISON_DIR / "adapters" / "slime-search-r1-eval.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["eval"]["datasets"] == [
        {
            "name": "official_search_r1_test",
            "path": "${oc.env:SEARCH_R1_EVAL_FILE}",
        }
    ]
