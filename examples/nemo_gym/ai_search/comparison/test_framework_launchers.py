# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the aligned four-framework launchers."""

import hashlib
import subprocess
from pathlib import Path

import yaml


COMPARISON_DIR = Path(__file__).parent
NEMO_SEARCH_R1_LAUNCHER = COMPARISON_DIR / "run_nemo_search_r1.sh"
NEMO_FOUR_WAY_CONFIG = COMPARISON_DIR.parent / "grpo_qwen2_5_7b_search_r1_four_way.yaml"
ORIGINAL_SEARCH_R1_LAUNCHER = COMPARISON_DIR / "run_original_search_r1.sh"
ORIGINAL_SEARCH_R1_PATCH = (
    COMPARISON_DIR / "adapters" / "original-search-r1-comparison.patch"
)
ORIGINAL_SEARCH_R1_RUNTIME_INPUT = (
    COMPARISON_DIR / "adapters" / "original-search-r1-runtime.in"
)
ORIGINAL_SEARCH_R1_RUNTIME_LOCK = (
    COMPARISON_DIR / "adapters" / "original-search-r1-runtime.lock"
)
ORIGINAL_SEARCH_R1_RUNTIME_PREP = (
    COMPARISON_DIR / "prepare_original_search_r1_runtime.sh"
)
CURRENT_VERL_LAUNCHER = COMPARISON_DIR / "run_current_verl_search_r1.sh"
CURRENT_VERL_PATCH = COMPARISON_DIR / "adapters" / "current-verl-comparison.patch"
SLIME_LAUNCHER = COMPARISON_DIR / "run_slime_search_r1.sh"
SLIME_CHECKPOINT_PREP = COMPARISON_DIR / "prepare_slime_search_r1_checkpoint.sh"
SLIME_SEARCH_R1_PATCH = COMPARISON_DIR / "adapters" / "slime-search-r1-comparison.patch"
GRPO_SYNC = COMPARISON_DIR.parents[3] / "nemo_rl" / "algorithms" / "grpo_sync.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_framework_launchers_have_valid_bash_syntax() -> None:
    for launcher in (
        NEMO_SEARCH_R1_LAUNCHER,
        ORIGINAL_SEARCH_R1_LAUNCHER,
        ORIGINAL_SEARCH_R1_RUNTIME_PREP,
        CURRENT_VERL_LAUNCHER,
        SLIME_LAUNCHER,
        SLIME_CHECKPOINT_PREP,
    ):
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_nemo_search_r1_launcher_freezes_aligned_protocol() -> None:
    source = _read(NEMO_SEARCH_R1_LAUNCHER)
    required_fragments = (
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=9904042da053be8e7fa275453c9221324d24aadb6f67323d040e6016da9bfaff",
        "expected_eval_sha256=bdcc57b4c3e88241bf7144f4e739c991c7a0ace4cd1ea26b6c602e2655445645",
        "prompts_per_step=8\n    total_steps=4",
        "train_global_batch_size=40",
        "train_micro_batch_size=5",
        "logprob_batch_size=1",
        "prompts_per_step=512\n    total_steps=500",
        "train_global_batch_size=256",
        "train_micro_batch_size=8",
        "logprob_batch_size=16",
        "data.shuffle=false",
        '"policy.model_name=${model_path}"',
        '"policy.tokenizer.name=${model_path}"',
        "policy.generation.temperature=1.0",
        "policy.generation.top_p=1.0",
        "policy.generation.val_temperature=0.0",
        "policy.generation.val_top_p=1.0",
        "grpo.val_num_generations_per_prompt=1",
        '"checkpointing.enabled=${checkpoint_enabled}"',
        "checkpointing.save_period=250",
        'export AI_SEARCH_OBSERVABILITY_MODE="${observability_mode}"',
        'trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"',
        "engine_prometheus_scope",
        "vllm-http-servers",
        "policy-and-vllm-worker-processes-step-2",
        'health_url="${retriever_url%/retrieve}/healthz"',
        "The strict NeMo four-way launcher does not accept positional overrides.",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source

    config = yaml.safe_load(NEMO_FOUR_WAY_CONFIG.read_text(encoding="utf-8"))
    assert config["data"]["shuffle"] is False
    assert config["policy"]["generation"] == {
        "val_temperature": 0.0,
        "val_top_p": 1.0,
    }


def test_campaign_launchers_save_only_midpoint_and_final_checkpoints() -> None:
    assert "checkpointing.save_period=250" in _read(NEMO_SEARCH_R1_LAUNCHER)
    assert "save_freq=250" in _read(ORIGINAL_SEARCH_R1_LAUNCHER)
    assert "save_freq=250" in _read(CURRENT_VERL_LAUNCHER)
    assert "save_interval=250" in _read(SLIME_LAUNCHER)


def test_four_gpu_diagnostics_are_explicit_and_campaign_stays_eight_gpu() -> None:
    for launcher in (
        NEMO_SEARCH_R1_LAUNCHER,
        ORIGINAL_SEARCH_R1_LAUNCHER,
        CURRENT_VERL_LAUNCHER,
        SLIME_LAUNCHER,
    ):
        source = _read(launcher)
        for fragment in (
            'allow_nonformal_preflight="${SEARCH_R1_ALLOW_NONFORMAL_PREFLIGHT:-0}"',
            '[[ "${allow_nonformal_preflight}" != "1" ]]',
            '[[ "${num_gpus}" != "4" ]]',
            '[[ "${run_mode}" == campaign ]]',
            'hardware_contract="nonformal-four-gpu-${run_mode}"',
            "printf 'hardware_contract=%s\\n'",
        ):
            assert fragment in source


def test_nemo_sync_trainer_prints_classified_response_work() -> None:
    source = _read(GRPO_SYNC)
    for fragment in (
        'rollout_metrics["mean_gen_tokens_per_sample"]',
        'rollout_metrics["mean_env_tokens_per_sample"]',
        'rollout_metrics["mean_total_tokens_per_sample"]',
        "Mean Generation Length:",
        "Mean Observation Length:",
        "Mean Response Sequence Length:",
    ):
        assert fragment in source


def test_original_search_r1_launcher_freezes_aligned_protocol() -> None:
    source = _read(ORIGINAL_SEARCH_R1_LAUNCHER)

    required_fragments = (
        "expected_upstream_base=598e61bd1d36895726d28a8d06b3a15bed19f5d3",
        "expected_patched_head=8f4c5b91e4092fb4d8e858cee0689d339aaf310f",
        "expected_patch_sha256=e8c0e873ccdc2c3220211ca1425de099cb2d7505f902e7658b161e60fa130bf3",
        "expected_runtime_lock_sha256=11a7246f36c9ea844e14b030631d8f8b1489cd245f663a5864bc8b3074d5f269",
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39",
        "expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461",
        "prompts_per_step=8\n    total_steps=4",
        "ppo_mini_batch_size=40\n    ppo_micro_batch_size=40",
        "prompts_per_step=512\n    total_steps=500",
        "ppo_micro_batch_size=64",
        "log_prob_micro_batch_size=128",
        "trainer_stop_step=$((total_steps + 1))",
        '"+actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_outer_steps}"',
        '"+actor_rollout_ref.actor.optim.betas=[0.9,0.999]"',
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.rollout.n_agent=5",
        "data.max_response_length=500",
        "data.max_start_length=2048",
        "data.max_obs_length=500",
        "data.shuffle_train_dataloader=false",
        '"+trainer.val_at_end=${val_at_end}"',
        '"+trainer.validation_predictions_path=${output_dir}/validation/predictions.jsonl"',
        'export AI_SEARCH_METRICS_PATH="${output_dir}/observability/step-metrics.jsonl"',
        'export AI_SEARCH_TENSORBOARD_DIR="${output_dir}/tensorboard"',
        'export AI_SEARCH_PROMETHEUS_PORT="${prometheus_port}"',
        "prometheus_scope=driver-step-metrics",
        "engine_prometheus_scope=missing-embedded-old-vllm",
        'original_python="${ORIGINAL_SEARCH_R1_PYTHON:-python3}"',
        "prepare_original_search_r1_runtime.sh first",
        'ORIGINAL_SEARCH_R1_VENV="${runtime_root}"',
        "The strict original Search-R1 launcher does not accept positional overrides.",
        "trainer_logger=\"['console','local']\"",
        '"trainer.logger=${trainer_logger}"',
        "SEARCH_R1_NSYS_PROFILE_STEP=2",
        "SEARCH_R1_NSYS_OUTPUT_PREFIX=original_search_r1_worker_%p",
        "actor-rollout-and-reference-worker-processes-full-step",
        "max_turns=4",
        "retriever.topk=3",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source


def test_original_search_r1_runtime_is_hash_locked() -> None:
    runtime_input = ORIGINAL_SEARCH_R1_RUNTIME_INPUT.read_bytes()
    runtime_lock = ORIGINAL_SEARCH_R1_RUNTIME_LOCK.read_bytes()
    assert hashlib.sha256(runtime_input).hexdigest() == (
        "c6262fbcd6b5d39cb59589e96a675103fb5da39d0bdccecc3171eb9b34019c49"
    )
    assert hashlib.sha256(runtime_lock).hexdigest() == (
        "11a7246f36c9ea844e14b030631d8f8b1489cd245f663a5864bc8b3074d5f269"
    )
    source = runtime_lock.decode()
    for requirement in (
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/",
        "prometheus-client==0.21.1",
        "ray==2.10.0",
        "tensorboard==2.17.1",
        "tensordict==0.5.0",
        "torch==2.4.0+cu121",
        "torchvision==0.19.0+cu121",
        "transformers==4.47.1",
        "vllm==0.6.3",
    ):
        assert requirement in source
    prep = _read(ORIGINAL_SEARCH_R1_RUNTIME_PREP)
    for fragment in (
        "expected_uv_version='uv 0.12.3 (x86_64-unknown-linux-gnu)'",
        "--torch-backend cu121",
        "--require-hashes",
        'if [[ "$(uname -m)" != x86_64 ]]',
        "if torch._C._GLIBCXX_USE_CXX11_ABI is not False:",
        "ORIGINAL_SEARCH_R1_RUNTIME_CREATE_PASS",
        "ORIGINAL_SEARCH_R1_RUNTIME_REUSE_PASS",
    ):
        assert fragment in prep
    assert "--seed" not in prep


def test_original_search_r1_patch_is_frozen_and_observable() -> None:
    patch = ORIGINAL_SEARCH_R1_PATCH.read_bytes()
    assert hashlib.sha256(patch).hexdigest() == (
        "e8c0e873ccdc2c3220211ca1425de099cb2d7505f902e7658b161e60fa130bf3"
    )
    source = patch.decode()
    for fragment in (
        "X-NeMo-Search-Batch-ID",
        "trace_identity_batch",
        "build_trace_identities",
        "OriginalSearchR1LocalLogger",
        "validation_predictions_path",
        "lr_warmup_steps",
        "val_at_end",
        "SEARCH_R1_NSYS_PROFILE_STEP",
        "start_nsys_profile",
        "capture-range",
        "prometheus-client==0.21.1",
        "tensorboard==2.17.1",
    ):
        assert fragment in source


def test_current_verl_launcher_freezes_aligned_protocol() -> None:
    source = _read(CURRENT_VERL_LAUNCHER)

    required_fragments = (
        "expected_verl_upstream_base=5cfb74fa04c7f6e5d98260b8f05157c6a9402695",
        "expected_verl_patched_head=fb72e8b195095ac3334e870176eb6eaa80184001",
        "expected_verl_patch_sha256=3a11219896823a85e7f2527351b9071b118e87564b3a359f2b4c26f1257980c4",
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39",
        "expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461",
        "prompts_per_step=8\n    total_steps=4",
        "ppo_mini_batch_size=8\n    ppo_micro_batch_size_per_gpu=5",
        "prompts_per_step=512\n    total_steps=500",
        "ppo_micro_batch_size_per_gpu=8",
        "log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.n=5",
        "data.max_prompt_length=2048",
        "data.max_response_length=4096",
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}",
        '"actor_rollout_ref.actor.optim.betas=[0.9,0.999]"',
        "lr_schedule_horizon_outer_steps=500",
        "lr_warmup_outer_steps=142",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        '"actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_outer_steps}"',
        "actor_rollout_ref.actor.optim.optimizer=AdamW",
        '"actor_rollout_ref.actor.optim.override_optimizer_config={eps:1.0e-8}"',
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_text",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
        "reward.custom_reward_function.name=verl_compute_score",
        "trainer.total_training_steps=${total_steps}",
        "SWANLAB_MODE=local",
        "['console','swanlab','tensorboard']",
        "default_trace_sample_rate=0.1",
        'export AI_SEARCH_TRACE_PATH="${output_dir}/trajectory-spans.jsonl"',
        "printf 'trace_sample_rate=%s\\n' \"${AI_SEARCH_TRACE_SAMPLE_RATE:-disabled}\"",
        "SEARCH_R1_OUTPUT_DIR already contains a run manifest",
        "The strict current-veRL launcher does not accept positional overrides.",
        'observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"',
        "global_profiler.tool=nsys",
        '"global_profiler.steps=[2]"',
        "actor_rollout_ref.actor.profiler.enable=true",
        "actor_rollout_ref.ref.profiler.enable=true",
        '"+ray_kwargs.ray_init._temp_dir=${output_dir}/ray"',
        "actor-and-reference-worker-processes-full-step",
        "nsys_rollout_engine_scope",
        "measurement_work_counters=scalar-transfer-queue-tags",
        "timed_rollout_text_dump=disabled",
        "trainer.rollout_data_dir=null",
        "disable_log_stats=false",
        "engine_prometheus_enabled=true",
        "actor_rollout_ref.rollout.disable_log_stats=${disable_log_stats}",
        "actor_rollout_ref.rollout.prometheus.enable=${engine_prometheus_enabled}",
        "actor_rollout_ref.rollout.prometheus.file=${output_dir}/observability/rollout-prometheus.yml",
        "actor_rollout_ref.rollout.prometheus.served_model_name=Qwen2.5-7B",
        "engine_prometheus_scope=vllm-http-servers",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source
    assert "optimizer_updates_per_outer_step=%s" in source
    assert '"$((prompts_per_step / ppo_mini_batch_size))"' in source


def test_current_verl_patch_is_frozen_and_measurement_only() -> None:
    patch = CURRENT_VERL_PATCH.read_bytes()
    assert hashlib.sha256(patch).hexdigest() == (
        "3a11219896823a85e7f2527351b9071b118e87564b3a359f2b4c26f1257980c4"
    )
    source = patch.decode()
    for fragment in (
        "Scalar, measurement-only Search-R1 work counters",
        '"generated_tokens"',
        '"observation_tokens"',
        '"search_count"',
        '"search_errors"',
        '"invalid_actions"',
        '"trajectory_wall_seconds"',
        "metrics.update(work_metrics)",
    ):
        assert fragment in source
    assert "rollout_data_dir" not in source


def test_slime_launcher_freezes_aligned_protocol() -> None:
    source = _read(SLIME_LAUNCHER)

    required_fragments = (
        "expected_slime_base=a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e",
        "expected_slime_patched_head=3c30f8b4954e40f7baa0073d83a62134c1aec82b",
        "expected_slime_patch_sha256=50b55d2602e0a9425dbaab5e16430080a54b39e9b340627f857addd3cdb7cb5d",
        "expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796",
        "expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39",
        "expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461",
        "prompts_per_step=8\n    num_rollout=4\n    global_batch_size=40",
        "actor_microbatch_mode=dynamic\\nactor_max_tokens_per_gpu=9216",
        "prompts_per_step=512\n    num_rollout=500\n    global_batch_size=256",
        '--actor-num-gpus-per-node "${num_gpus}"',
        '--rollout-num-gpus "${num_gpus}"',
        "--colocate",
        "--n-samples-per-prompt 5",
        "--rollout-max-prompt-len 2048",
        "--rollout-max-response-len 500",
        "--rollout-max-context-len 6144",
        "--kl-loss-coef 0.001",
        "--eps-clip 0.2",
        "--lr-decay-style constant",
        '--lr-decay-iters "${lr_schedule_horizon_optimizer_updates}"',
        "lr_schedule_horizon_optimizer_updates=5000",
        '--lr-warmup-iters "${lr_warmup_optimizer_updates}"',
        "lr_warmup_optimizer_updates=1420",
        "--adam-beta1 0.9",
        "--adam-beta2 0.999",
        "--adam-eps 1e-8",
        "--custom-generate-function-path slime_search_r1_adapter.generate",
        "--custom-rm-path framework_eval_adapters.slime_reward",
        "--custom-megatron-before-train-step-hook-path slime_search_r1_adapter.align_outer_step_lr",
        "save_debug_rollouts=0",
        "save_debug_rollouts=1",
        'if [[ "${save_debug_rollouts}" == 1 ]]',
        "timed_rollout_text_dump=%s",
        "--use-tensorboard",
        "printf 'swanlab_source=%s\\n'",
        "default_trace_sample_rate=0.1",
        'export AI_SEARCH_TRACE_PATH="${output_dir}/trajectory-spans.jsonl"',
        '"AI_SEARCH_TRACE_SAMPLE_RATE":"%s"',
        "SEARCH_R1_OUTPUT_DIR already contains a run manifest",
        "The strict slime launcher does not accept positional overrides.",
        'observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"',
        "SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID=1",
        '"capture-range":"cudaProfilerApi"',
        "engine_prometheus_scope=not-collected-native-always-on",
        "engine_prometheus_scope=sglang-router-and-engine-http-servers",
        'ray_tmpdir="${SEARCH_R1_RAY_TMPDIR:-${output_dir}/ray}"',
        'if (( ${#ray_tmpdir} > 32 )); then',
        '--temp-dir "${ray_tmpdir}"',
    )
    for fragment in required_fragments:
        assert fragment in source
    assert 'if [[ "${num_gpus}" != "8" ]]' in source
    assert "physical_gpus=%s\\nactor_gpus=%s\\nrollout_gpus=%s" in source
    assert "pkill" not in source


def test_slime_profile_patch_is_frozen_and_measurement_only() -> None:
    patch = SLIME_SEARCH_R1_PATCH.read_bytes()
    assert hashlib.sha256(patch).hexdigest() == (
        "50b55d2602e0a9425dbaab5e16430080a54b39e9b340627f857addd3cdb7cb5d"
    )
    source = patch.decode()
    for fragment in (
        "SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID",
        "start_nsys_profile",
        "stop_nsys_profile",
        "search_r1_outer_step",
    ):
        assert fragment in source


def test_slime_checkpoint_conversion_requires_exact_roundtrip() -> None:
    source = _read(SLIME_CHECKPOINT_PREP)
    for fragment in (
        "expected_slime_patched_head=3c30f8b4954e40f7baa0073d83a62134c1aec82b",
        "slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f",
        "--ckpt-format torch_dist",
        "convert_torch_dist_to_hf.py",
        "verify_safetensors_equivalence.py",
        "safetensors-equivalence.json",
        "checkpoint-files.sha256",
        'chmod -R a+rX -- "${ref_load}"',
        "timed_benchmark_work",
        "SEARCH_R1_SLIME_CHECKPOINT_PREP_PASS",
    ):
        assert fragment in source


def test_strict_framework_launchers_reject_positional_overrides() -> None:
    for launcher in (
        NEMO_SEARCH_R1_LAUNCHER,
        ORIGINAL_SEARCH_R1_LAUNCHER,
        CURRENT_VERL_LAUNCHER,
        SLIME_LAUNCHER,
    ):
        source = _read(launcher)
        assert "does not accept positional overrides" in source
        assert '"${@}"' not in source


def test_all_framework_launchers_share_observability_modes() -> None:
    for launcher in (
        NEMO_SEARCH_R1_LAUNCHER,
        ORIGINAL_SEARCH_R1_LAUNCHER,
        CURRENT_VERL_LAUNCHER,
        SLIME_LAUNCHER,
    ):
        source = _read(launcher)
        for fragment in (
            'observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"',
            "baseline)",
            "clean)",
            "profile)",
            'trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"',
            "SEARCH_R1_OBSERVABILITY_MODE must be baseline, clean, or profile.",
            "SEARCH_R1_OBSERVABILITY_MODE=profile requires SEARCH_R1_RUN_MODE=performance.",
            '"${run_mode}" == campaign && "${observability_mode}" == clean',
        ):
            assert fragment in source


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
    assert config["eval"]["defaults"]["temperature"] == 0.0
