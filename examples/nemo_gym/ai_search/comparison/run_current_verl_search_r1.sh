#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nemo_root=$(cd -- "${comparison_dir}/../../../.." && pwd)
verl_root="${CURRENT_VERL_ROOT:?CURRENT_VERL_ROOT must point to the frozen veRL checkout}"
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
train_file="${SEARCH_R1_TRAIN_FILE:?SEARCH_R1_TRAIN_FILE must point to four_way_train/train.parquet}"
eval_file="${SEARCH_R1_EVAL_FILE:?SEARCH_R1_EVAL_FILE must point to four_way_eval/test.parquet}"
retriever_url="${SEARCH_R1_RETRIEVER_URL:?SEARCH_R1_RETRIEVER_URL must point to the shared E5 /retrieve endpoint}"
output_dir="${SEARCH_R1_OUTPUT_DIR:?SEARCH_R1_OUTPUT_DIR must be set}"
run_mode="${SEARCH_R1_RUN_MODE:-smoke}"
observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"
seed="${SEARCH_R1_SEED:-42}"
num_gpus="${SEARCH_R1_NUM_GPUS:-8}"
allow_nonformal_preflight="${SEARCH_R1_ALLOW_NONFORMAL_PREFLIGHT:-0}"
workload_mode="${SEARCH_R1_WORKLOAD_MODE:-training}"
fast_iteration="${SEARCH_R1_FAST_ITERATION:-0}"
controlled_data_manifest="${SEARCH_R1_CONTROLLED_DATA_MANIFEST:-}"
filter_overlong_prompts=true
lr_schedule_horizon_outer_steps=500
lr_warmup_outer_steps=142

expected_verl_upstream_base=5cfb74fa04c7f6e5d98260b8f05157c6a9402695
expected_verl_patched_head=fb72e8b195095ac3334e870176eb6eaa80184001
expected_verl_patch_sha256=3a11219896823a85e7f2527351b9071b118e87564b3a359f2b4c26f1257980c4
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39
expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461
expected_controlled_train_sha256=213253af32ebad44378a47329e9f99a26dbda81d3175ac6a2c11d0f001ef3b20
expected_controlled_manifest_sha256=09b4fa8a7127873c9083ff7dfa736ff087ad2c0d6b570dd6349d7bbfe6db1003
controlled_semantic_rows_sha256=disabled

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    total_steps=1
    # Current veRL expresses this value in prompts and multiplies it by
    # rollout.n internally. Eight prompts therefore become the same 40
    # trajectory global mini-batch used by the other launchers.
    ppo_mini_batch_size=8
    ppo_micro_batch_size_per_gpu=5
    log_prob_micro_batch_size_per_gpu=1
    agent_workers=40
    save_freq=-1
    test_freq=-1
    val_before_train=false
    default_trace_sample_rate=1.0
    ;;
  performance)
    prompts_per_step=8
    total_steps=4
    if [[ "${fast_iteration}" == "1" ]]; then
      total_steps=2
    fi
    ppo_mini_batch_size=8
    ppo_micro_batch_size_per_gpu=5
    log_prob_micro_batch_size_per_gpu=1
    agent_workers=40
    save_freq=-1
    test_freq=-1
    val_before_train=false
    default_trace_sample_rate=0.1
    ;;
  campaign)
    prompts_per_step=512
    total_steps=500
    ppo_mini_batch_size=256
    ppo_micro_batch_size_per_gpu=8
    log_prob_micro_batch_size_per_gpu=16
    agent_workers=256
    save_freq=250
    test_freq=50
    val_before_train=true
    default_trace_sample_rate=0.01
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

case "${fast_iteration}" in
  0)
    iteration_mode=full
    case "${run_mode}" in
      smoke)
        warmup_steps=0
        measured_steps=1
        ;;
      performance)
        warmup_steps=1
        measured_steps=2,3,4
        ;;
      campaign)
        warmup_steps=0
        measured_steps=1-500
        ;;
    esac
    ;;
  1)
    if [[ "${run_mode}" != performance ]]; then
      echo "SEARCH_R1_FAST_ITERATION=1 is only valid for performance runs." >&2
      exit 1
    fi
    iteration_mode=fast-screening
    warmup_steps=1
    measured_steps=2
    ;;
  *)
    echo "SEARCH_R1_FAST_ITERATION must be 0 or 1, not ${fast_iteration}." >&2
    exit 1
    ;;
esac

case "${workload_mode}" in
  training)
    optimizer_lr=1e-6
    rollout_request_temperature=1.0
    prompt_encoding=qwen-chat-template
    weights_frozen=false
    rollout_batch_invariant=false
    deterministic_request_identity=false
    ;;
  controlled)
    if [[ "${run_mode}" != performance ]] \
      || [[ "${observability_mode}" != clean && "${observability_mode}" != profile ]]; then
      echo "SEARCH_R1_WORKLOAD_MODE=controlled requires a clean or profile performance run." >&2
      exit 1
    fi
    optimizer_lr=0.0
    rollout_request_temperature=0.0
    prompt_encoding=search-r1-content-concat
    weights_frozen=true
    rollout_batch_invariant=true
    deterministic_request_identity=true
    # Controlled performance runs consume the fixed 16-row, two-step fixture.
    # Keep truncation=error as the safety gate, but avoid scanning the full
    # training corpus before every one-step profiling iteration.
    filter_overlong_prompts=false
    expected_train_sha256="${expected_controlled_train_sha256}"
    controlled_semantic_rows_sha256=281e6efdf2a9a3d083010f68d82e20d6bd8469976ee1fffdcf6a0616e158b75d
    if [[ -z "${controlled_data_manifest}" ]]; then
      echo "SEARCH_R1_CONTROLLED_DATA_MANIFEST is required for controlled work." >&2
      exit 1
    fi
    ;;
  *)
    echo "SEARCH_R1_WORKLOAD_MODE must be training or controlled, not ${workload_mode}." >&2
    exit 1
    ;;
esac

case "${observability_mode}" in
  baseline)
    logger="['console']"
    trace_sample_rate=disabled
    disable_log_stats=true
    engine_prometheus_enabled=false
    engine_prometheus_scope=disabled
    ;;
  clean)
    logger="['console','swanlab','tensorboard']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-${default_trace_sample_rate}}"
    disable_log_stats=false
    engine_prometheus_enabled=true
    engine_prometheus_scope=vllm-http-servers
    ;;
  profile)
    logger="['console','swanlab','tensorboard']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"
    disable_log_stats=false
    engine_prometheus_enabled=true
    engine_prometheus_scope=vllm-http-servers
    ;;
  *)
    echo "SEARCH_R1_OBSERVABILITY_MODE must be baseline, clean, or profile." >&2
    exit 1
    ;;
esac
if [[ "${workload_mode}" == controlled ]]; then
  trace_sample_rate=1.0
fi
if [[ "${observability_mode}" == profile && "${run_mode}" != performance ]]; then
  echo "SEARCH_R1_OBSERVABILITY_MODE=profile requires SEARCH_R1_RUN_MODE=performance." >&2
  exit 1
fi

if (( $# != 0 )); then
  echo "The strict current-veRL launcher does not accept positional overrides." >&2
  exit 1
fi

actual_verl_head=$(git -C "${verl_root}" rev-parse HEAD)
if ! git -C "${verl_root}" merge-base --is-ancestor \
  "${expected_verl_upstream_base}" "${actual_verl_head}"; then
  echo "Current veRL patch head does not descend from ${expected_verl_upstream_base}." >&2
  exit 1
fi
actual_verl_patch_sha256="$({
  git -C "${verl_root}" diff --full-index --binary --unified=0 \
    "${expected_verl_upstream_base}..${actual_verl_head}"
} | sha256sum | cut -d ' ' -f 1)"
if [[ "${actual_verl_patch_sha256}" != "${expected_verl_patch_sha256}" ]]; then
  echo "Current veRL comparison diff does not match ${expected_verl_patch_sha256}." >&2
  exit 1
fi
if ! git -C "${verl_root}" diff --quiet || ! git -C "${verl_root}" diff --cached --quiet; then
  echo "Current veRL checkout has uncommitted tracked changes." >&2
  exit 1
fi
if [[ "$(basename -- "$(readlink -f "${model_path}")")" != "${expected_model_revision}" ]]; then
  echo "SEARCH_R1_MODEL_PATH does not resolve to revision ${expected_model_revision}." >&2
  exit 1
fi
for required_file in \
  "${train_file}" \
  "${eval_file}" \
  "${comparison_dir}/framework_eval_adapters.py" \
  "${comparison_dir}/verl_search_r1_adapter.py" \
  "${comparison_dir}/adapters/current-verl-comparison.patch" \
  "${comparison_dir}/adapters/current-verl-search-r1-agent-loop.example.yaml"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing aligned current-veRL artifact: ${required_file}" >&2
    exit 1
  fi
done
printf '%s  %s\n' \
  "${expected_verl_patch_sha256}" \
  "${comparison_dir}/adapters/current-verl-comparison.patch" | sha256sum --check --status
printf '%s  %s\n' "${expected_train_sha256}" "${train_file}" | sha256sum --check --status
printf '%s  %s\n' "${expected_eval_sha256}" "${eval_file}" | sha256sum --check --status
if [[ "${workload_mode}" == controlled ]]; then
  printf '%s  %s\n' \
    "${expected_controlled_manifest_sha256}" \
    "${controlled_data_manifest}" | sha256sum --check --status
fi
if [[ "${retriever_url}" != http://*/retrieve && "${retriever_url}" != https://*/retrieve ]]; then
  echo "SEARCH_R1_RETRIEVER_URL must be an HTTP(S) /retrieve endpoint." >&2
  exit 1
fi
hardware_contract=eight-gpu
if [[ "${num_gpus}" != "8" ]]; then
  if [[ "${allow_nonformal_preflight}" != "1" ]] \
    || [[ "${num_gpus}" != "4" ]] \
    || [[ "${run_mode}" == campaign ]]; then
    echo "Aligned current-veRL runs require eight GPUs; only explicitly enabled four-GPU smoke and performance diagnostics are allowed." >&2
    exit 1
  fi
  hardware_contract="nonformal-four-gpu-${run_mode}"
fi
if [[ "${seed}" == *[!0-9]* || -z "${seed}" ]]; then
  echo "SEARCH_R1_SEED must be a non-negative integer." >&2
  exit 1
fi
if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" != 1 ]] && \
  { ! git -C "${nemo_root}" diff --quiet || ! git -C "${nemo_root}" diff --cached --quiet; }; then
  echo "NeMo comparison adapters have uncommitted tracked changes." >&2
  exit 1
fi
if [[ -e "${output_dir}/manifest.txt" ]]; then
  echo "SEARCH_R1_OUTPUT_DIR already contains a run manifest: ${output_dir}" >&2
  exit 1
fi

mkdir -p \
  "${output_dir}/checkpoints" \
  "${output_dir}/nsight" \
  "${output_dir}/nsight-tmp" \
  "${output_dir}/observability" \
  "${output_dir}/ray" \
  "${output_dir}/rollouts" \
  "${output_dir}/validation" \
  "${output_dir}/swanlab" \
  "${output_dir}/tensorboard"

export PYTHONPATH="${comparison_dir}:${verl_root}${PYTHONPATH:+:${PYTHONPATH}}"
export SEARCH_R1_RETRIEVER_URL="${retriever_url}"
unset AI_SEARCH_TRACE_PATH AI_SEARCH_TRACE_SAMPLE_RATE
unset SWANLAB_MODE SWANLAB_LOG_DIR TENSORBOARD_DIR
nsys_executable=disabled
nsys_version=disabled
profile_ray_tmpdir=disabled
if [[ "${observability_mode}" != baseline ]]; then
  export AI_SEARCH_TRACE_PATH="${output_dir}/trajectory-spans.jsonl"
  export AI_SEARCH_TRACE_SAMPLE_RATE="${trace_sample_rate}"
  export SWANLAB_MODE=local
  export SWANLAB_LOG_DIR="${output_dir}/swanlab"
  export TENSORBOARD_DIR="${output_dir}/tensorboard"
fi
if [[ "${observability_mode}" == profile ]]; then
  profile_ray_tmpdir="${SEARCH_R1_RAY_TMPDIR:-${output_dir}/ray}"
  if [[ "${profile_ray_tmpdir}" != /* ]]; then
    echo "SEARCH_R1_RAY_TMPDIR must be an absolute path." >&2
    exit 1
  fi
  if (( ${#profile_ray_tmpdir} > 32 )); then
    echo "SEARCH_R1_RAY_TMPDIR is too long for Ray Unix sockets: ${profile_ray_tmpdir}" >&2
    exit 1
  fi
  mkdir -p "${profile_ray_tmpdir}"
  nsys_executable="${SEARCH_R1_NSYS_BIN:-$(command -v nsys || true)}"
  if [[ -z "${nsys_executable}" || ! -x "${nsys_executable}" ]]; then
    if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" != 1 ]]; then
      echo "Profile mode requires an executable nsys; set SEARCH_R1_NSYS_BIN." >&2
      exit 1
    fi
    nsys_executable=required-at-runtime
    nsys_version=required-at-runtime
  else
    nsys_executable=$(readlink -f "${nsys_executable}")
    export PATH="$(dirname -- "${nsys_executable}"):${PATH}"
    nsys_version=$("${nsys_executable}" --version 2>&1 | tail -n 1)
  fi
  export NSYS_TMPDIR="${SEARCH_R1_NSYS_TMPDIR:-${output_dir}/nsight-tmp}"
  if [[ "${NSYS_TMPDIR}" != /* ]]; then
    echo "SEARCH_R1_NSYS_TMPDIR must be an absolute path." >&2
    exit 1
  fi
  mkdir -p "${NSYS_TMPDIR}"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export SEARCH_R1_WORKLOAD_MODE="${workload_mode}"
if [[ "${workload_mode}" == controlled ]]; then
  # Keep normal training on the fastest kernels.  The controlled comparison
  # enables only vLLM's batch-invariant generation path, not veRL's broader
  # full-determinism mode (which intentionally changes training kernels).
  export VLLM_BATCH_INVARIANT=1
else
  unset VLLM_BATCH_INVARIANT
fi

{
  printf 'framework=current-verl\n'
  printf 'run_mode=%s\nobservability_mode=%s\n' "${run_mode}" "${observability_mode}"
  printf 'iteration_mode=%s\nwarmup_steps=%s\nmeasured_steps=%s\n' \
    "${iteration_mode}" "${warmup_steps}" "${measured_steps}"
  printf 'workload_mode=%s\nprompt_encoding=%s\n' \
    "${workload_mode}" "${prompt_encoding}"
  printf 'rollout_request_temperature=%s\ntraining_logprob_temperature=1.0\n' \
    "${rollout_request_temperature}"
  printf 'weights_frozen_by_zero_lr=%s\n' "${weights_frozen}"
  printf 'rollout_batch_invariant=%s\ndeterministic_request_identity=%s\n' \
    "${rollout_batch_invariant}" "${deterministic_request_identity}"
  printf 'filter_overlong_prompts=%s\n' "${filter_overlong_prompts}"
  printf 'source_upstream_base=%s\nsource_patched_head=%s\n' \
    "${expected_verl_upstream_base}" "${actual_verl_head}"
  printf 'reference_patch_head=%s\nsource_patch_sha256=%s\n' \
    "${expected_verl_patched_head}" "${actual_verl_patch_sha256}"
  printf 'adapter_commit=%s\n' "$(git -C "${comparison_dir}" rev-parse HEAD)"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\n' "${expected_train_sha256}"
  printf 'eval_sha256=%s\n' "${expected_eval_sha256}"
  printf 'controlled_data_manifest_sha256=%s\n' \
    "$([[ "${workload_mode}" == controlled ]] && echo "${expected_controlled_manifest_sha256}" || echo disabled)"
  printf 'controlled_semantic_rows_sha256=%s\n' "${controlled_semantic_rows_sha256}"
  printf 'retriever_url=%s\n' "${retriever_url}"
  printf 'seed=%s\n' "${seed}"
  printf 'training_gpus=%s\n' "${num_gpus}"
  printf 'hardware_contract=%s\n' "${hardware_contract}"
  printf 'prompts_per_step=%s\n' "${prompts_per_step}"
  printf 'rollouts_per_prompt=5\n'
  printf 'trajectories_per_step=%s\n' "$((prompts_per_step * 5))"
  printf 'ppo_mini_batch_size=%s\n' "${ppo_mini_batch_size}"
  printf 'ppo_micro_batch_size_per_gpu=%s\n' "${ppo_micro_batch_size_per_gpu}"
  printf 'log_prob_micro_batch_size_per_gpu=%s\n' "${log_prob_micro_batch_size_per_gpu}"
  printf 'optimizer_updates_per_outer_step=%s\n' \
    "$((prompts_per_step / ppo_mini_batch_size))"
  printf 'total_steps=%s\n' "${total_steps}"
  printf 'optimizer=AdamW\noptimizer_lr=%s\noptimizer_weight_decay=0.01\n' \
    "${optimizer_lr}"
  printf 'optimizer_betas=0.9,0.999\noptimizer_epsilon=1e-8\n'
  printf 'measurement_work_counters=scalar-transfer-queue-tags\n'
  printf 'timed_rollout_text_dump=disabled\n'
  printf 'lr_schedule_horizon_outer_steps=%s\n' "${lr_schedule_horizon_outer_steps}"
  printf 'lr_warmup_outer_steps=%s\nlr_after_warmup=constant\n' "${lr_warmup_outer_steps}"
  printf 'trace_path=%s\n' "${AI_SEARCH_TRACE_PATH:-disabled}"
  printf 'trace_sample_rate=%s\n' "${AI_SEARCH_TRACE_SAMPLE_RATE:-disabled}"
  printf 'engine_prometheus_scope=%s\n' "${engine_prometheus_scope}"
  printf 'engine_prometheus_config=%s\n' \
    "$([[ "${engine_prometheus_enabled}" == true ]] && echo "${output_dir}/observability/rollout-prometheus.yml" || echo disabled)"
  printf 'swanlab_mode=%s\ntensorboard_dir=%s\n' "${SWANLAB_MODE:-disabled}" "${TENSORBOARD_DIR:-disabled}"
  printf 'nsys_executable=%s\nnsys_version=%s\n' "${nsys_executable}" "${nsys_version}"
  printf 'nsys_profile_step=%s\nray_tmpdir=%s\n' "$([[ "${observability_mode}" == profile ]] && echo 2 || echo disabled)" "${profile_ray_tmpdir}"
  printf 'nsys_scope=%s\n' "$([[ "${observability_mode}" == profile ]] && echo actor-and-reference-worker-processes-full-step || echo disabled)"
  printf 'nsys_rollout_engine_scope=%s\n' "$([[ "${observability_mode}" == profile ]] && echo missing || echo disabled)"
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign && "${observability_mode}" == clean ]] && echo candidate || echo false)"
} > "${output_dir}/manifest.txt"

command=(
  python3 -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false
  "data.train_files=['${train_file}']"
  "data.val_files=['${eval_file}']"
  "data.train_batch_size=${prompts_per_step}"
  data.max_prompt_length=2048
  data.max_response_length=4096
  "data.filter_overlong_prompts=${filter_overlong_prompts}"
  data.truncation=error
  data.shuffle=false
  "actor_rollout_ref.model.path=${model_path}"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  "actor_rollout_ref.actor.optim.lr=${optimizer_lr}"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_outer_steps}"
  actor_rollout_ref.actor.optim.weight_decay=0.01
  "actor_rollout_ref.actor.optim.betas=[0.9,0.999]"
  actor_rollout_ref.actor.optim.optimizer=AdamW
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim
  "actor_rollout_ref.actor.optim.override_optimizer_config={eps:1.0e-8}"
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}"
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.2
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  "actor_rollout_ref.rollout.disable_log_stats=${disable_log_stats}"
  "actor_rollout_ref.rollout.prometheus.enable=${engine_prometheus_enabled}"
  "actor_rollout_ref.rollout.prometheus.file=${output_dir}/observability/rollout-prometheus.yml"
  actor_rollout_ref.rollout.prometheus.served_model_name=Qwen2.5-7B
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=4096
  actor_rollout_ref.rollout.max_model_len=6144
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  "actor_rollout_ref.rollout.seed=${seed}"
  actor_rollout_ref.rollout.n=5
  actor_rollout_ref.rollout.calculate_log_probs=true
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}"
  "actor_rollout_ref.rollout.agent.num_workers=${agent_workers}"
  actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_text
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${comparison_dir}/adapters/current-verl-search-r1-agent-loop.example.yaml"
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=false
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu}"
  actor_rollout_ref.ref.fsdp_config.param_offload=false
  "reward.custom_reward_function.path=${comparison_dir}/framework_eval_adapters.py"
  reward.custom_reward_function.name=verl_compute_score
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.balance_batch=false
  "trainer.logger=${logger}"
  trainer.project_name=search-r1-four-way
  "trainer.experiment_name=current-verl-${run_mode}-${observability_mode}-seed-${seed}"
  trainer.nnodes=1
  "trainer.n_gpus_per_node=${num_gpus}"
  "trainer.total_training_steps=${total_steps}"
  trainer.total_epochs=15
  "trainer.save_freq=${save_freq}"
  "trainer.test_freq=${test_freq}"
  "trainer.val_before_train=${val_before_train}"
  trainer.val_only=false
  trainer.resume_mode=disable
  "trainer.default_local_dir=${output_dir}/checkpoints"
  trainer.rollout_data_dir=null
  "trainer.validation_data_dir=${output_dir}/validation"
)

if [[ "${observability_mode}" == profile ]]; then
  command+=(
    global_profiler.tool=nsys
    "global_profiler.steps=[2]"
    global_profiler.profile_continuous_steps=false
    "global_profiler.save_path=${output_dir}/nsight"
    global_profiler.global_tool_config.nsys.discrete=false
    "global_profiler.global_tool_config.nsys.controller_nsight_options.trace='cuda,nvtx,cublas'"
    +global_profiler.global_tool_config.nsys.controller_nsight_options.sample=none
    +global_profiler.global_tool_config.nsys.controller_nsight_options.cpuctxsw=none
    "global_profiler.global_tool_config.nsys.worker_nsight_options.trace='cuda,nvtx,cublas'"
    +global_profiler.global_tool_config.nsys.worker_nsight_options.sample=none
    +global_profiler.global_tool_config.nsys.worker_nsight_options.cpuctxsw=none
    actor_rollout_ref.actor.profiler.enable=true
    actor_rollout_ref.actor.profiler.all_ranks=true
    actor_rollout_ref.ref.profiler.enable=true
    actor_rollout_ref.ref.profiler.all_ranks=true
    "+ray_kwargs.ray_init._temp_dir=${profile_ray_tmpdir}"
  )
fi

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
