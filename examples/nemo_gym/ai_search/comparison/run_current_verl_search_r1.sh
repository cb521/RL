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
lr_schedule_horizon_outer_steps=500
lr_warmup_outer_steps=142

expected_verl_upstream_base=5cfb74fa04c7f6e5d98260b8f05157c6a9402695
expected_verl_patched_head=fb72e8b195095ac3334e870176eb6eaa80184001
expected_verl_patch_sha256=3a11219896823a85e7f2527351b9071b118e87564b3a359f2b4c26f1257980c4
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39
expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    total_steps=1
    ppo_mini_batch_size=40
    ppo_micro_batch_size_per_gpu=1
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
    ppo_mini_batch_size=40
    ppo_micro_batch_size_per_gpu=1
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
    save_freq=100
    test_freq=50
    val_before_train=true
    default_trace_sample_rate=0.01
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

case "${observability_mode}" in
  baseline)
    logger="['console']"
    trace_sample_rate=disabled
    ;;
  clean)
    logger="['console','swanlab','tensorboard']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-${default_trace_sample_rate}}"
    ;;
  profile)
    logger="['console','swanlab','tensorboard']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"
    ;;
  *)
    echo "SEARCH_R1_OBSERVABILITY_MODE must be baseline, clean, or profile." >&2
    exit 1
    ;;
esac
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
if [[ "${retriever_url}" != http://*/retrieve && "${retriever_url}" != https://*/retrieve ]]; then
  echo "SEARCH_R1_RETRIEVER_URL must be an HTTP(S) /retrieve endpoint." >&2
  exit 1
fi
if [[ "${num_gpus}" != "8" ]]; then
  echo "Aligned current-veRL runs require exactly eight training GPUs." >&2
  exit 1
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
if [[ "${observability_mode}" != baseline ]]; then
  export AI_SEARCH_TRACE_PATH="${output_dir}/trajectory-spans.jsonl"
  export AI_SEARCH_TRACE_SAMPLE_RATE="${trace_sample_rate}"
  export SWANLAB_MODE=local
  export SWANLAB_LOG_DIR="${output_dir}/swanlab"
  export TENSORBOARD_DIR="${output_dir}/tensorboard"
fi
if [[ "${observability_mode}" == profile ]]; then
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

{
  printf 'framework=current-verl\n'
  printf 'run_mode=%s\nobservability_mode=%s\n' "${run_mode}" "${observability_mode}"
  printf 'source_upstream_base=%s\nsource_patched_head=%s\n' \
    "${expected_verl_upstream_base}" "${actual_verl_head}"
  printf 'reference_patch_head=%s\nsource_patch_sha256=%s\n' \
    "${expected_verl_patched_head}" "${actual_verl_patch_sha256}"
  printf 'adapter_commit=%s\n' "$(git -C "${comparison_dir}" rev-parse HEAD)"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\n' "${expected_train_sha256}"
  printf 'eval_sha256=%s\n' "${expected_eval_sha256}"
  printf 'retriever_url=%s\n' "${retriever_url}"
  printf 'seed=%s\n' "${seed}"
  printf 'training_gpus=%s\n' "${num_gpus}"
  printf 'prompts_per_step=%s\n' "${prompts_per_step}"
  printf 'rollouts_per_prompt=5\n'
  printf 'trajectories_per_step=%s\n' "$((prompts_per_step * 5))"
  printf 'ppo_mini_batch_size=%s\n' "${ppo_mini_batch_size}"
  printf 'ppo_micro_batch_size_per_gpu=%s\n' "${ppo_micro_batch_size_per_gpu}"
  printf 'log_prob_micro_batch_size_per_gpu=%s\n' "${log_prob_micro_batch_size_per_gpu}"
  printf 'optimizer_updates_per_outer_step=%s\n' \
    "$(((prompts_per_step * 5) / ppo_mini_batch_size))"
  printf 'total_steps=%s\n' "${total_steps}"
  printf 'optimizer=AdamW\noptimizer_lr=1e-6\noptimizer_weight_decay=0.01\n'
  printf 'optimizer_betas=0.9,0.999\noptimizer_epsilon=1e-8\n'
  printf 'measurement_work_counters=scalar-transfer-queue-tags\n'
  printf 'timed_rollout_text_dump=disabled\n'
  printf 'lr_schedule_horizon_outer_steps=%s\n' "${lr_schedule_horizon_outer_steps}"
  printf 'lr_warmup_outer_steps=%s\nlr_after_warmup=constant\n' "${lr_warmup_outer_steps}"
  printf 'trace_path=%s\n' "${AI_SEARCH_TRACE_PATH:-disabled}"
  printf 'trace_sample_rate=%s\n' "${AI_SEARCH_TRACE_SAMPLE_RATE:-disabled}"
  printf 'swanlab_mode=%s\ntensorboard_dir=%s\n' "${SWANLAB_MODE:-disabled}" "${TENSORBOARD_DIR:-disabled}"
  printf 'nsys_executable=%s\nnsys_version=%s\n' "${nsys_executable}" "${nsys_version}"
  printf 'nsys_profile_step=%s\nray_tmpdir=%s\n' "$([[ "${observability_mode}" == profile ]] && echo 2 || echo disabled)" "$([[ "${observability_mode}" == profile ]] && echo "${output_dir}/ray" || echo disabled)"
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
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle=false
  "actor_rollout_ref.model.path=${model_path}"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
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
    actor_rollout_ref.actor.profiler.enable=true
    actor_rollout_ref.actor.profiler.all_ranks=true
    actor_rollout_ref.ref.profiler.enable=true
    actor_rollout_ref.ref.profiler.all_ranks=true
    "+ray_kwargs.ray_init._temp_dir=${output_dir}/ray"
  )
fi

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
