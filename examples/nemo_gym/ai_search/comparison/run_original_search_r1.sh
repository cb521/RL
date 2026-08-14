#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nemo_root=$(cd -- "${comparison_dir}/../../../.." && pwd)
search_r1_root="${ORIGINAL_SEARCH_R1_ROOT:?ORIGINAL_SEARCH_R1_ROOT must point to the frozen patched Search-R1 checkout}"
original_python="${ORIGINAL_SEARCH_R1_PYTHON:-python3}"
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

expected_upstream_base=598e61bd1d36895726d28a8d06b3a15bed19f5d3
expected_patched_head=8f4c5b91e4092fb4d8e858cee0689d339aaf310f
expected_patch_sha256=e8c0e873ccdc2c3220211ca1425de099cb2d7505f902e7658b161e60fa130bf3
expected_runtime_lock_sha256=11a7246f36c9ea844e14b030631d8f8b1489cd245f663a5864bc8b3074d5f269
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39
expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461
lr_warmup_outer_steps=142

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    total_steps=1
    ppo_mini_batch_size=40
    ppo_micro_batch_size=40
    log_prob_micro_batch_size=8
    val_batch_size=8
    save_freq=-1
    test_freq=-1
    val_before_train=false
    val_at_end=false
    default_trace_sample_rate=1.0
    ;;
  performance)
    prompts_per_step=8
    total_steps=4
    ppo_mini_batch_size=40
    ppo_micro_batch_size=40
    log_prob_micro_batch_size=8
    val_batch_size=8
    save_freq=-1
    test_freq=-1
    val_before_train=false
    val_at_end=false
    default_trace_sample_rate=0.1
    ;;
  campaign)
    prompts_per_step=512
    total_steps=500
    ppo_mini_batch_size=256
    ppo_micro_batch_size=64
    log_prob_micro_batch_size=128
    val_batch_size=256
    save_freq=250
    test_freq=50
    val_before_train=true
    val_at_end=true
    default_trace_sample_rate=0.01
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

case "${observability_mode}" in
  baseline)
    trainer_logger="['console']"
    trace_sample_rate=disabled
    prometheus_scope=disabled
    engine_prometheus_scope=disabled
    ;;
  clean)
    trainer_logger="['console','local']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-${default_trace_sample_rate}}"
    prometheus_scope=driver-step-metrics
    engine_prometheus_scope=missing-embedded-old-vllm
    ;;
  profile)
    trainer_logger="['console','local']"
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"
    prometheus_scope=driver-step-metrics
    engine_prometheus_scope=missing-embedded-old-vllm
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
prometheus_port="${SEARCH_R1_PROMETHEUS_PORT:-9108}"
runtime_lock="${comparison_dir}/adapters/original-search-r1-runtime.lock"
# The old trainer starts at global step 1 and stops after incrementing it.
trainer_stop_step=$((total_steps + 1))

if (( $# != 0 )); then
  echo "The strict original Search-R1 launcher does not accept positional overrides." >&2
  exit 1
fi

actual_patched_head="$(git -C "${search_r1_root}" rev-parse HEAD)"
if ! git -C "${search_r1_root}" merge-base --is-ancestor \
  "${expected_upstream_base}" "${actual_patched_head}"; then
  echo "Search-R1 patch head does not descend from ${expected_upstream_base}." >&2
  exit 1
fi
actual_patch_sha256="$({
  git -C "${search_r1_root}" diff --full-index --binary \
    "${expected_upstream_base}..${actual_patched_head}"
} | sha256sum | cut -d ' ' -f 1)"
if [[ "${actual_patch_sha256}" != "${expected_patch_sha256}" ]]; then
  echo "Search-R1 comparison diff does not match ${expected_patch_sha256}." >&2
  exit 1
fi
if ! git -C "${search_r1_root}" diff --quiet || \
  ! git -C "${search_r1_root}" diff --cached --quiet; then
  echo "Patched Search-R1 checkout has uncommitted tracked changes." >&2
  exit 1
fi
if [[ "$(basename -- "$(readlink -f "${model_path}")")" != "${expected_model_revision}" ]]; then
  echo "SEARCH_R1_MODEL_PATH does not resolve to revision ${expected_model_revision}." >&2
  exit 1
fi
for required_file in \
  "${train_file}" \
  "${eval_file}" \
  "${search_r1_root}/verl/trainer/main_ppo.py" \
  "${runtime_lock}" \
  "${comparison_dir}/prepare_original_search_r1_runtime.sh" \
  "${comparison_dir}/original_search_r1_export.py"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing aligned original Search-R1 artifact: ${required_file}" >&2
    exit 1
  fi
done
printf '%s  %s\n' "${expected_train_sha256}" "${train_file}" | sha256sum --check --status
printf '%s  %s\n' "${expected_eval_sha256}" "${eval_file}" | sha256sum --check --status
printf '%s  %s\n' "${expected_runtime_lock_sha256}" "${runtime_lock}" | sha256sum --check --status
if [[ "${retriever_url}" != http://*/retrieve && "${retriever_url}" != https://*/retrieve ]]; then
  echo "SEARCH_R1_RETRIEVER_URL must be an HTTP(S) /retrieve endpoint." >&2
  exit 1
fi
hardware_contract=eight-gpu
if [[ "${num_gpus}" != "8" ]]; then
  if [[ "${allow_nonformal_preflight}" != "1" ]] \
    || [[ "${num_gpus}" != "4" ]] \
    || [[ "${run_mode}" == campaign ]]; then
    echo "Aligned original Search-R1 runs require eight GPUs; only explicitly enabled four-GPU smoke and performance diagnostics are allowed." >&2
    exit 1
  fi
  hardware_contract="nonformal-four-gpu-${run_mode}"
fi
if [[ "${seed}" == *[!0-9]* || -z "${seed}" ]]; then
  echo "SEARCH_R1_SEED must be a non-negative integer." >&2
  exit 1
fi
if [[ "${prometheus_port}" == *[!0-9]* || -z "${prometheus_port}" ]] || \
  (( prometheus_port < 1 || prometheus_port > 65535 )); then
  echo "SEARCH_R1_PROMETHEUS_PORT must be an integer in [1, 65535]." >&2
  exit 1
fi
runtime_python_path=$(command -v "${original_python}" || true)
if [[ -z "${runtime_python_path}" || ! -x "${runtime_python_path}" ]]; then
  echo "Cannot execute ORIGINAL_SEARCH_R1_PYTHON=${original_python}." >&2
  exit 1
fi
if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" != 1 ]]; then
  runtime_root=$(cd -- "$(dirname -- "${runtime_python_path}")/.." && pwd)
  runtime_marker="${runtime_root}/.original-search-r1-runtime"
  if [[ ! -f "${runtime_marker}" ]]; then
    echo "Prepare ORIGINAL_SEARCH_R1_PYTHON with prepare_original_search_r1_runtime.sh first." >&2
    exit 1
  fi
  ORIGINAL_SEARCH_R1_VENV="${runtime_root}" \
  ORIGINAL_SEARCH_R1_REQUIRE_GPU=1 \
    bash "${comparison_dir}/prepare_original_search_r1_runtime.sh"
  visible_gpus=$("${runtime_python_path}" -c 'import torch; print(torch.cuda.device_count())')
  if [[ "${visible_gpus}" != "${num_gpus}" ]]; then
    echo "Expected ${num_gpus} visible training GPUs, found ${visible_gpus}." >&2
    exit 1
  fi
  if ! git -C "${nemo_root}" diff --quiet || ! git -C "${nemo_root}" diff --cached --quiet; then
    echo "NeMo comparison adapters have uncommitted tracked changes." >&2
    exit 1
  fi
fi
if [[ -e "${output_dir}/manifest.txt" ]]; then
  echo "SEARCH_R1_OUTPUT_DIR already contains a run manifest: ${output_dir}" >&2
  exit 1
fi

mkdir -p \
  "${output_dir}/checkpoints" \
  "${output_dir}/nsight-tmp" \
  "${output_dir}/ray" \
  "${output_dir}/tensorboard" \
  "${output_dir}/validation" \
  "${output_dir}/observability"

export PYTHONPATH="${comparison_dir}:${search_r1_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname -- "${runtime_python_path}"):${PATH}"
unset AI_SEARCH_TRACE_PATH AI_SEARCH_TRACE_SAMPLE_RATE AI_SEARCH_METRICS_PATH
unset AI_SEARCH_TENSORBOARD_DIR AI_SEARCH_PROMETHEUS_PORT
unset SEARCH_R1_NSYS_PROFILE_STEP SEARCH_R1_NSYS_OUTPUT_PREFIX
nsys_executable=disabled
nsys_version=disabled
if [[ "${observability_mode}" != baseline ]]; then
  export AI_SEARCH_TRACE_PATH="${output_dir}/observability/trajectory-spans.jsonl"
  export AI_SEARCH_TRACE_SAMPLE_RATE="${trace_sample_rate}"
  export AI_SEARCH_METRICS_PATH="${output_dir}/observability/step-metrics.jsonl"
  export AI_SEARCH_TENSORBOARD_DIR="${output_dir}/tensorboard"
  export AI_SEARCH_PROMETHEUS_PORT="${prometheus_port}"
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
  export SEARCH_R1_NSYS_PROFILE_STEP=2
  export SEARCH_R1_NSYS_OUTPUT_PREFIX=original_search_r1_worker_%p
  export SEARCH_R1_RAY_TMPDIR="${SEARCH_R1_RAY_TMPDIR:-${output_dir}/ray}"
  if [[ "${SEARCH_R1_RAY_TMPDIR}" != /* ]]; then
    echo "SEARCH_R1_RAY_TMPDIR must be an absolute path." >&2
    exit 1
  fi
  if (( ${#SEARCH_R1_RAY_TMPDIR} > 32 )); then
    echo "SEARCH_R1_RAY_TMPDIR is too long for Ray Unix sockets: ${SEARCH_R1_RAY_TMPDIR}" >&2
    exit 1
  fi
  mkdir -p "${SEARCH_R1_RAY_TMPDIR}"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

{
  printf 'framework=original-search-r1\n'
  printf 'run_mode=%s\nobservability_mode=%s\n' "${run_mode}" "${observability_mode}"
  printf 'source_upstream_base=%s\n' "${expected_upstream_base}"
  printf 'source_patched_head=%s\n' "${actual_patched_head}"
  printf 'reference_patch_head=%s\n' "${expected_patched_head}"
  printf 'source_patch_sha256=%s\n' "${actual_patch_sha256}"
  printf 'runtime_lock_sha256=%s\nruntime_python=%s\n' "${expected_runtime_lock_sha256}" "${runtime_python_path}"
  printf 'adapter_commit=%s\n' "$(git -C "${comparison_dir}" rev-parse HEAD)"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\neval_sha256=%s\n' "${expected_train_sha256}" "${expected_eval_sha256}"
  printf 'retriever_url=%s\nseed=%s\ntraining_gpus=%s\n' "${retriever_url}" "${seed}" "${num_gpus}"
  printf 'hardware_contract=%s\n' "${hardware_contract}"
  printf 'prompts_per_step=%s\nrollouts_per_prompt=5\n' "${prompts_per_step}"
  printf 'trajectories_per_step=%s\ntotal_steps=%s\n' "$((prompts_per_step * 5))" "${total_steps}"
  printf 'trainer_stop_step=%s\nppo_mini_batch_size=%s\n' "${trainer_stop_step}" "${ppo_mini_batch_size}"
  printf 'ppo_micro_batch_size=%s\nlog_prob_micro_batch_size=%s\n' "${ppo_micro_batch_size}" "${log_prob_micro_batch_size}"
  printf 'effective_ppo_micro_batch_size_per_gpu=%s\n' \
    "$((ppo_micro_batch_size / num_gpus))"
  printf 'optimizer=AdamW\noptimizer_lr=1e-6\noptimizer_weight_decay=0.01\n'
  printf 'optimizer_betas=0.9,0.999\noptimizer_epsilon=1e-8\n'
  printf 'lr_warmup_outer_steps=%s\nlr_after_warmup=constant\n' "${lr_warmup_outer_steps}"
  printf 'trace_path=%s\ntrace_sample_rate=%s\n' "${AI_SEARCH_TRACE_PATH:-disabled}" "${AI_SEARCH_TRACE_SAMPLE_RATE:-disabled}"
  printf 'metrics_path=%s\ntensorboard_dir=%s\n' "${AI_SEARCH_METRICS_PATH:-disabled}" "${AI_SEARCH_TENSORBOARD_DIR:-disabled}"
  printf 'prometheus_url=%s\n' "$([[ -n "${AI_SEARCH_PROMETHEUS_PORT:-}" ]] && printf 'http://127.0.0.1:%s/metrics' "${AI_SEARCH_PROMETHEUS_PORT}" || echo disabled)"
  printf 'prometheus_scope=%s\nengine_prometheus_scope=%s\n' \
    "${prometheus_scope}" "${engine_prometheus_scope}"
  printf 'nsys_executable=%s\nnsys_version=%s\n' "${nsys_executable}" "${nsys_version}"
  printf 'nsys_profile_step=%s\nray_tmpdir=%s\n' "${SEARCH_R1_NSYS_PROFILE_STEP:-disabled}" "${SEARCH_R1_RAY_TMPDIR:-disabled}"
  printf 'nsys_scope=%s\n' "$([[ "${observability_mode}" == profile ]] && echo actor-rollout-and-reference-worker-processes-full-step || echo disabled)"
  printf 'nsys_rollout_engine_scope=%s\n' "$([[ "${observability_mode}" == profile ]] && echo covered || echo disabled)"
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign && "${observability_mode}" == clean ]] && echo candidate || echo false)"
} > "${output_dir}/manifest.txt"

command=(
  "${runtime_python_path}" -m verl.trainer.main_ppo
  "data.train_files=${train_file}"
  "data.val_files=${eval_file}"
  data.train_data_num=null
  data.val_data_num=null
  "data.train_batch_size=${prompts_per_step}"
  "data.val_batch_size=${val_batch_size}"
  data.max_prompt_length=4096
  data.max_response_length=500
  data.max_start_length=2048
  data.max_obs_length=500
  data.shuffle_train_dataloader=false
  algorithm.adv_estimator=grpo
  algorithm.no_think_rl=false
  "actor_rollout_ref.model.path=${model_path}"
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  "+actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_outer_steps}"
  "+actor_rollout_ref.actor.optim.weight_decay=0.01"
  "+actor_rollout_ref.actor.optim.betas=[0.9,0.999]"
  actor_rollout_ref.actor.use_kl_loss=true
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size}"
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.grad_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.state_masking=true
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.n_agent=5
  "actor_rollout_ref.rollout.log_prob_micro_batch_size=${log_prob_micro_batch_size}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size=${log_prob_micro_batch_size}"
  actor_rollout_ref.ref.fsdp_config.param_offload=false
  "trainer.logger=${trainer_logger}"
  +trainer.val_only=false
  "+trainer.val_before_train=${val_before_train}"
  "+trainer.val_at_end=${val_at_end}"
  "+trainer.validation_predictions_path=${output_dir}/validation/predictions.jsonl"
  trainer.nnodes=1
  "trainer.n_gpus_per_node=${num_gpus}"
  "trainer.save_freq=${save_freq}"
  "trainer.test_freq=${test_freq}"
  trainer.project_name=search-r1-four-way
  "trainer.experiment_name=original-search-r1-${run_mode}-${observability_mode}-seed-${seed}"
  trainer.total_epochs=15
  "trainer.total_training_steps=${trainer_stop_step}"
  trainer.default_hdfs_dir=null
  "trainer.default_local_dir=${output_dir}/checkpoints"
  max_turns=4
  "retriever.url=${retriever_url}"
  retriever.topk=3
)

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "${search_r1_root}"
exec "${command[@]}"
