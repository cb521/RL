#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plugin_dir=$(cd -- "${comparison_dir}/.." && pwd)
nemo_root=$(cd -- "${comparison_dir}/../../../.." && pwd)
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
train_file="${SEARCH_R1_NEMO_TRAIN_FILE:?SEARCH_R1_NEMO_TRAIN_FILE must point to the composite train JSONL}"
eval_file="${SEARCH_R1_NEMO_EVAL_FILE:?SEARCH_R1_NEMO_EVAL_FILE must point to the composite evaluation JSONL}"
retriever_url="${SEARCH_R1_RETRIEVER_URL:?SEARCH_R1_RETRIEVER_URL must point to the shared E5 /retrieve endpoint}"
output_dir="${SEARCH_R1_OUTPUT_DIR:?SEARCH_R1_OUTPUT_DIR must be set}"
run_mode="${SEARCH_R1_RUN_MODE:-smoke}"
observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"
seed="${SEARCH_R1_SEED:-42}"
num_gpus="${SEARCH_R1_NUM_GPUS:-8}"
allow_nonformal_preflight="${SEARCH_R1_ALLOW_NONFORMAL_PREFLIGHT:-0}"
fast_iteration="${SEARCH_R1_FAST_ITERATION:-0}"
workload_mode="${SEARCH_R1_WORKLOAD_MODE:-training}"
runtime_profile="${SEARCH_R1_NEMO_RUNTIME_PROFILE:-discard}"
controlled_data_manifest="${SEARCH_R1_CONTROLLED_DATA_MANIFEST:-}"

expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=9904042da053be8e7fa275453c9221324d24aadb6f67323d040e6016da9bfaff
expected_eval_sha256=bdcc57b4c3e88241bf7144f4e739c991c7a0ace4cd1ea26b6c602e2655445645
expected_controlled_train_sha256=2b14840f263d12cda7005000775c8b39ffc397312d2f0f174f85123b789d7e21
expected_controlled_manifest_sha256=09b4fa8a7127873c9083ff7dfa736ff087ad2c0d6b570dd6349d7bbfe6db1003
controlled_semantic_rows_sha256=disabled
lr_warmup_outer_steps=142

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    total_steps=1
    train_global_batch_size=40
    train_micro_batch_size=5
    logprob_batch_size=1
    val_period=1000000
    val_at_start=false
    val_at_end=false
    checkpoint_enabled=false
    default_trace_sample_rate=1.0
    ;;
  performance)
    prompts_per_step=8
    total_steps=4
    if [[ "${fast_iteration}" == "1" ]]; then
      total_steps=2
    fi
    train_global_batch_size=40
    train_micro_batch_size=5
    logprob_batch_size=1
    val_period=1000000
    val_at_start=false
    val_at_end=false
    checkpoint_enabled=false
    default_trace_sample_rate=0.1
    ;;
  campaign)
    prompts_per_step=512
    total_steps=500
    train_global_batch_size=256
    train_micro_batch_size=8
    logprob_batch_size=16
    val_period=50
    val_at_start=true
    val_at_end=true
    checkpoint_enabled=true
    default_trace_sample_rate=0.01
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

case "${workload_mode}" in
  training)
    optimizer_lr=1e-6
    rollout_request_temperature=1.0
    prompt_encoding=search-r1-content-concat
    weights_frozen=false
    rollout_batch_invariant=false
    ;;
  controlled)
    if [[ "${run_mode}" != performance || "${observability_mode}" != clean ]]; then
      echo "SEARCH_R1_WORKLOAD_MODE=controlled requires a clean performance run." >&2
      exit 1
    fi
    optimizer_lr=0.0
    rollout_request_temperature=0.0
    prompt_encoding=search-r1-content-concat
    weights_frozen=true
    rollout_batch_invariant=true
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

case "${runtime_profile}" in
  discard)
    # Match the current veRL comparison configuration: discard stale rollout
    # weights and KV cache because the next wake is followed by a full refit.
    vllm_sleep_level=2
    vllm_gpu_memory_utilization=config-default
    ;;
  offload)
    # Preserve rollout weights on CPU while releasing their GPU allocation.
    vllm_sleep_level=1
    vllm_gpu_memory_utilization=config-default
    ;;
  resident)
    if [[ "${run_mode}" != performance || "${num_gpus}" != "4" ]]; then
      echo "SEARCH_R1_NEMO_RUNTIME_PROFILE=resident is restricted to four-GPU performance diagnostics." >&2
      exit 1
    fi
    # Keep enough KV capacity for ten 4K-context trajectories per engine while
    # leaving more training headroom than the initial 15% screening candidate.
    vllm_sleep_level=0
    vllm_gpu_memory_utilization=0.13
    ;;
  *)
    echo "SEARCH_R1_NEMO_RUNTIME_PROFILE must be discard, offload, or resident, not ${runtime_profile}." >&2
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

case "${observability_mode}" in
  baseline)
    trace_sample_rate=disabled
    ;;
  clean)
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-${default_trace_sample_rate}}"
    ;;
  profile)
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"
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
  echo "The strict NeMo four-way launcher does not accept positional overrides." >&2
  exit 1
fi
if [[ "$(basename -- "$(readlink -f "${model_path}")")" != "${expected_model_revision}" ]]; then
  echo "SEARCH_R1_MODEL_PATH does not resolve to revision ${expected_model_revision}." >&2
  exit 1
fi
for required_file in \
  "${train_file}" \
  "${eval_file}" \
  "${plugin_dir}/grpo_qwen2_5_7b_search_r1_four_way.yaml" \
  "${plugin_dir}/run_ai_search_observed.sh"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing aligned NeMo Search-R1 artifact: ${required_file}" >&2
    exit 1
  fi
done
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
    echo "Aligned NeMo runs require eight GPUs; only explicitly enabled four-GPU smoke and performance diagnostics are allowed." >&2
    exit 1
  fi
  hardware_contract="nonformal-four-gpu-${run_mode}"
fi
if [[ "${seed}" == *[!0-9]* || -z "${seed}" ]]; then
  echo "SEARCH_R1_SEED must be a non-negative integer." >&2
  exit 1
fi
if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" != "1" ]] && \
  { ! git -C "${nemo_root}" diff --quiet || ! git -C "${nemo_root}" diff --cached --quiet; }; then
  echo "NeMo RL has uncommitted tracked changes; formal runs require a clean source tree." >&2
  exit 1
fi
if [[ -e "${output_dir}/manifest.txt" ]]; then
  echo "SEARCH_R1_OUTPUT_DIR already contains a run manifest: ${output_dir}" >&2
  exit 1
fi

mkdir -p \
  "${output_dir}/checkpoints" \
  "${output_dir}/logs" \
  "${output_dir}/observability" \
  "${output_dir}/validation"

export AI_SEARCH_CONFIG="${plugin_dir}/grpo_qwen2_5_7b_search_r1_four_way.yaml"
export AI_SEARCH_MAX_STEPS="${total_steps}"
export AI_SEARCH_NUM_PROMPTS="${prompts_per_step}"
export AI_SEARCH_NUM_GENERATIONS=5
export AI_SEARCH_RUN_DIR="${output_dir}"
export AI_SEARCH_OBSERVABILITY_DIR="${output_dir}/observability"
export AI_SEARCH_OBSERVABILITY_MODE="${observability_mode}"
if [[ "${trace_sample_rate}" == disabled ]]; then
  unset AI_SEARCH_TRACE_SAMPLE_RATE
else
  export AI_SEARCH_TRACE_SAMPLE_RATE="${trace_sample_rate}"
fi
export AI_SEARCH_RETRIEVER_URL="${retriever_url}"
export SEARCH_R1_WORKLOAD_MODE="${workload_mode}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_RL_RAY_DASHBOARD=0
export TOKENIZERS_PARALLELISM=false
if [[ "${workload_mode}" == controlled ]]; then
  export VLLM_BATCH_INVARIANT=1
else
  unset VLLM_BATCH_INVARIANT
fi

nemo_commit=$(git -C "${nemo_root}" rev-parse HEAD)
nsys_profile_step_range=disabled
nsys_scope=disabled
nsys_rollout_engine_scope=disabled
if [[ "${observability_mode}" == profile ]]; then
  nsys_profile_step_range="${NRL_NSYS_PROFILE_STEP_RANGE:-2:3}"
  nsys_scope="${NRL_NSYS_WORKER_PATTERNS:-policy-and-vllm-worker-processes-step-2}"
  if [[ -z "${NRL_NSYS_WORKER_PATTERNS:-}" \
    || "${NRL_NSYS_WORKER_PATTERNS}" == *vllm* ]]; then
    nsys_rollout_engine_scope=covered
  else
    nsys_rollout_engine_scope=not-covered-targeted-worker-profile
  fi
fi
{
  printf 'framework=nemo-rl\n'
  printf 'run_mode=%s\nobservability_mode=%s\n' "${run_mode}" "${observability_mode}"
  printf 'iteration_mode=%s\nwarmup_steps=%s\nmeasured_steps=%s\n' \
    "${iteration_mode}" "${warmup_steps}" "${measured_steps}"
  printf 'workload_mode=%s\nprompt_encoding=%s\n' \
    "${workload_mode}" "${prompt_encoding}"
  printf 'rollout_request_temperature=%s\ntraining_logprob_temperature=1.0\n' \
    "${rollout_request_temperature}"
  printf 'weights_frozen_by_zero_lr=%s\n' "${weights_frozen}"
  printf 'rollout_batch_invariant=%s\n' "${rollout_batch_invariant}"
  printf 'nemo_runtime_profile=%s\n' "${runtime_profile}"
  printf 'vllm_sleep_level=%s\n' "${vllm_sleep_level}"
  printf 'vllm_gpu_memory_utilization=%s\n' "${vllm_gpu_memory_utilization}"
  printf 'vllm_max_num_batched_tokens=8192\n'
  printf 'source_commit=%s\n' "${nemo_commit}"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\neval_sha256=%s\n' "${expected_train_sha256}" "${expected_eval_sha256}"
  printf 'controlled_data_manifest_sha256=%s\n' \
    "$([[ "${workload_mode}" == controlled ]] && echo "${expected_controlled_manifest_sha256}" || echo disabled)"
  printf 'controlled_semantic_rows_sha256=%s\n' "${controlled_semantic_rows_sha256}"
  printf 'retriever_url=%s\nseed=%s\ntraining_gpus=%s\n' "${retriever_url}" "${seed}" "${num_gpus}"
  printf 'hardware_contract=%s\n' "${hardware_contract}"
  printf 'prompts_per_step=%s\nrollouts_per_prompt=5\n' "${prompts_per_step}"
  printf 'trajectories_per_step=%s\ntotal_steps=%s\n' "$((prompts_per_step * 5))" "${total_steps}"
  printf 'train_global_batch_size=%s\ntrain_micro_batch_size_per_gpu=%s\n' "${train_global_batch_size}" "${train_micro_batch_size}"
  printf 'logprob_batch_size_per_gpu=%s\n' "${logprob_batch_size}"
  printf 'optimizer_updates_per_outer_step=%s\n' "$(((prompts_per_step * 5) / train_global_batch_size))"
  printf 'optimizer=AdamW\noptimizer_lr=%s\noptimizer_weight_decay=0.01\n' \
    "${optimizer_lr}"
  printf 'optimizer_betas=0.9,0.999\noptimizer_epsilon=1e-8\n'
  printf 'lr_warmup_outer_steps=%s\nlr_after_warmup=constant\n' "${lr_warmup_outer_steps}"
  printf 'training_shuffle=false\nvalidation_sampling=greedy\n'
  printf 'validation_period=%s\nvalidation_at_start=%s\nvalidation_at_end=%s\n' \
    "${val_period}" "${val_at_start}" "${val_at_end}"
  printf 'checkpoint_enabled=%s\ncheckpoint_period=100\n' "${checkpoint_enabled}"
  printf 'trace_sample_rate=%s\n' "${trace_sample_rate}"
  printf 'engine_prometheus_scope=%s\n' \
    "$([[ "${observability_mode}" == baseline ]] && echo disabled || echo vllm-http-servers)"
  printf 'nsys_profile_step_range=%s\n' "${nsys_profile_step_range}"
  printf 'nsys_scope=%s\n' "${nsys_scope}"
  printf 'nsys_rollout_engine_scope=%s\n' "${nsys_rollout_engine_scope}"
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign && "${observability_mode}" == clean ]] && echo candidate || echo false)"
} > "${output_dir}/manifest.txt"

command=(
  bash "${plugin_dir}/run_ai_search_observed.sh"
  "grpo.seed=${seed}"
  data.shuffle=false
  "data.train.data_path=${train_file}"
  "data.validation.data_path=${eval_file}"
  "policy.model_name=${model_path}"
  "policy.tokenizer.name=${model_path}"
  "policy.train_global_batch_size=${train_global_batch_size}"
  "policy.train_micro_batch_size=${train_micro_batch_size}"
  "policy.logprob_batch_size=${logprob_batch_size}"
  "policy.optimizer.kwargs.lr=${optimizer_lr}"
  policy.generation.temperature=1.0
  policy.generation.top_p=1.0
  policy.generation.val_temperature=0.0
  policy.generation.val_top_p=1.0
  "grpo.val_period=${val_period}"
  "grpo.val_at_start=${val_at_start}"
  "grpo.val_at_end=${val_at_end}"
  grpo.val_num_generations_per_prompt=1
  "checkpointing.enabled=${checkpoint_enabled}"
  checkpointing.save_period=250
  "cluster.gpus_per_node=${num_gpus}"
  "logger.swanlab.name=nemo-${run_mode}-seed-${seed}"
)

if [[ "${workload_mode}" == controlled ]]; then
  # Forward the flag into vLLM's inner workers as well as the outer Ray actor.
  command+=(+policy.generation.vllm_cfg.env_vars.VLLM_BATCH_INVARIANT=1)
fi
if [[ "${runtime_profile}" != discard ]]; then
  command+=("policy.generation.vllm_cfg.sleep_level=${vllm_sleep_level}")
fi
if [[ "${runtime_profile}" == resident ]]; then
  command+=(
    policy.generation.vllm_cfg.gpu_memory_utilization=0.13
  )
fi

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf 'AI_SEARCH_MAX_STEPS=%q AI_SEARCH_NUM_PROMPTS=%q AI_SEARCH_NUM_GENERATIONS=5 ' \
    "${AI_SEARCH_MAX_STEPS}" "${AI_SEARCH_NUM_PROMPTS}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

health_url="${retriever_url%/retrieve}/healthz"
if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 10 \
  --retry 12 --retry-delay 5 --retry-max-time 120 --retry-all-errors \
  "${health_url}" >/dev/null; then
  echo "The shared E5 retriever is not healthy at ${health_url}." >&2
  exit 1
fi

exec "${command[@]}"
