#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nemo_root=$(cd -- "${comparison_dir}/../../../.." && pwd)
slime_root="${SLIME_ROOT:?SLIME_ROOT must point to the frozen slime checkout}"
megatron_root="${SLIME_MEGATRON_ROOT:?SLIME_MEGATRON_ROOT must point to Megatron-LM in the slime runtime}"
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
ref_load="${SEARCH_R1_SLIME_REF_LOAD:?SEARCH_R1_SLIME_REF_LOAD must point to the converted Qwen2.5-7B Megatron checkpoint}"
train_file="${SEARCH_R1_TRAIN_FILE:?SEARCH_R1_TRAIN_FILE must point to four_way_train/train.parquet}"
eval_file="${SEARCH_R1_EVAL_FILE:?SEARCH_R1_EVAL_FILE must point to four_way_eval/test.parquet}"
retriever_url="${SEARCH_R1_RETRIEVER_URL:?SEARCH_R1_RETRIEVER_URL must point to the shared E5 /retrieve endpoint}"
output_dir="${SEARCH_R1_OUTPUT_DIR:?SEARCH_R1_OUTPUT_DIR must be set}"
ray_tmpdir="${SEARCH_R1_RAY_TMPDIR:-${output_dir}/ray}"
run_mode="${SEARCH_R1_RUN_MODE:-smoke}"
observability_mode="${SEARCH_R1_OBSERVABILITY_MODE:-clean}"
seed="${SEARCH_R1_SEED:-42}"
num_gpus="${SEARCH_R1_NUM_GPUS:-8}"
allow_nonformal_preflight="${SEARCH_R1_ALLOW_NONFORMAL_PREFLIGHT:-0}"

expected_slime_base=a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e
expected_slime_patched_head=3c30f8b4954e40f7baa0073d83a62134c1aec82b
expected_slime_patch_sha256=50b55d2602e0a9425dbaab5e16430080a54b39e9b340627f857addd3cdb7cb5d
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39
expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    num_rollout=1
    global_batch_size=40
    eval_interval=""
    save_interval=""
    lr_schedule_horizon_optimizer_updates=500
    lr_warmup_optimizer_updates=142
    default_trace_sample_rate=1.0
    save_debug_rollouts=0
    ;;
  performance)
    prompts_per_step=8
    num_rollout=4
    global_batch_size=40
    eval_interval=""
    save_interval=""
    lr_schedule_horizon_optimizer_updates=500
    lr_warmup_optimizer_updates=142
    default_trace_sample_rate=0.1
    save_debug_rollouts=0
    ;;
  campaign)
    prompts_per_step=512
    num_rollout=500
    global_batch_size=256
    eval_interval=50
    save_interval=250
    # Ten optimizer mini-batches are consumed by every 2,560-trajectory outer step.
    lr_schedule_horizon_optimizer_updates=5000
    lr_warmup_optimizer_updates=1420
    default_trace_sample_rate=0.01
    save_debug_rollouts=1
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac
if [[ "${observability_mode}" == profile && "${run_mode}" != performance ]]; then
  echo "SEARCH_R1_OBSERVABILITY_MODE=profile requires SEARCH_R1_RUN_MODE=performance." >&2
  exit 1
fi

case "${observability_mode}" in
  baseline)
    enable_tensorboard=0
    trace_sample_rate=disabled
    engine_prometheus_scope=not-collected-native-always-on
    ;;
  clean)
    enable_tensorboard=1
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-${default_trace_sample_rate}}"
    engine_prometheus_scope=sglang-router-and-engine-http-servers
    ;;
  profile)
    enable_tensorboard=1
    trace_sample_rate="${SEARCH_R1_TRACE_SAMPLE_RATE:-1.0}"
    engine_prometheus_scope=sglang-router-and-engine-http-servers
    ;;
  *)
    echo "SEARCH_R1_OBSERVABILITY_MODE must be baseline, clean, or profile." >&2
    exit 1
    ;;
esac

if (( $# != 0 )); then
  echo "The strict slime launcher does not accept positional overrides." >&2
  exit 1
fi

actual_slime_head=$(git -C "${slime_root}" rev-parse HEAD)
if ! git -C "${slime_root}" merge-base --is-ancestor \
  "${expected_slime_base}" "${actual_slime_head}"; then
  echo "slime patch head does not descend from ${expected_slime_base}." >&2
  exit 1
fi
actual_slime_patch_sha256="$({
  git -C "${slime_root}" diff --full-index --binary \
    "${expected_slime_base}..${actual_slime_head}"
} | sha256sum | cut -d ' ' -f 1)"
if [[ "${actual_slime_patch_sha256}" != "${expected_slime_patch_sha256}" ]]; then
  echo "slime comparison diff does not match ${expected_slime_patch_sha256}." >&2
  exit 1
fi
if ! git -C "${slime_root}" diff --quiet || ! git -C "${slime_root}" diff --cached --quiet; then
  echo "slime checkout has uncommitted tracked changes." >&2
  exit 1
fi
if [[ "$(basename -- "$(readlink -f "${model_path}")")" != "${expected_model_revision}" ]]; then
  echo "SEARCH_R1_MODEL_PATH does not resolve to revision ${expected_model_revision}." >&2
  exit 1
fi
for required_path in \
  "${ref_load}" \
  "${megatron_root}" \
  "${train_file}" \
  "${eval_file}" \
  "${slime_root}/train.py" \
  "${slime_root}/scripts/models/qwen2.5-7B.sh" \
  "${comparison_dir}/framework_eval_adapters.py" \
  "${comparison_dir}/slime_search_r1_adapter.py" \
  "${comparison_dir}/adapters/slime-search-r1-comparison.patch" \
  "${comparison_dir}/adapters/slime-search-r1-eval.example.yaml"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing aligned slime artifact: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${ref_load}/latest_checkpointed_iteration.txt" ]]; then
  echo "SEARCH_R1_SLIME_REF_LOAD is not a complete Megatron checkpoint." >&2
  exit 1
fi
printf '%s  %s\n' "${expected_train_sha256}" "${train_file}" | sha256sum --check --status
printf '%s  %s\n' "${expected_eval_sha256}" "${eval_file}" | sha256sum --check --status
if [[ "${retriever_url}" != http://*/retrieve && "${retriever_url}" != https://*/retrieve ]]; then
  echo "SEARCH_R1_RETRIEVER_URL must be an HTTP(S) /retrieve endpoint." >&2
  exit 1
fi
if [[ "${ray_tmpdir}" != /* ]]; then
  echo "SEARCH_R1_RAY_TMPDIR must be an absolute path." >&2
  exit 1
fi
if (( ${#ray_tmpdir} > 32 )); then
  echo "SEARCH_R1_RAY_TMPDIR is too long for Ray Unix sockets: ${ray_tmpdir}" >&2
  exit 1
fi
hardware_contract=eight-gpu
if [[ "${num_gpus}" != "8" ]]; then
  if [[ "${run_mode}" != smoke ]] \
    || [[ "${allow_nonformal_preflight}" != "1" ]] \
    || [[ "${num_gpus}" != "4" ]]; then
    echo "Aligned slime runs require eight GPUs; only an explicitly enabled four-GPU smoke preflight is allowed." >&2
    exit 1
  fi
  hardware_contract=nonformal-four-gpu-smoke
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
  "${output_dir}/common-eval" \
  "${output_dir}/debug-rollouts" \
  "${output_dir}/nsight-tmp" \
  "${ray_tmpdir}" \
  "${output_dir}/swanlab" \
  "${output_dir}/tensorboard"

export PYTHONPATH="${comparison_dir}:${slime_root}:${megatron_root}${PYTHONPATH:+:${PYTHONPATH}}"
export SEARCH_R1_RETRIEVER_URL="${retriever_url}"
unset AI_SEARCH_TRACE_PATH AI_SEARCH_TRACE_SAMPLE_RATE TENSORBOARD_DIR
unset SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID
nsys_executable=disabled
nsys_version=disabled
if [[ "${observability_mode}" != baseline ]]; then
  export AI_SEARCH_TRACE_PATH="${output_dir}/trajectory-spans.jsonl"
  export AI_SEARCH_TRACE_SAMPLE_RATE="${trace_sample_rate}"
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
  export SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID=1
fi
export SEARCH_R1_EVAL_FILE="${eval_file}"
export SEARCH_R1_COMMON_EVAL_DIR="${output_dir}/common-eval"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

{
  printf 'framework=slime\n'
  printf 'run_mode=%s\nobservability_mode=%s\n' "${run_mode}" "${observability_mode}"
  printf 'source_upstream_base=%s\nsource_patched_head=%s\n' "${expected_slime_base}" "${actual_slime_head}"
  printf 'reference_patch_head=%s\nsource_patch_sha256=%s\n' "${expected_slime_patched_head}" "${actual_slime_patch_sha256}"
  printf 'adapter_commit=%s\n' "$(git -C "${comparison_dir}" rev-parse HEAD)"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\n' "${expected_train_sha256}"
  printf 'eval_sha256=%s\n' "${expected_eval_sha256}"
  printf 'retriever_url=%s\n' "${retriever_url}"
  printf 'seed=%s\n' "${seed}"
  printf 'physical_gpus=%s\nactor_gpus=%s\nrollout_gpus=%s\ncolocated=true\n' \
    "${num_gpus}" "${num_gpus}" "${num_gpus}"
  printf 'hardware_contract=%s\n' "${hardware_contract}"
  printf 'prompts_per_rollout=%s\n' "${prompts_per_step}"
  printf 'rollouts_per_prompt=5\n'
  printf 'trajectories_per_rollout=%s\n' "$((prompts_per_step * 5))"
  printf 'global_batch_size=%s\n' "${global_batch_size}"
  printf 'actor_microbatch_mode=dynamic\nactor_max_tokens_per_gpu=9216\n'
  printf 'optimizer_updates_per_rollout=%s\n' "$(((prompts_per_step * 5) / global_batch_size))"
  printf 'num_rollout=%s\n' "${num_rollout}"
  printf 'optimizer=AdamW\noptimizer_lr=1e-6\noptimizer_weight_decay=0.01\n'
  printf 'optimizer_betas=0.9,0.999\noptimizer_epsilon=1e-8\n'
  printf 'lr_schedule_horizon_optimizer_updates=%s\n' "${lr_schedule_horizon_optimizer_updates}"
  printf 'lr_warmup_optimizer_updates=%s\n' "${lr_warmup_optimizer_updates}"
  printf 'lr_warmup_outer_steps=142\nlr_scheduler_granularity=outer_step\n'
  printf 'lr_after_warmup=constant\n'
  printf 'timed_rollout_text_dump=%s\n' \
    "$([[ "${save_debug_rollouts}" == 1 ]] && echo campaign-only || echo disabled)"
  printf 'trace_path=%s\n' "${AI_SEARCH_TRACE_PATH:-disabled}"
  printf 'ray_tmpdir=%s\n' "${ray_tmpdir}"
  printf 'trace_sample_rate=%s\n' "${AI_SEARCH_TRACE_SAMPLE_RATE:-disabled}"
  printf 'engine_prometheus_scope=%s\n' "${engine_prometheus_scope}"
  printf 'tensorboard_dir=%s\n' "${TENSORBOARD_DIR:-disabled}"
  printf 'swanlab_source=%s\n' "$([[ "${enable_tensorboard}" == 1 ]] && echo post-run-tensorboard-conversion || echo disabled)"
  printf 'nsys_executable=%s\nnsys_version=%s\n' "${nsys_executable}" "${nsys_version}"
  printf 'nsys_profile_rollout_id=%s\nnsys_scope=%s\n' "${SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID:-disabled}" "$([[ "${observability_mode}" == profile ]] && echo actor-processes-full-outer-step || echo disabled)"
  printf 'nsys_rollout_engine_scope=%s\n' "$([[ "${observability_mode}" == profile ]] && echo missing || echo disabled)"
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign && "${observability_mode}" == clean ]] && echo candidate || echo false)"
} > "${output_dir}/manifest.txt"

# shellcheck source=/dev/null
source "${slime_root}/scripts/models/qwen2.5-7B.sh"

command=(
  python3 "${slime_root}/train.py"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${num_gpus}"
  --rollout-num-gpus "${num_gpus}"
  --colocate
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${model_path}"
  --ref-load "${ref_load}"
  --prompt-data "${train_file}"
  --input-key prompt
  --label-key reward_model
  --metadata-key extra_info
  --apply-chat-template
  --rollout-batch-size "${prompts_per_step}"
  --n-samples-per-prompt 5
  --num-rollout "${num_rollout}"
  --global-batch-size "${global_batch_size}"
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --rollout-seed "${seed}"
  --rollout-max-prompt-len 2048
  --rollout-max-response-len 500
  --rollout-max-context-len 6144
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.2
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --lr-decay-iters "${lr_schedule_horizon_optimizer_updates}"
  --lr-warmup-iters "${lr_warmup_optimizer_updates}"
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.999
  --adam-eps 1e-8
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --seed "${seed}"
  --custom-generate-function-path slime_search_r1_adapter.generate
  --custom-rm-path framework_eval_adapters.slime_reward
  --custom-megatron-before-train-step-hook-path slime_search_r1_adapter.align_outer_step_lr
)

if [[ "${save_debug_rollouts}" == 1 ]]; then
  command+=(
    --save-debug-rollout-data "${output_dir}/debug-rollouts/rollout_{rollout_id}.pt"
  )
fi

if [[ "${enable_tensorboard}" == "1" ]]; then
  command+=(
    --use-tensorboard
    --tb-project-name search-r1-four-way
    --tb-experiment-name "slime-${run_mode}-${observability_mode}-seed-${seed}"
  )
fi
if [[ -n "${eval_interval}" ]]; then
  command+=(
    --eval-config "${comparison_dir}/adapters/slime-search-r1-eval.example.yaml"
    --eval-interval "${eval_interval}"
    --custom-eval-rollout-log-function-path framework_eval_adapters.slime_log_eval_rollout_data
  )
else
  command+=(--skip-eval-before-train)
fi
if [[ -n "${save_interval}" ]]; then
  command+=(
    --save "${output_dir}/checkpoints"
    --save-interval "${save_interval}"
  )
fi

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

master_addr="${MASTER_ADDR:-127.0.0.1}"
ray start \
  --head \
  --node-ip-address "${master_addr}" \
  --num-gpus "${num_gpus}" \
  --temp-dir "${ray_tmpdir}" \
  --disable-usage-stats
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM

if [[ "${observability_mode}" == profile ]]; then
  runtime_env_json=$(printf \
    '{"env_vars":{"PYTHONPATH":"%s","PATH":"%s","NSYS_TMPDIR":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","SEARCH_R1_RETRIEVER_URL":"%s","SEARCH_R1_EVAL_FILE":"%s","SEARCH_R1_COMMON_EVAL_DIR":"%s","TENSORBOARD_DIR":"%s","AI_SEARCH_TRACE_PATH":"%s","AI_SEARCH_TRACE_SAMPLE_RATE":"%s","SEARCH_R1_NSYS_PROFILE_ROLLOUT_ID":"1"},"nsight":{"trace":"cuda,nvtx,cublas,nccl,osrt","cuda-memory-usage":"true","cpuctxsw":"none","capture-range":"cudaProfilerApi","capture-range-end":"stop","kill":"none","o":"slime_actor_%%p"}}' \
    "${PYTHONPATH}" \
    "${PATH}" \
    "${NSYS_TMPDIR}" \
    "${retriever_url}" \
    "${eval_file}" \
    "${SEARCH_R1_COMMON_EVAL_DIR}" \
    "${TENSORBOARD_DIR:-}" \
    "${AI_SEARCH_TRACE_PATH:-}" \
    "${AI_SEARCH_TRACE_SAMPLE_RATE:-}")
else
  runtime_env_json=$(printf \
    '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","SEARCH_R1_RETRIEVER_URL":"%s","SEARCH_R1_EVAL_FILE":"%s","SEARCH_R1_COMMON_EVAL_DIR":"%s","TENSORBOARD_DIR":"%s","AI_SEARCH_TRACE_PATH":"%s","AI_SEARCH_TRACE_SAMPLE_RATE":"%s"}}' \
    "${PYTHONPATH}" \
    "${retriever_url}" \
    "${eval_file}" \
    "${SEARCH_R1_COMMON_EVAL_DIR}" \
    "${TENSORBOARD_DIR:-}" \
    "${AI_SEARCH_TRACE_PATH:-}" \
    "${AI_SEARCH_TRACE_SAMPLE_RATE:-}")
fi

ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${runtime_env_json}" \
  -- "${command[@]}"
