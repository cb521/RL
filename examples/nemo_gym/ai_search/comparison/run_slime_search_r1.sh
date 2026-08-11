#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
slime_root="${SLIME_ROOT:?SLIME_ROOT must point to the frozen slime checkout}"
megatron_root="${SLIME_MEGATRON_ROOT:?SLIME_MEGATRON_ROOT must point to Megatron-LM in the slime runtime}"
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
ref_load="${SEARCH_R1_SLIME_REF_LOAD:?SEARCH_R1_SLIME_REF_LOAD must point to the converted Qwen2.5-7B Megatron checkpoint}"
train_file="${SEARCH_R1_TRAIN_FILE:?SEARCH_R1_TRAIN_FILE must point to four_way_train/train.parquet}"
eval_file="${SEARCH_R1_EVAL_FILE:?SEARCH_R1_EVAL_FILE must point to four_way_eval/test.parquet}"
retriever_url="${SEARCH_R1_RETRIEVER_URL:?SEARCH_R1_RETRIEVER_URL must point to the shared E5 /retrieve endpoint}"
output_dir="${SEARCH_R1_OUTPUT_DIR:?SEARCH_R1_OUTPUT_DIR must be set}"
run_mode="${SEARCH_R1_RUN_MODE:-smoke}"
seed="${SEARCH_R1_SEED:-42}"
num_gpus="${SEARCH_R1_NUM_GPUS:-8}"

expected_slime_commit=a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e
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
    enable_tensorboard=0
    ;;
  performance)
    prompts_per_step=8
    num_rollout=4
    global_batch_size=40
    eval_interval=""
    save_interval=""
    enable_tensorboard=1
    ;;
  campaign)
    prompts_per_step=512
    num_rollout=500
    global_batch_size=256
    eval_interval=50
    save_interval=100
    enable_tensorboard=1
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

if [[ "$(git -C "${slime_root}" rev-parse HEAD)" != "${expected_slime_commit}" ]]; then
  echo "slime checkout moved from ${expected_slime_commit}." >&2
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
if [[ "${num_gpus}" != "8" ]]; then
  echo "Aligned slime runs require exactly eight total actor-plus-rollout GPUs." >&2
  exit 1
fi
if [[ "${seed}" == *[!0-9]* || -z "${seed}" ]]; then
  echo "SEARCH_R1_SEED must be a non-negative integer." >&2
  exit 1
fi

mkdir -p \
  "${output_dir}/checkpoints" \
  "${output_dir}/common-eval" \
  "${output_dir}/debug-rollouts" \
  "${output_dir}/swanlab" \
  "${output_dir}/tensorboard"

export PYTHONPATH="${comparison_dir}:${slime_root}:${megatron_root}${PYTHONPATH:+:${PYTHONPATH}}"
export SEARCH_R1_RETRIEVER_URL="${retriever_url}"
export SEARCH_R1_EVAL_FILE="${eval_file}"
export SEARCH_R1_COMMON_EVAL_DIR="${output_dir}/common-eval"
export TENSORBOARD_DIR="${output_dir}/tensorboard"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

{
  printf 'framework=slime\n'
  printf 'run_mode=%s\n' "${run_mode}"
  printf 'source_commit=%s\n' "${expected_slime_commit}"
  printf 'adapter_commit=%s\n' "$(git -C "${comparison_dir}" rev-parse HEAD)"
  printf 'model_revision=%s\n' "${expected_model_revision}"
  printf 'train_sha256=%s\n' "${expected_train_sha256}"
  printf 'eval_sha256=%s\n' "${expected_eval_sha256}"
  printf 'retriever_url=%s\n' "${retriever_url}"
  printf 'seed=%s\n' "${seed}"
  printf 'actor_gpus=4\nrollout_gpus=4\ncolocated=true\n'
  printf 'prompts_per_rollout=%s\n' "${prompts_per_step}"
  printf 'rollouts_per_prompt=5\n'
  printf 'trajectories_per_rollout=%s\n' "$((prompts_per_step * 5))"
  printf 'global_batch_size=%s\n' "${global_batch_size}"
  printf 'optimizer_updates_per_rollout=%s\n' "$(((prompts_per_step * 5) / global_batch_size))"
  printf 'num_rollout=%s\n' "${num_rollout}"
  printf 'swanlab_source=post-run-tensorboard-conversion\n'
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign ]] && echo candidate || echo false)"
} > "${output_dir}/manifest.txt"

# shellcheck source=/dev/null
source "${slime_root}/scripts/models/qwen2.5-7B.sh"

command=(
  python3 "${slime_root}/train.py"
  --actor-num-nodes 1
  --actor-num-gpus-per-node 4
  --rollout-num-gpus 4
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
  --lr-decay-style linear
  --lr-warmup-fraction 0.285
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
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
  --save-debug-rollout-data "${output_dir}/debug-rollouts/rollout_{rollout_id}.pt"
)

if [[ "${enable_tensorboard}" == "1" ]]; then
  command+=(
    --use-tensorboard
    --tb-project-name search-r1-four-way
    --tb-experiment-name "slime-${run_mode}-seed-${seed}"
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
  --disable-usage-stats
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM

runtime_env_json=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","SEARCH_R1_RETRIEVER_URL":"%s","SEARCH_R1_EVAL_FILE":"%s","SEARCH_R1_COMMON_EVAL_DIR":"%s","TENSORBOARD_DIR":"%s"}}' \
  "${PYTHONPATH}" \
  "${retriever_url}" \
  "${eval_file}" \
  "${SEARCH_R1_COMMON_EVAL_DIR}" \
  "${TENSORBOARD_DIR}")

ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${runtime_env_json}" \
  -- "${command[@]}" "${@}"
