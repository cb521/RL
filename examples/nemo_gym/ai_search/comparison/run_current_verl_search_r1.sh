#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
verl_root="${CURRENT_VERL_ROOT:?CURRENT_VERL_ROOT must point to the frozen veRL checkout}"
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
train_file="${SEARCH_R1_TRAIN_FILE:?SEARCH_R1_TRAIN_FILE must point to four_way_train/train.parquet}"
eval_file="${SEARCH_R1_EVAL_FILE:?SEARCH_R1_EVAL_FILE must point to four_way_eval/test.parquet}"
retriever_url="${SEARCH_R1_RETRIEVER_URL:?SEARCH_R1_RETRIEVER_URL must point to the shared E5 /retrieve endpoint}"
output_dir="${SEARCH_R1_OUTPUT_DIR:?SEARCH_R1_OUTPUT_DIR must be set}"
run_mode="${SEARCH_R1_RUN_MODE:-smoke}"
seed="${SEARCH_R1_SEED:-42}"
num_gpus="${SEARCH_R1_NUM_GPUS:-8}"

expected_verl_commit=5cfb74fa04c7f6e5d98260b8f05157c6a9402695
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_train_sha256=64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39
expected_eval_sha256=7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461

case "${run_mode}" in
  smoke)
    prompts_per_step=8
    total_steps=1
    ppo_mini_batch_size=40
    agent_workers=40
    save_freq=-1
    test_freq=-1
    val_before_train=false
    logger="['console']"
    ;;
  performance)
    prompts_per_step=8
    total_steps=4
    ppo_mini_batch_size=40
    agent_workers=40
    save_freq=-1
    test_freq=-1
    val_before_train=false
    logger="['console','swanlab','tensorboard']"
    ;;
  campaign)
    prompts_per_step=512
    total_steps=500
    ppo_mini_batch_size=256
    agent_workers=256
    save_freq=100
    test_freq=50
    val_before_train=true
    logger="['console','swanlab','tensorboard']"
    ;;
  *)
    echo "SEARCH_R1_RUN_MODE must be smoke, performance, or campaign, not ${run_mode}." >&2
    exit 1
    ;;
esac

if [[ "$(git -C "${verl_root}" rev-parse HEAD)" != "${expected_verl_commit}" ]]; then
  echo "Current veRL checkout moved from ${expected_verl_commit}." >&2
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
  "${comparison_dir}/adapters/current-verl-search-r1-agent-loop.example.yaml"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing aligned current-veRL artifact: ${required_file}" >&2
    exit 1
  fi
done
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

mkdir -p \
  "${output_dir}/checkpoints" \
  "${output_dir}/rollouts" \
  "${output_dir}/validation" \
  "${output_dir}/swanlab" \
  "${output_dir}/tensorboard"

export PYTHONPATH="${comparison_dir}:${verl_root}${PYTHONPATH:+:${PYTHONPATH}}"
export SEARCH_R1_RETRIEVER_URL="${retriever_url}"
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR="${output_dir}/swanlab"
export TENSORBOARD_DIR="${output_dir}/tensorboard"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

{
  printf 'framework=current-verl\n'
  printf 'run_mode=%s\n' "${run_mode}"
  printf 'source_commit=%s\n' "${expected_verl_commit}"
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
  printf 'total_steps=%s\n' "${total_steps}"
  printf 'formal_parity_result=%s\n' "$([[ "${run_mode}" == campaign ]] && echo candidate || echo false)"
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
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
  actor_rollout_ref.actor.optim.weight_decay=0.01
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  "actor_rollout_ref.rollout.agent.num_workers=${agent_workers}"
  actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_text
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${comparison_dir}/adapters/current-verl-search-r1-agent-loop.example.yaml"
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=false
  "reward.custom_reward_function.path=${comparison_dir}/framework_eval_adapters.py"
  reward.custom_reward_function.name=verl_compute_score
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.balance_batch=false
  "trainer.logger=${logger}"
  trainer.project_name=search-r1-four-way
  "trainer.experiment_name=current-verl-${run_mode}-seed-${seed}"
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
  "trainer.rollout_data_dir=${output_dir}/rollouts"
  "trainer.validation_data_dir=${output_dir}/validation"
)

if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}" "${@}"
