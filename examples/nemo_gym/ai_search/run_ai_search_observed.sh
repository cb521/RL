#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

AI_SEARCH_PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SEARCH_OBSERVABILITY_MODE="${AI_SEARCH_OBSERVABILITY_MODE:-clean}"
AI_SEARCH_RUN_DIR="${AI_SEARCH_RUN_DIR:-/tmp/nemo-rl-ai-search/runs/grpo-ai-search-observed}"
AI_SEARCH_OBSERVABILITY_DIR="${AI_SEARCH_OBSERVABILITY_DIR:-${AI_SEARCH_RUN_DIR}/observability}"

case "${AI_SEARCH_OBSERVABILITY_MODE}" in
  clean)
    if [[ -n "${NRL_NSYS_WORKER_PATTERNS:-}" || -n "${NRL_NSYS_PROFILE_STEP_RANGE:-}" ]]; then
      echo 'Clean measurement mode refuses Nsight settings; use AI_SEARCH_OBSERVABILITY_MODE=profile.' >&2
      exit 1
    fi
    AI_SEARCH_TRACE_SAMPLE_RATE="${AI_SEARCH_TRACE_SAMPLE_RATE:-0.01}"
    ;;
  profile)
    AI_SEARCH_MAX_STEPS="${AI_SEARCH_MAX_STEPS:-4}"
    AI_SEARCH_TRACE_SAMPLE_RATE="${AI_SEARCH_TRACE_SAMPLE_RATE:-1.0}"
    NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*vllm*}"
    NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-2:3}"
    # Ray profiles several policy and generation processes concurrently. CUDA
    # and NVTX traces are process-local, but device-wide GPU metric sampling is
    # exclusive and would make those workers contend for the same counters.
    # The clean one-second monitor already records device utilization and power.
    NRL_NSYS_EXTRA_OPTIONS="${NRL_NSYS_EXTRA_OPTIONS:-{\"cuda-memory-usage\":\"true\",\"cpuctxsw\":\"none\"}}"
    export AI_SEARCH_MAX_STEPS NRL_NSYS_WORKER_PATTERNS
    export NRL_NSYS_PROFILE_STEP_RANGE NRL_NSYS_EXTRA_OPTIONS
    ;;
  *)
    echo "AI_SEARCH_OBSERVABILITY_MODE must be clean or profile, not ${AI_SEARCH_OBSERVABILITY_MODE}." >&2
    exit 1
    ;;
esac

mkdir -p \
  "${AI_SEARCH_OBSERVABILITY_DIR}" \
  "${AI_SEARCH_OBSERVABILITY_DIR}/swanlab" \
  "${AI_SEARCH_RUN_DIR}"

export AI_SEARCH_RUN_DIR
export AI_SEARCH_CONFIG="${AI_SEARCH_CONFIG:-${AI_SEARCH_PLUGIN_DIR}/grpo_qwen2_5_7b_search_r1_observed.yaml}"
export AI_SEARCH_TRACE_PATH="${AI_SEARCH_TRACE_PATH:-${AI_SEARCH_OBSERVABILITY_DIR}/trajectory-spans.jsonl}"
export AI_SEARCH_TRACE_SAMPLE_RATE
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${AI_SEARCH_OBSERVABILITY_DIR}/swanlab}"

{
  printf 'mode=%s\n' "${AI_SEARCH_OBSERVABILITY_MODE}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'trace_path=%s\n' "${AI_SEARCH_TRACE_PATH}"
  printf 'trace_sample_rate=%s\n' "${AI_SEARCH_TRACE_SAMPLE_RATE}"
  printf 'swanlab_mode=%s\n' "${SWANLAB_MODE}"
  printf 'swanlab_logdir=%s\n' "${SWANLAB_LOGDIR}"
  printf 'nsys_worker_patterns=%s\n' "${NRL_NSYS_WORKER_PATTERNS:-disabled}"
  printf 'nsys_step_range=%s\n' "${NRL_NSYS_PROFILE_STEP_RANGE:-disabled}"
} > "${AI_SEARCH_OBSERVABILITY_DIR}/manifest.txt"

exec bash "${AI_SEARCH_PLUGIN_DIR}/run_ai_search.sh" "$@"
