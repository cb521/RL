# Fast Search-R1 Profiling Loop

This loop is for screening NeMo RL performance changes on the frozen 4xH20
Search-R1 workload. It does not replace the full headline benchmark.

## Why Two Steps Instead of One

Step 1 warms model execution, allocators, kernels, the rollout service, weight
refit, and training state. Measuring only that step mixes startup effects with
the change under test. Fast screening therefore runs:

1. one untimed warmup outer step;
2. one measured outer step with the full 8 prompts x 5 trajectories.

The algorithm, 40-trajectory batch, model, data, search service, sampling, and
optimizer contract stay unchanged. Only the number of repeated outer steps is
reduced from four to two.

## Two-Tier Workflow

### Tier 1: low-overhead clean screen

Use stage timers, trajectory spans, Prometheus counters, and one-second GPU/host
samples. Do not enable Nsight for every candidate.

```bash
SEARCH_R1_RETRIEVER_URL=http://R6KD-CX8aaS-GPU-09:18000/retrieve \
SEARCH_R1_E5_RUN_KEY=<existing-e5-key> \
SEARCH_R1_FAST_RUN_KEY=<unique-run-key> \
SEARCH_R1_NEMO_VARIANT=<variant-name> \
SEARCH_R1_OBSERVABILITY_MODE=clean \
bash submit_blackwell_nemo_fast_iteration.sh
```

Compare total step time, the eight functional stages, actual generated and
processed tokens, peak memory, correctness/error gates, and stage-normalized
throughput. A one-step result is directional evidence, not a final claim.

### Tier 2: narrow target profile

Profile only the worker class implicated by Tier 1. The default is policy-only
and captures measured step 2 (`2:3`, inclusive/exclusive).

```bash
SEARCH_R1_RETRIEVER_URL=http://R6KD-CX8aaS-GPU-09:18000/retrieve \
SEARCH_R1_E5_RUN_KEY=<existing-e5-key> \
SEARCH_R1_FAST_RUN_KEY=<unique-run-key> \
SEARCH_R1_NEMO_VARIANT=<variant-name> \
SEARCH_R1_OBSERVABILITY_MODE=profile \
SEARCH_R1_NSYS_BIN=<workspace-path-to-nsys> \
NRL_NSYS_WORKER_PATTERNS='*policy*' \
bash submit_blackwell_nemo_fast_iteration.sh
```

Use `*vllm*` instead when rollout is the suspected bottleneck, or explicitly use
`*policy*,*vllm*` only when a cross-worker timeline is required. CUDA and NVTX
are captured; CPU sampling/context switches are disabled to bound overhead.

### Promotion Gate

Only candidates that improve the target stage without correctness errors or an
unsafe memory increase are promoted to the original four-step run: one warmup
plus three measured steps. Headline A/B numbers come from clean runs; a separate
bounded Nsight run explains the result.

## Evidence Basis

- NeMo RL officially supports selecting Ray worker types with
  `NRL_NSYS_WORKER_PATTERNS` and an inclusive/exclusive step range with
  `NRL_NSYS_PROFILE_STEP_RANGE`:
  <https://docs.nvidia.com/nemo/rl/latest/nsys-profiling.html>
- Nsight Systems supports delayed NVTX/API capture and explicit range-end
  behavior, avoiding startup-wide traces:
  <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- PyTorch Profiler schedules separate wait, warmup, and active windows:
  <https://docs.pytorch.org/docs/stable/profiler.html>
- vLLM's official profiling example uses multiple warmup iterations and one
  measured iteration, and recommends dynamic capture for a server:
  <https://docs.vllm.ai/en/stable/contributing/profiling/>

These sources support the shape of the loop. The exact one-warmup/one-measured
choice is a project-specific screening tradeoff validated against the existing
full Search-R1 runs.
