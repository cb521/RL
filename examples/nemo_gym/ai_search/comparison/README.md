# Aligned AI-search training comparison

This directory defines the controlled workload used to compare NeMo RL,
Search-R1, and Alibaba ZeroSearch. It is a performance experiment, not a model
quality benchmark.

All three runs use Qwen2.5-3B-Instruct, BF16 full-parameter GRPO, four prompts
per optimizer step, four trajectories per prompt, at most three model calls
(two searchable rounds plus one final answer), top-3 retrieval, normalized
exact-match answer reward, temperature 1.0, and the same
32-train/8-validation fixture.
The first step warms caches; steps 2-4 are the measured window. A final
validation and checkpoint are reported separately from core-step throughput.

The fixture contains synthetic two-hop facts. Each question first identifies a
project custodian and then asks for that person's birthplace. Answers and names
are generated identifiers, so the policy must use the supplied evidence rather
than pretrained factual memory. `prepare_aligned_data.py` writes both the NeMo
Gym JSONL view and the nested Parquet schema consumed by the two veRL forks.

All runs call `retriever_server.py`, a deterministic in-memory BM25 service.
It accepts both the Search-R1 batch protocol and ZeroSearch's single-query
protocol. Keeping retrieval and corpus identical isolates differences in the
trainer, model serving, and agent loop. Framework-native action syntax remains
different: NeMo Gym uses structured tool calls; Search-R1 and ZeroSearch use
`<search>`/`<answer>` tags. Reports must therefore include actual token counts
and token-normalized throughput beside wall time.

The measured 4x A100 result and bottleneck analysis are in
[`PERFORMANCE.md`](PERFORMANCE.md). Machine-readable aggregates and per-step
records are in `a100_sxm4_4x_20260809.json`.

Generate the fixture with:

```bash
uv run --with pandas --with pyarrow \
  python examples/nemo_gym/ai_search/comparison/prepare_aligned_data.py
```

Generated Parquet and runtime artifacts remain untracked. Each experiment log
records the exact source commits, commands, hardware, package versions, stage
timings, search request log, and one-second GPU/host memory samples.

The separate full-quality experiment follows Search-R1's official 7B training
setup instead of this small performance fixture. Its exact parity contract,
including the few places where the paper and repository disagree, is recorded
in [`SEARCH_R1_PARITY.md`](SEARCH_R1_PARITY.md).

The later NeMo RL, original Search-R1, current veRL, and slime comparison is
defined in [`FOUR_WAY_COMPARISON.md`](FOUR_WAY_COMPARISON.md). It separates
native-example results from a strict protocol-aligned scoreboard and fixes the
quality metrics, performance workload, critical-path buckets, and validity
gates before any four-way compute is launched.

Download and convert the official question data with:

```bash
bash examples/nemo_gym/ai_search/comparison/download_search_r1_data.sh \
  /path/to/nq_hotpotqa_train

uv run --with pandas --with pyarrow \
  python examples/nemo_gym/ai_search/comparison/prepare_search_r1_data.py \
  --source-dir /path/to/nq_hotpotqa_train
```

The 2018 Wikipedia corpus and E5 index are separate artifacts of roughly 70 GB
combined. Stage them on storage visible to the retrieval service rather than in
this repository.

## Four-layer performance evidence

The formal comparison uses four complementary views. They answer different
questions and must not be collapsed into one profiler run:

1. SwanLab or W&B records step-level quality, throughput, GPU utilization, and
   trainer supply. The observed recipe enables SwanLab and TensorBoard; setting
   `SWANLAB_MODE=local` needs no cloud account.
2. Nsight Systems records CUDA, NCCL, and NVTX activity for a short steady-state
   window. It is run separately because profiler overhead would bias the clean
   end-to-end result.
3. Prometheus records bounded-cardinality engine and retrieval-service counters,
   gauges, and latency histograms. The local resource server and the observed
   official E5 launcher expose `/metrics`.
4. Sampled JSONL spans join model calls, tool calls, resource-server queueing,
   and remote E5 batches by trajectory session and provider batch IDs.

Run a low-overhead measurement with local SwanLab storage:

```bash
SWANLAB_MODE=local \
AI_SEARCH_OBSERVABILITY_MODE=clean \
AI_SEARCH_RUN_DIR=/fast/local/run/search-r1-clean \
  bash examples/nemo_gym/ai_search/run_ai_search_observed.sh
```

Run Nsight only on a short, separate window (step 2 by default):

```bash
AI_SEARCH_OBSERVABILITY_MODE=profile \
AI_SEARCH_MAX_STEPS=4 \
  bash examples/nemo_gym/ai_search/run_ai_search_observed.sh
```

Profile mode requires the `nsys` CLI on `PATH`. Point `NSYS_TMPDIR` at
node-local storage, and record the CLI version plus package checksum in the run
manifest; profile data and SQLite conversion scratch can be large and should
not use a network-mounted temporary directory.

The official E5 service wrapper preserves the upstream E5 model, corpus, and
float16 FlatIP search while loading the 21-million-vector index in bounded
chunks. Its request, encode, index, and document-fetch metrics can be sampled
with:

```bash
uv run python examples/nemo_gym/ai_search/comparison/collect_prometheus.py \
  --endpoint e5=http://RETRIEVER_HOST:8000/metrics \
  --output /path/to/run/prometheus.jsonl
```

Merge the independent evidence after a run:

```bash
uv run python examples/nemo_gym/ai_search/comparison/analyze_observability.py \
  --console /path/to/run/console.log \
  --trace /path/to/run/trajectory-spans.jsonl \
  --prometheus /path/to/run/prometheus.jsonl \
  --warmup-steps 1 \
  --output /path/to/run/observability-summary.json
```

Framework adapters export final evaluation predictions to a common JSONL
schema with `example_id`, `data_source`, `golden_answers`, `response`, and
`status`. Token counts, wall time, search errors, and invalid actions are
optional evidence fields. Recompute quality and paired uncertainty instead of
comparing framework-native reward logs:

```bash
uv run python examples/nemo_gym/ai_search/comparison/evaluate_search_r1.py \
  --predictions nemo=/path/to/nemo-predictions.jsonl \
  --predictions original_search_r1=/path/to/search-r1-predictions.jsonl \
  --predictions current_verl=/path/to/verl-predictions.jsonl \
  --predictions slime=/path/to/slime-predictions.jsonl \
  --output /path/to/four-way-quality.json
```

By default, the evaluator rejects missing or different example IDs and ground
truth. It reports normalized exact match, answer F1, per-source results,
retrieval answer recall, format and termination rates, search/error/repetition
statistics, and paired bootstrap confidence intervals.

NeMo Gym's trajectory-collection mode preserves the complete verify result,
including the stable source ID, source dataset, generated actions, injected
observations, token usage, and native diagnostics. Convert that artifact before
running the common evaluator:

```bash
uv run python \
  examples/nemo_gym/ai_search/comparison/export_nemo_gym_predictions.py \
  --input /path/to/logs/exp_001/trajectory_collection.jsonl \
  --output /path/to/nemo-predictions.jsonl
```

### Common evaluation data and framework adapters

The official test file reuses IDs across its seven sources. Prepare a shared
Parquet view that keeps every source column and row in place while adding the
globally unique `{data_source}:{id}`, source, question, and answers under
`extra_info`:

```bash
uv run --with pandas --with pyarrow python \
  examples/nemo_gym/ai_search/comparison/prepare_four_way_eval_data.py \
  --source /path/to/nq_hotpotqa_train/test.parquet \
  --output /path/to/four-way/test.parquet \
  --manifest /path/to/four-way/test.manifest.json
```

The converter rereads its output and rejects any change to the prompt, answer,
row order, reward model, or other source field. On the frozen official file it
finds 51,713 unique composite IDs but only 26,843 unique raw IDs.

For the original Search-R1 fork at the frozen base revision, apply the disclosed
export-only patch and put this comparison directory on `PYTHONPATH`:

```bash
git -C /path/to/Search-R1 apply \
  /path/to/nemo-rl/examples/nemo_gym/ai_search/comparison/adapters/original-search-r1-validation.patch
export PYTHONPATH="/path/to/nemo-rl/examples/nemo_gym/ai_search/comparison:${PYTHONPATH}"
```

Add this Hydra override to validation:

```text
+trainer.validation_predictions_path=/path/to/original-search-r1-predictions.jsonl
```

The patch calls `original_search_r1_export.py` only after generation and native
reward computation. It streams response-only text, loss-mask-derived generated
and observation token counts, native reward, status, and stable identity to a
partial file, then publishes it atomically after validation finishes.

For current veRL, point `reward.custom_reward_function.path` at
`framework_eval_adapters.py`, set its name to `verl_compute_score`, use one
validation rollout, and enable `trainer.validation_data_dir`. The reward remains
normalized answer EM but also returns stable evidence fields that current veRL
includes in its validation JSONL. Convert that dump with:

```bash
uv run python \
  examples/nemo_gym/ai_search/comparison/export_verl_predictions.py \
  --input /path/to/verl-validation/500.jsonl \
  --output /path/to/current-verl-predictions.jsonl
```

For slime, use the prepared Parquet as the evaluation dataset with `prompt`,
`reward_model`, and `extra_info` as the input, label, and metadata keys. Put the
comparison directory on `PYTHONPATH`, set
`SEARCH_R1_COMMON_EVAL_DIR=/path/to/slime-eval`, and configure these public
hooks:

```text
custom_rm_path: framework_eval_adapters.slime_reward
--custom-eval-rollout-log-function-path \
  framework_eval_adapters.slime_log_eval_rollout_data
```

The slime reward deliberately consumes `sample.response`, not
`prompt + response`; this removes the native example's dependency on the
demonstration `<answer>` tag. The logging hook preserves slime's normal metric
logging and atomically writes one common record per question, including token
counts derived from its response loss mask. All three adapters fail on missing
or duplicate IDs instead of silently joining by row order.

The clean and Nsight runs use the same model, data, retriever, and batch sizes,
but remain separate measurements. Trace sampling and profiler state are recorded
explicitly; the profile mode intentionally raises trace sampling to 100 percent.
Summarize the per-worker Nsight reports without adding overlapping CUDA and
NVTX durations to the end-to-end critical path:

```bash
uv run python examples/nemo_gym/ai_search/comparison/analyze_nsys.py \
  --input /path/to/dtensor_policy_worker_2:3_100.nsys-rep \
  --input /path/to/vllm_generation_worker_2:3_101.nsys-rep \
  --output /path/to/nsight-summary.json
```

The summary preserves report-local CUDA API, kernel, memory-operation,
launch-queue, and NVTX rankings. Its percentages are deliberately scoped to
each report because workers and nested ranges overlap in wall-clock time.
For the four-framework comparison, the same collector and analysis schema are
used around NeMo RL, the original Search-R1 veRL fork, current veRL, and slime.
Framework-native stages are mapped into a shared top-level critical path;
unsupported or non-equivalent behavior is reported rather than silently
normalized.
