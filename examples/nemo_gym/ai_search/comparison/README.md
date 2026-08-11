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
   `SWANLAB_MODE=local` needs no cloud account. The observed launcher installs
   SwanLab's optional dashboard extra at the same version as the environment's
   existing SwanLab package. It also pins Peewee 3.19.0 because SwanBoard
   0.1.8b1 does not cap that dependency and its local transaction code is not
   compatible with Peewee 4.
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

Measure the low-overhead observer cost with an otherwise identical baseline.
This mode retains the recipe's native TensorBoard logging but disables local
SwanLab, the Ray GPU monitor, and trajectory spans:

```bash
AI_SEARCH_OBSERVABILITY_MODE=baseline \
AI_SEARCH_RUN_DIR=/fast/local/run/search-r1-baseline \
  bash examples/nemo_gym/ai_search/run_ai_search_observed.sh
```

The surrounding benchmark launcher must also disable Prometheus polling and
its one-second resource sampler for the baseline run. Compare the same steady
steps, generated/trained token counts, and completed trajectories; report both
the absolute difference and `(clean / baseline - 1) * 100` percent overhead.
The baseline is only an instrumentation-control row, not a replacement for the
four-layer clean evidence.

After producing both `observability.json` files, build the paired report with:

```bash
python examples/nemo_gym/ai_search/comparison/compare_observability_overhead.py \
  --baseline /fast/local/run/search-r1-baseline/observability.json \
  --clean /fast/local/run/search-r1-clean/observability.json \
  --output /fast/local/run/search-r1-observability-overhead.json
```

The comparison fails if measured step IDs or completed trajectory counts differ.
It retains per-step timing, throughput, result, and actual-work deltas so a
token-length change cannot be mistaken for instrumentation overhead.

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

Before a four-way run, prove that the NeMo JSONL view and the external-framework
Parquet view remain row-identical. Run this once for train and once for
validation, and archive both output manifests:

```bash
uv run --with pyarrow python \
  examples/nemo_gym/ai_search/comparison/verify_four_way_data_views.py \
  --jsonl /path/to/search-r1-composite/train.jsonl \
  --parquet /path/to/four-way/train.parquet \
  --output /path/to/evidence/train-view-equivalence.json
```

The verifier compares ordered composite IDs, sources, questions, complete
prompt messages, answer lists, and reward ground truth. The frozen full views
pass for all 169,615 training rows and 51,713 evaluation rows.

For the original Search-R1 fork at the frozen base revision, apply and commit
the disclosed comparison diff, then put this comparison directory on
`PYTHONPATH`:

```bash
git -C /path/to/Search-R1 apply \
  /path/to/nemo-rl/examples/nemo_gym/ai_search/comparison/adapters/original-search-r1-comparison.patch
git -C /path/to/Search-R1 add \
  benchmark_aligned.sh search_r1 verl
git -C /path/to/Search-R1 commit -m 'Apply aligned Search-R1 comparison patch'
export PYTHONPATH="/path/to/nemo-rl/examples/nemo_gym/ai_search/comparison:${PYTHONPATH}"
```

Add this Hydra override to validation:

```text
+trainer.validation_predictions_path=/path/to/original-search-r1-predictions.jsonl
```

The launcher verifies the full diff SHA-256, so the local commit hash may differ
without weakening the source gate. The patch calls
`original_search_r1_export.py` only after generation and native reward
computation. It streams response-only text, loss-mask-derived generated and
observation token counts, native reward, status, and stable identity to a
partial file, then publishes it atomically after validation finishes.

For current veRL, point `reward.custom_reward_function.path` at
`framework_eval_adapters.py`, set its name to `verl_compute_score`, use one
validation rollout, and enable `trainer.validation_data_dir`. The reward remains
normalized answer EM but also returns stable evidence fields that current veRL
includes in its validation JSONL. Convert that dump with:

Put this comparison directory on `PYTHONPATH` and point veRL's Agent Loop at
`adapters/current-verl-search-r1-agent-loop.example.yaml`:

```text
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/current-verl-search-r1-agent-loop.example.yaml
actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_text
actor_rollout_ref.rollout.prompt_length=2048
actor_rollout_ref.rollout.response_length=4096
actor_rollout_ref.rollout.calculate_log_probs=true
actor_rollout_ref.rollout.n=5
actor_rollout_ref.rollout.val_kwargs.n=1
```

Set `SEARCH_R1_RETRIEVER_URL` to the shared E5 service. The adapter uses the
current public Hydra Agent Loop factory, server manager, selected-token
log-probabilities, response mask, rollout trace carrier, and async reward loop.
It makes no source change to veRL. Four executable generations, the optional
fifth answer-only generation, all token caps, stop-delimiter retention, E5
top-3 retrieval, and provider batch IDs are fixed in code. Agent diagnostics
flow through `tool_extra_fields` into the custom reward and therefore into the
native validation dump.

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
custom_generate_function_path: slime_search_r1_adapter.generate
custom_rm_path: framework_eval_adapters.slime_reward
--custom-eval-rollout-log-function-path \
  framework_eval_adapters.slime_log_eval_rollout_data
```

`adapters/slime-search-r1-eval.example.yaml` is a complete evaluation-dataset
template. The aligned generator uses slime's own SGLang token IDs, selected
token log probabilities, response loss mask, and trace carrier. It fixes four
executable turns, a fifth search-disabled generation when unfinished, 500-token
action and observation limits, the 2,048-token starting prompt, the 4,096-token
recorded response, and E5 top-3 retrieval. Provider batch IDs join its retrieval
spans to the shared E5 service.

The slime reward deliberately consumes `sample.response`, not
`prompt + response`; this removes the native example's dependency on the
demonstration `<answer>` tag. The logging hook preserves slime's normal metric
logging and atomically writes one common record per question, including token
counts derived from its response loss mask. All three adapters fail on missing
or duplicate IDs instead of silently joining by row order.

### Aligned framework launchers

The launchers below fail closed on source revision, model revision, data hashes,
retriever URL, and the eight-GPU requirement. `smoke` runs one 8-question x
5-trajectory update, `performance` runs one warm-up plus three measured updates,
and `campaign` runs the official 512-question x 5-trajectory, 500-step quality
campaign. Only a completed campaign can be considered for the quality table.

Run NeMo RL with its composite JSONL view and strict four-way overlay:

```bash
SEARCH_R1_MODEL_PATH=/path/to/models--Qwen--Qwen2.5-7B/snapshots/d149729... \
SEARCH_R1_NEMO_TRAIN_FILE=/path/to/search-r1-composite/train.jsonl \
SEARCH_R1_NEMO_EVAL_FILE=/path/to/search-r1-composite/validation.jsonl \
SEARCH_R1_RETRIEVER_URL=http://retriever:8000/retrieve \
SEARCH_R1_OUTPUT_DIR=/fast/local/nemo-performance \
SEARCH_R1_RUN_MODE=performance \
  bash examples/nemo_gym/ai_search/comparison/run_nemo_search_r1.sh
```

This launcher fixes the source order, model snapshot, optimizer workload,
greedy validation sampling, and all hashes. It routes performance runs through
the four-layer observed launcher; use `SEARCH_R1_OBSERVABILITY_MODE=baseline`
for the paired instrumentation-off measurement and `profile` only for the
separate Nsight run.

Run the original Search-R1 fork from its patched comparison checkout with:

```bash
ORIGINAL_SEARCH_R1_ROOT=/path/to/Search-R1-at-d5b269d \
SEARCH_R1_MODEL_PATH=/path/to/models--Qwen--Qwen2.5-7B/snapshots/d149729... \
SEARCH_R1_TRAIN_FILE=/path/to/four-way/train.parquet \
SEARCH_R1_EVAL_FILE=/path/to/four-way/test.parquet \
SEARCH_R1_RETRIEVER_URL=http://retriever:8000/retrieve \
SEARCH_R1_OUTPUT_DIR=/fast/local/original-search-r1-performance \
SEARCH_R1_RUN_MODE=performance \
  bash examples/nemo_gym/ai_search/comparison/run_original_search_r1.sh
```

Its manifest records the untouched public base `598e61b` separately from the
comparison patch head `d5b269d`. The patch stack adds benchmark counters,
outer-step scheduler alignment, optional final validation, and common
validation export. It also writes sampled generation/retrieval spans, passes a
provider-batch ID to E5, mirrors reduced metrics to raw JSONL and TensorBoard,
and exposes driver metrics on Prometheus port 9108 by default. It is not
labeled as an upstream Search-R1 release.

After the timed process exits, mirror its TensorBoard source of truth into a
local SwanLab directory without an account:

```bash
swanlab convert -t tensorboard \
  --tb_logdir /fast/local/original-search-r1-performance/tensorboard \
  --mode local \
  -p search-r1-four-way \
  -l /fast/local/original-search-r1-performance/swanlab
```

Run current veRL from its frozen checkout with:

```bash
CURRENT_VERL_ROOT=/path/to/verl-at-5cfb74f \
SEARCH_R1_MODEL_PATH=/path/to/models--Qwen--Qwen2.5-7B/snapshots/d149729... \
SEARCH_R1_TRAIN_FILE=/path/to/four-way/train.parquet \
SEARCH_R1_EVAL_FILE=/path/to/four-way/test.parquet \
SEARCH_R1_RETRIEVER_URL=http://retriever:8000/retrieve \
SEARCH_R1_OUTPUT_DIR=/fast/local/current-verl-performance \
SEARCH_R1_RUN_MODE=performance \
  bash examples/nemo_gym/ai_search/comparison/run_current_verl_search_r1.sh
```

Current veRL writes console metrics, TensorBoard events, local SwanLab data,
native rollout dumps, validation dumps, checkpoints, and a run manifest under
the output directory. `SWANLAB_MODE=local` avoids any account or network login.

slime needs the same Hugging Face snapshot for SGLang and a one-time
Megatron-Core `torch_dist` conversion for training. In the frozen slime runtime,
create it before the timed run:

```bash
cd /path/to/slime-at-a74ae3a
source scripts/models/qwen2.5-7B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /path/to/Qwen2.5-7B \
  --save /fast/local/Qwen2.5-7B_torch_dist
```

Then launch slime with:

```bash
SLIME_ROOT=/path/to/slime-at-a74ae3a \
SLIME_MEGATRON_ROOT=/path/to/Megatron-LM \
SEARCH_R1_MODEL_PATH=/path/to/models--Qwen--Qwen2.5-7B/snapshots/d149729... \
SEARCH_R1_SLIME_REF_LOAD=/fast/local/Qwen2.5-7B_torch_dist \
SEARCH_R1_TRAIN_FILE=/path/to/four-way/train.parquet \
SEARCH_R1_EVAL_FILE=/path/to/four-way/test.parquet \
SEARCH_R1_RETRIEVER_URL=http://retriever:8000/retrieve \
SEARCH_R1_OUTPUT_DIR=/fast/local/slime-performance \
SEARCH_R1_RUN_MODE=performance \
  bash examples/nemo_gym/ai_search/comparison/run_slime_search_r1.sh
```

The slime launcher uses four actor GPUs and four rollout GPUs with colocated
lifecycle management. It keeps the conversion and initialization outside the
measured update window, emits TensorBoard and raw rollout artifacts, and uses
the common evaluation hook for campaign checkpoints. Set
`SEARCH_R1_PRINT_COMMAND=1` on any external launcher to validate and print the
fully resolved command without starting Ray or allocating model memory.

The current-veRL and slime launchers also write sampled common JSONL spans for
model generations and retrieval calls. `performance` samples trajectories
deterministically at 10% by default, while `smoke` records all trajectories and
`campaign` records 1%.
Set `SEARCH_R1_TRACE_SAMPLE_RATE` to override the rate. The retrieval span and
the E5 service span carry the same provider batch ID, so queue, encoding,
Faiss, document fetch, and network time can be joined without relying on wall
clock proximity.

The frozen slime revision has no native SwanLab logger. Preserve its raw
TensorBoard events as the source of truth, then mirror them to a local SwanLab
directory outside the timed window:

```bash
swanlab convert -t tensorboard \
  --tb_logdir /fast/local/slime-performance/tensorboard \
  --mode local \
  -p search-r1-four-way \
  -l /fast/local/slime-performance/swanlab
```

Record the SwanLab and converter dependency versions in the manifest. This
conversion is a visualization layer only and is never counted as training time.

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
