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

The clean and Nsight runs use the same model, data, retriever, and batch sizes,
but remain separate measurements. Trace sampling and profiler state are recorded
explicitly; the profile mode intentionally raises trace sampling to 100 percent.
For the four-framework comparison, the same collector and analysis schema are
used around NeMo RL, the original Search-R1 veRL fork, current veRL, and slime.
Framework-native stages are mapped into a shared top-level critical path;
unsupported or non-equivalent behavior is reported rather than silently
normalized.
