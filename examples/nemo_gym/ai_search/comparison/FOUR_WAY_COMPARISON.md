# Four-way Search-R1 comparison contract

This document defines the quality and performance comparison among NeMo RL,
the original Search-R1 veRL fork, current veRL, and slime. It intentionally
keeps two scoreboards:

1. **Native recipes** answer what a user gets from each published example.
2. **Protocol-aligned recipes** answer how the training systems behave when the
   model, data, search results, rollout budget, reward, and update workload are
   held constant.

Native results must not be presented as a framework ranking because the
published examples use different model sizes, prompts, turn limits, rollout
groups, rewards, and GPU layouts. Protocol-aligned results must not be
presented as untouched upstream recipes because current veRL and slime require
explicit adapters.

## Frozen implementations

| Name | Source | Frozen revision | Classification |
| --- | --- | --- | --- |
| NeMo RL | This repository | Recorded in each run manifest | Strict text-action reproduction |
| Original Search-R1 | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Upstream base `598e61bd1d36895726d28a8d06b3a15bed19f5d3`; aligned patch head `8f4c5b91e4092fb4d8e858cee0689d339aaf310f` | Paper's public veRL fork plus a disclosed comparison patch stack |
| Current veRL | [verl-project/verl](https://github.com/verl-project/verl) | Upstream base `5cfb74fa04c7f6e5d98260b8f05157c6a9402695`; measurement-only patch head `fb72e8b195095ac3334e870176eb6eaa80184001` | Current Agent Loop plus disclosed Search-R1 and measurement adapters |
| slime | [THUDM/slime](https://github.com/THUDM/slime/tree/main/examples/search-r1) | Upstream base `a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e`; measurement-only patch head `510ed5bdd9942bfb71c2b74d1928f6cefde646df` | Published `Search-R1 lite` example plus disclosed alignment and profiling adapters |

The current veRL row is not an upstream runnable Search-R1 recipe. Its main
branch retains `preprocess_search_r1_dataset.py`,
`search_r1_like_qa_em.py`, the generic Agent Loop, and an integration guide.
The guide explicitly says that the in-tree `SearchTool` was removed. The old
example launcher was removed by veRL commit `3467d90a`, and the old tool was
removed by commit `5a506cc5`. The aligned comparison will therefore add a
minimal adapter on the frozen current revision and publish that diff. A second,
frozen measurement-only diff carries scalar generated-token, observation-token,
search, error, and trajectory-time counters through TransferQueue tags. This
avoids writing full rollout text during timed steps. The manifest records the
upstream base, patch head, and diff SHA-256.

The original Search-R1 patch commits are comparison artifacts, not upstream
releases. They add timing and token counters, align the outer-step scheduler
and agent-turn boundary, export validation predictions, and make final
validation optional for short performance runs. Each manifest records both
the untouched upstream base and the exact patch head. Formal original-fork
runs use the dedicated Python 3.10/CUDA 12.1 lock at SHA-256
`11a7246f36c9ea844e14b030631d8f8b1489cd245f663a5864bc8b3074d5f269`;
resolving its Hydra config inside a current-veRL image is not runtime evidence.

## Verified native differences

| Dimension | Original Search-R1 | Current veRL historical example | slime lite |
| --- | --- | --- | --- |
| Default policy | Qwen2.5-7B base in the `v0.2` script | Qwen2.5-3B-Instruct | Qwen2.5-3B base |
| Action interface | `<search>` / `<answer>` text tags | Structured `search` function tool | `<search>` / `<answer>` text tags |
| Search rounds | Four executable rounds plus a final answer-only generation | Two assistant turns | Two generation turns and no extra final generation |
| Rollouts per prompt | Five | Five | Eight |
| Response limit | 500 generated tokens per action; 4,096 total sequence | 3,000 response tokens; 15,000 model context | 512 response tokens |
| Reward | Normalized answer exact match | Normalized answer exact match | Effective exact match; `format_score=0.2` is configured but unused by the current scorer |
| Training batch | 512 prompts, expanded to 2,560 trajectories; mini-batch 256 | 512 prompts, five rollouts; mini-batch 256 | Rollout batch 32, eight rollouts; global batch 256 |
| Rollout engine | vLLM hybrid engine | SGLang Agent Loop | SGLang custom generation |
| Trainer | Older veRL FSDP | Current veRL FSDP | Megatron-Core |

The table describes source defaults, not performance results. The current veRL
historical example is used only to identify the intended integration pattern;
the benchmark executes the frozen current revision with the disclosed adapter.

## Protocol-aligned quality contract

Every quality run must satisfy all fields below before its result enters the
aligned scoreboard.

| Dimension | Required value |
| --- | --- |
| Initial policy | `Qwen/Qwen2.5-7B` base at revision `d149729398750b98c0af14eb82c78cfe92750796` |
| Train questions | NQ + HotpotQA, 169,615 rows; canonical JSONL SHA-256 `9904042da053be8e7fa275453c9221324d24aadb6f67323d040e6016da9bfaff`; shared framework Parquet SHA-256 `64325c44a1ac79c53fc70ad36551e34b4d2ac0fa79cf0d3cca1c4d244bdeaa39` |
| Evaluation questions | Seven-source Search-R1 test set, 51,713 rows; canonical JSONL SHA-256 `bdcc57b4c3e88241bf7144f4e739c991c7a0ace4cd1ea26b6c602e2655445645`; shared framework Parquet SHA-256 `7c7d10d003dce8b0c6c2c0c4177974d0767cd2a380123faf6ee51473bc8e2461` |
| Cross-format equivalence | Every ordered composite ID, source, question, prompt, answer list, and reward ground truth is identical between the NeMo JSONL and external-framework Parquet views; train semantic SHA-256 `fb30832270c8d5074a240d4ea1e8257d8078287523cad6f3d6bbfb6471f30839`, evaluation `65e58c1eb660b6162468483ac4489ed3fd1c06d87a83ead2d16f65337d5233d2` |
| Training order | Shared deterministic Parquet row order with framework-internal training shuffle disabled; any epoch wrap must preserve that order |
| Retriever | `intfloat/e5-base-v2` revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, float16 GPU FlatIP, top 3 |
| Wikipedia | 21,015,324 Wiki18 documents; corpus SHA-256 `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d` |
| E5 index | 21,015,324 x 768 inner-product vectors; SHA-256 `69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166` |
| Prompt and actions | Case-sensitive Search-R1 prompt and `<search>`, `<information>`, `<answer>` text protocol |
| Action budget | Four executable generations, followed by one answer-only generation when unfinished |
| Generated/action limits | At most 500 new tokens per generation and 4,096 tokens in the training sequence |
| Sampling | Temperature 1.0, top-p 1.0; exact seed recorded |
| Validation sampling | One greedy response per question (the original Search-R1 `do_sample=False` behavior) |
| Group | Five trajectories per prompt |
| Reward | Last `<answer>` only; normalized exact match in `{0, 1}`; no format or retrieval shaping |
| Loss mask | All retrieved observations excluded from policy loss |
| Advantage | GRPO group mean and sample standard deviation; no leave-one-out baseline |
| Objective | KL coefficient 0.001, low-variance KL, symmetric ratio clip 0.2, no entropy bonus |
| Optimizer | AdamW, learning rate 1e-6, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8; 142 outer-step scheduler advances of linear warm-up, followed by a constant learning rate |
| Outer step | 512 questions x five trajectories |
| Optimizer mini-batch | 256 trajectories, ten updates per outer step |
| Campaign | 500 completed outer steps; validate every 50; save at steps 250 and 500 |

The retrieval service is shared when network topology permits. If a framework
needs a separate service process, a pre-run query suite must prove identical
document IDs and ordering. Any failed search, silently skipped trajectory,
changed prompt, different model revision, or incomplete optimizer update makes
that run ineligible until it is rerun or explicitly reported outside the
aligned scoreboard.

The public recipe saves every 100 steps. This comparison instead keeps only
the midpoint and final checkpoints for every framework. Checkpoint cadence is
outside the timed systems window and does not change the optimizer, scheduler,
data order, or fixed step-500 headline result.

### Required adapter disclosures

- **Original Search-R1:** retain the public text-action loop and reward. Add
  timing, batch-correlation headers, local metrics, sampled spans, and artifact
  export without changing generated tokens or loss masks.
- **Current veRL:** implement the exact text-action state machine on the current
  Agent Loop API. Restoring the deleted structured `SearchTool` is sufficient
  for a native demonstration but not for the strict scoreboard because its
  function-call syntax and chat template differ from Search-R1. Its disclosed
  measurement patch adds only scalar work counters to native step metrics; the
  timed launcher disables `rollout_data_dir` so observability does not add a
  full-text rollout dump that the other frameworks do not perform.
- **slime:** change 3B to the frozen 7B base model, eight rollouts to five and
  two turns to four plus the final answer-only generation. Make the exact-match
  scorer consume the generated response directly: the current scorer ignores
  its configured `format_score=0.2` and requires two `<answer>` blocks because
  it receives `prompt + response`, relying on the example answer in the prompt.
  Preserve SGLang token IDs/log probabilities and the observation loss mask.
  Colocate both logical actor and rollout pools on the same eight physical
  GPUs, and record all three counts rather than presenting the logical pools
  as additive. Use slime's public
  before-train-step hook to hold LR constant across the ten optimizer
  mini-batches in one outer step, matching the other three schedulers; the hook
  does not change gradients or optimizer work. A separate, frozen
  measurement-only patch brackets one outer step on actor rank zero with an
  NVTX range. The profile launcher enables dynamic-message matching for
  PyTorch's NVTX marker and starts collection at that range. It ignores the
  range end as a collection-stop trigger. After the Ray job succeeds, the
  launcher-side wrapper gives each worker session a unique job/PID-derived
  name; the outer launcher discovers and stops the sole active prefixed
  session before shutting Ray down, because ending a full-step collection
  inside `range_pop()` can deadlock the Ray actor in Nsight/CUPTI and waiting
  on Ray's reaped worker launcher can leave Nsight attached to a zombie. The
  named range still
  identifies the exact target step, but aggregate Nsight tables also contain
  later rank-zero activity and are labeled accordingly. The launcher keeps
  Ray alive until the rank-zero report is ready. Other actor ranks and the
  separate SGLang rollout engines are explicitly recorded as missing; clean
  and baseline runs never enter that branch. Full debug rollout serialization
  is campaign-only and is disabled in smoke and timed performance runs.
- **NeMo RL:** use `grpo_qwen2_5_7b_search_r1.yaml` with training shuffle
  disabled for the four-way aligned campaign; the paper-reproduction recipe
  may retain its native shuffle setting. Any diagnostic micro-batch override
  is a systems preflight, not a quality result. The sync trainer prints the
  already-computed mean generated, environment-observation, and complete
  response token counts so baseline and observed runs expose the same scalar
  work without serializing trajectories.

## Quality and accuracy reporting

Checkpoint selection is fixed before evaluation. The primary result is the
step-500 checkpoint; the best validation checkpoint may be shown separately
but cannot replace it silently. All checkpoints are evaluated by the same
offline evaluator over the same examples, rather than trusting four native log
formats.

The report includes:

- normalized exact match and answer F1, overall and per source;
- bootstrap 95% confidence intervals and paired per-question differences;
- training reward and held-out EM curves against steps, trajectories, generated
  tokens, GPU-hours, and wall time;
- retrieval answer recall, search error rate, format validity, natural
  termination, searches per question, and repeated-query rate;
- success and failure counts, including aborted, truncated, retried, or skipped
  trajectories;
- a fixed, seed-selected qualitative sample classified as direct answer,
  useful search, irrelevant search, repeated search, post-retrieval correction,
  format failure, truncation, or wrong answer.

The official `id` column is only unique inside each source: the 51,713-row test
set has just 26,843 distinct raw IDs. All common artifacts therefore use
`{data_source}:{id}` as `example_id`; this produces 51,713 distinct IDs while
retaining the untouched raw `id` column. Framework adapters must carry this ID
through metadata. Joining predictions by row order or by raw `id` alone is
invalid.

At least one common seed is mandatory. Three seeds are preferred when the
approved compute budget permits them. A single-seed result is labeled as such
and does not support small quality rankings without paired uncertainty.

## Clean end-to-end performance contract

Performance runs are separate from 500-step quality training. They start from
the same frozen policy and use an identical ordered prompt batch and retrieval
service. Each implementation performs at least one warm-up update followed by
three measured updates. The clean run enables low-overhead logging only:
TensorBoard plus local SwanLab, one-second GPU/host sampling, Prometheus, and a
fixed sampled trajectory trace. Nsight Systems runs in a separate short job and
never supplies the headline end-to-end time.

The formal clean workload is fixed as follows:

| Dimension | Required value |
| --- | --- |
| Training hardware | One identical node with eight H100 PCIe GPUs; exact SKU, memory, clocks, power limit, driver, and topology recorded |
| Retrieval hardware | One separate identical retrieval GPU and the same network path for every framework |
| Prompt batch | The same ordered eight questions, identified by composite `example_id` |
| Rollout group | Five trajectories per question, exactly 40 requested and completed trajectories per outer step |
| Update window | Four completed outer steps: step 1 warm-up, steps 2-4 measured |
| Optimizer work | One optimizer update over all 40 trajectories per outer step; no filtering, retries, or silent replacement |
| Actor micro-batch | Five trajectories per GPU for NeMo RL and both veRL implementations; slime uses its native dynamic packing with a recorded 9,216-token/GPU cap |
| Initialization | Model load, engine construction, corpus/index load, compilation, and first health check reported separately |
| Excluded work | Validation and checkpoint writing disabled in the measured window |
| Sampling and protocol | The quality-contract values above, including seed, action limits, reward, masks, and E5 top 3 |

The three steady steps are intentionally small enough to run in every public
implementation and large enough to expose rollout/trainer transitions. They do
not estimate trained quality. A row with a different GPU SKU, prompt order,
trajectory count, optimizer-update count, retriever path, or instrument set is
diagnostic-only and cannot enter the headline performance table.

The result reports both wall time and actual work:

- completed samples/s, model-generated tokens/s, response-sequence tokens/s,
  processed training-sequence tokens/s where the framework exposes the latter,
  and GPU-hours per million generated and processed tokens;
- prompt, generated, retrieved-observation, padding, and loss-mask token counts;
- median, p95, mean, maximum, and dispersion across steady steps;
- peak and mean training/retrieval GPU memory and utilization, power, host RSS,
  CPU load, network traffic, retrieval QPS, and retrieval latency;
- initialization, validation, and checkpoint costs outside the core update.

Those token rates are not interchangeable. Model-generated tokens exclude E5
observations. Response-sequence tokens include generated tokens plus injected
observations. Processed training-sequence tokens include the prompt and
response tokens traversed by the training stack. The analyzer reconstructs
each rate from its classified per-step work counter and the same end-to-end
step boundary. Framework-native throughput labels remain in a separate
`native_throughput` section; in particular, current veRL's
`perf/total_num_tokens` is prompt plus response, while slime's
`perf/tokens_per_gpu_per_sec` includes observation tokens. Missing categories
remain missing rather than being inferred as zero.

### Non-overlapping critical-path buckets

Every measured step maps into the following top-level buckets, which must sum
to 100% of step wall time:

1. rollout preparation and policy-to-engine synchronization;
2. model generation and multi-turn agent scheduling;
3. retrieval wait on the critical path;
4. trajectory post-processing, reward, and advantage;
5. policy log probability;
6. reference log probability;
7. training preparation;
8. actor forward, backward, gradient communication, and optimizer step;
9. other synchronization, idle, and unclassified time.

Nested retrieval stages (queue, encode, Faiss search, document fetch,
serialization, and network), CUDA kernels, NCCL, and concurrent spans are also
reported as inclusive evidence. They are not added again to the top-level
percentages. The per-trajectory timeline and shared provider batch ID identify
whether retrieval is on the critical path or overlapped with other rollouts.

## Four observability layers

1. **TensorBoard and local SwanLab** provide step quality, throughput, token,
   trainer, and resource trends without requiring a cloud account.
2. **Nsight Systems** provides CUDA, NCCL, NVTX, CPU scheduling, memory, and
   kernel gaps for a short steady-state window.
3. **Prometheus** provides bounded-cardinality request, queue, active-batch,
   latency, KV-cache, preemption, token, and error series from rollout and
   retrieval services where the native engine exports them.
4. **Trajectory JSONL spans** join model calls, actions, queues, HTTP batches,
   and E5 stages by trajectory and provider batch IDs.

The trajectory analyzer reports both the sum of per-request span durations and
the wall-clock union of concurrent spans, plus model/retrieval overlap. This
prevents forty concurrent requests from being misread as forty times the
critical-path time. These unions are still inclusive, and sampled clean traces
are not promoted to exact top-level buckets.

Instrumentation overhead is measured by an otherwise identical on/off pair.
If a framework cannot expose an internal span, the report marks that field
missing and retains the common external wall-time boundary. Missing evidence is
never replaced by a zero-duration stage.

Every strict launcher uses `SEARCH_R1_OBSERVABILITY_MODE=baseline|clean|profile`.
Baseline disables the optional local dashboard and common trajectory writer;
clean enables the framework's local metric sink and the run-mode sampling rate;
profile is a separate run with a 100% default trajectory sample rate and Nsight
capture. The same mode names therefore have the same measurement meaning even
when a framework's native logger or profiler implementation differs.

Prometheus coverage is also recorded rather than assumed. In clean and profile
modes, NeMo RL exposes and scrapes every native vLLM HTTP server, current veRL
enables vLLM statistics and scrapes every server announced by its
`LLMServerManager`, and slime scrapes both the SGLang router and every announced
SGLang engine. slime enables SGLang metric emission natively even in baseline,
so its manifest records that emission as always on while the external scraper
remains off. The original Search-R1 fork exports the added driver/step metrics,
but its embedded old vLLM engine has no separate HTTP metrics endpoint in this
runtime; engine-level Prometheus coverage is therefore marked `missing`, not
zero. All four runs separately scrape the shared E5 service.

The Nsight process coverage is recorded, not assumed. NeMo RL and the patched
original Search-R1 cover their policy and rollout workers. Current veRL's
native Nsight path covers actor and reference workers, but its asynchronous
vLLM server explicitly supports only the Torch and NPU profilers, so its
rollout-engine Nsight scope is marked `missing`. The slime measurement patch
covers actor rank zero while its other actor ranks and separate SGLang rollout
engines are marked `missing`. Common trajectory spans and external resource samples remain
available for those rollout stages; a missing GPU timeline is never reported
as zero work.

## Result validity checklist

A row enters the final four-way table only when its manifest and artifacts
prove all of the following:

- frozen source and adapter commits, resolved dependency lock, container, full
  command, hardware, topology, and environment variables;
- identical model/data/retrieval hashes and protocol values;
- exact requested and completed trajectory/update counts with zero silent drops;
- clean-run and Nsight artifacts kept separate;
- raw step metrics, resource samples, Prometheus snapshots, trajectory spans,
  checkpoint/evaluation outputs, and representative decoded trajectories;
- a machine-readable summary using common units and explicit missing fields.

Until all four rows pass these gates, partial measurements are reported as
diagnostics and the comparison remains incomplete.
