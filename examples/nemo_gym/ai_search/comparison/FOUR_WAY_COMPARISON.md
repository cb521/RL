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
| Original Search-R1 | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | `2d0e225716fe3ccc071c9d020f5561548fdefc54` | Paper's public veRL fork |
| Current veRL | [verl-project/verl](https://github.com/verl-project/verl) | `5cfb74fa04c7f6e5d98260b8f05157c6a9402695` | Current Agent Loop plus a disclosed Search-R1 adapter |
| slime | [THUDM/slime](https://github.com/THUDM/slime/tree/main/examples/search-r1) | `a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e` | Published `Search-R1 lite` example plus a disclosed aligned recipe |

The current veRL row is not an upstream runnable Search-R1 recipe. Its main
branch retains `preprocess_search_r1_dataset.py`,
`search_r1_like_qa_em.py`, the generic Agent Loop, and an integration guide.
The guide explicitly says that the in-tree `SearchTool` was removed. The old
example launcher was removed by veRL commit `3467d90a`, and the old tool was
removed by commit `5a506cc5`. The aligned comparison will therefore add a
minimal adapter on the frozen current revision and publish that diff.

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
| Train questions | NQ + HotpotQA, 169,615 rows, converted SHA-256 `249fca5d8251e3e37a30e987433181f9fe4b65be84af48863d011fdf3b338f13` |
| Evaluation questions | Seven-source Search-R1 test set, 51,713 rows, converted SHA-256 `0ca2dd62c567e70a9b1d336f2f183f310b4e33a98bfcb664bcca37c326825801` |
| Retriever | `intfloat/e5-base-v2` revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, float16 GPU FlatIP, top 3 |
| Wikipedia | 21,015,324 Wiki18 documents; corpus SHA-256 `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d` |
| E5 index | 21,015,324 x 768 inner-product vectors; SHA-256 `69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166` |
| Prompt and actions | Case-sensitive Search-R1 prompt and `<search>`, `<information>`, `<answer>` text protocol |
| Action budget | Four executable generations, followed by one answer-only generation when unfinished |
| Generated/action limits | At most 500 new tokens per generation and 4,096 tokens in the training sequence |
| Sampling | Temperature 1.0, top-p 1.0; exact seed recorded |
| Group | Five trajectories per prompt |
| Reward | Last `<answer>` only; normalized exact match in `{0, 1}`; no format or retrieval shaping |
| Loss mask | All retrieved observations excluded from policy loss |
| Advantage | GRPO group mean and sample standard deviation; no leave-one-out baseline |
| Objective | KL coefficient 0.001, low-variance KL, symmetric ratio clip 0.2, no entropy bonus |
| Optimizer | AdamW, learning rate 1e-6, weight decay 0.01, 28.5% linear warm-up |
| Outer step | 512 questions x five trajectories |
| Optimizer mini-batch | 256 trajectories, ten updates per outer step |
| Campaign | 500 completed outer steps; validate every 50; save every 100 |

The retrieval service is shared when network topology permits. If a framework
needs a separate service process, a pre-run query suite must prove identical
document IDs and ordering. Any failed search, silently skipped trajectory,
changed prompt, different model revision, or incomplete optimizer update makes
that run ineligible until it is rerun or explicitly reported outside the
aligned scoreboard.

### Required adapter disclosures

- **Original Search-R1:** retain the public text-action loop and reward. Add
  timing, batch-correlation headers, and artifact export without changing
  generated tokens or loss masks.
- **Current veRL:** implement the exact text-action state machine on the current
  Agent Loop API. Restoring the deleted structured `SearchTool` is sufficient
  for a native demonstration but not for the strict scoreboard because its
  function-call syntax and chat template differ from Search-R1.
- **slime:** change 3B to the frozen 7B base model, eight rollouts to five and
  two turns to four plus the final answer-only generation. Make the exact-match
  scorer consume the generated response directly: the current scorer ignores
  its configured `format_score=0.2` and requires two `<answer>` blocks because
  it receives `prompt + response`, relying on the example answer in the prompt.
  Preserve SGLang token IDs/log probabilities and the observation loss mask.
  Record its actor/rollout GPU split rather than hiding it.
- **NeMo RL:** use `grpo_qwen2_5_7b_search_r1.yaml`; any diagnostic micro-batch
  override is a systems preflight, not a quality result.

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

The result reports both wall time and actual work:

- completed samples/s, generated tokens/s, trained tokens/s, and GPU-hours per
  million generated and trained tokens;
- prompt, generated, retrieved-observation, padding, and loss-mask token counts;
- median, p95, mean, maximum, and dispersion across steady steps;
- peak and mean training/retrieval GPU memory and utilization, power, host RSS,
  CPU load, network traffic, retrieval QPS, and retrieval latency;
- initialization, validation, and checkpoint costs outside the core update.

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
   latency, and error series from rollout and retrieval services.
4. **Trajectory JSONL spans** join model calls, actions, queues, HTTP batches,
   and E5 stages by trajectory and provider batch IDs.

Instrumentation overhead is measured by an otherwise identical on/off pair.
If a framework cannot expose an internal span, the report marks that field
missing and retains the common external wall-time boundary. Missing evidence is
never replaced by a zero-duration stage.

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
