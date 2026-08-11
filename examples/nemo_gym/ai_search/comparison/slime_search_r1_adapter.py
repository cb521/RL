# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict Search-R1 text-action rollout implemented on slime's public hook."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from framework_observability import trace_span as common_trace_span


MAX_EXECUTABLE_TURNS = 4
MAX_ACTION_TOKENS = 500
MAX_OBSERVATION_TOKENS = 500
MAX_START_TOKENS = 2048
MAX_RESPONSE_TOKENS = 4096
TOP_K = 3
_ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_INVALID_OBSERVATION = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between "
    "<answer> and </answer>. Let me try again.\n"
)
_SEARCH_SEMAPHORE: asyncio.Semaphore | None = None
_SEARCH_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


def _runtime() -> SimpleNamespace:
    """Load slime lazily so pure protocol tests do not require its GPU stack."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.trace_utils import (
        bind_trace,
        build_sglang_meta_trace_attrs,
        trace_span,
    )
    from slime.utils.types import Sample

    return SimpleNamespace(
        GenerateState=GenerateState,
        Sample=Sample,
        bind_trace=bind_trace,
        build_sglang_meta_trace_attrs=build_sglang_meta_trace_attrs,
        post=post,
        trace_span=trace_span,
    )


def adapter_contract() -> dict[str, Any]:
    """Return the fixed protocol values for run manifests."""
    return {
        "action_protocol": "search-r1-text-v1",
        "executable_turns": MAX_EXECUTABLE_TURNS,
        "final_answer_only_generation": True,
        "max_action_tokens": MAX_ACTION_TOKENS,
        "max_observation_tokens": MAX_OBSERVATION_TOKENS,
        "max_start_tokens": MAX_START_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "retrieval_top_k": TOP_K,
        "observation_loss_mask": 0,
        "lr_scheduler_granularity": "outer_step",
    }


def align_outer_step_lr(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Keep all optimizer mini-batches in one Search-R1 outer step at one LR.

    slime advances Megatron's sample-count scheduler after every optimizer
    mini-batch. The Search-R1 veRL fork, current veRL, and NeMo RL advance their
    LR scheduler once after the full rollout group has been trained. This
    public slime hook resets the current mini-batch to the outer-step LR and
    prepositions the last mini-batch so slime's normal post-update increment
    lands on the next outer-step boundary. Optimizer work and gradients are not
    changed.
    """
    del model
    outer_batch_size = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
    optimizer_batch_size = int(args.global_batch_size)
    if outer_batch_size <= 0 or optimizer_batch_size <= 0:
        raise ValueError("Search-R1 scheduler batch sizes must be positive")
    if outer_batch_size % optimizer_batch_size:
        raise ValueError(
            "Search-R1 outer trajectories must divide exactly into optimizer batches"
        )
    updates_per_outer_step = outer_batch_size // optimizer_batch_size
    if not 0 <= int(step_id) < updates_per_outer_step:
        raise ValueError(
            f"Unexpected slime optimizer step {step_id}; expected "
            f"0..{updates_per_outer_step - 1}"
        )
    if int(rollout_id) < 0:
        raise ValueError("Search-R1 rollout_id must be non-negative")
    if getattr(opt_param_scheduler, "optimizer", optimizer) is not optimizer:
        raise ValueError("Search-R1 scheduler and optimizer do not match")

    outer_start = int(rollout_id) * outer_batch_size
    opt_param_scheduler.num_steps = outer_start
    current_values = [
        (
            opt_param_scheduler.get_lr(param_group),
            opt_param_scheduler.get_wd(param_group)
            * param_group.get("wd_mult", 1.0),
        )
        for param_group in optimizer.param_groups
    ]

    # train_one_step() calls scheduler.step(optimizer_batch_size) after the
    # optimizer update. On the last mini-batch, place the scheduler one update
    # before the next outer boundary so its persisted state is also exact.
    if int(step_id) == updates_per_outer_step - 1:
        opt_param_scheduler.num_steps = (
            outer_start + outer_batch_size - optimizer_batch_size
        )
    for param_group, (learning_rate, weight_decay) in zip(
        optimizer.param_groups, current_values, strict=True
    ):
        param_group["lr"] = learning_rate
        param_group["weight_decay"] = weight_decay


def parse_action(response: str) -> tuple[str | None, str]:
    """Return the first executable Search-R1 action in a model response."""
    match = _ACTION_PATTERN.search(response)
    if match is None:
        return None, ""
    return match.group(1), match.group(2).strip()


def format_passages(raw_hits: Any) -> str:
    """Format one E5 result row exactly like the original Search-R1 loop."""
    if not isinstance(raw_hits, list):
        raise ValueError("retriever result row must be a list")
    passages = []
    for rank, raw_hit in enumerate(raw_hits, start=1):
        if not isinstance(raw_hit, Mapping):
            raise ValueError("retriever hits must be mappings")
        document = raw_hit.get("document", raw_hit)
        if not isinstance(document, Mapping):
            raise ValueError("retriever hit document must be a mapping")
        contents = document.get("contents")
        if not isinstance(contents, str):
            raise ValueError("retriever document.contents must be a string")
        title, separator, text = contents.partition("\n")
        if not separator:
            text = ""
        passages.append(f"Doc {rank}(Title: {title}) {text}\n")
    return "".join(passages)


def _search_url(args: Any) -> str:
    value = (
        getattr(args, "search_r1_retriever_url", None)
        or os.environ.get("SEARCH_R1_RETRIEVER_URL")
        or os.environ.get("AI_SEARCH_RETRIEVER_URL")
        or "http://127.0.0.1:8000/retrieve"
    )
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ValueError("Search-R1 retriever URL must be HTTP(S)")
    return value


def _search_semaphore() -> asyncio.Semaphore:
    global _SEARCH_SEMAPHORE, _SEARCH_SEMAPHORE_LOOP
    loop = asyncio.get_running_loop()
    if _SEARCH_SEMAPHORE is None or _SEARCH_SEMAPHORE_LOOP is not loop:
        concurrency = int(os.environ.get("SEARCH_R1_SEARCH_CONCURRENCY", "256"))
        if concurrency <= 0:
            raise ValueError("SEARCH_R1_SEARCH_CONCURRENCY must be positive")
        _SEARCH_SEMAPHORE = asyncio.Semaphore(concurrency)
        _SEARCH_SEMAPHORE_LOOP = loop
    return _SEARCH_SEMAPHORE


def _routing_headers(args: Any, sample: Any) -> dict[str, str] | None:
    session_id = getattr(sample, "session_id", None)
    if session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": str(session_id)}
    return None


def _finish_reason(output: Any) -> str:
    if not isinstance(output, Mapping):
        raise ValueError("SGLang output must be a mapping")
    meta = output.get("meta_info")
    if not isinstance(meta, Mapping):
        raise ValueError("SGLang output.meta_info must be a mapping")
    finish = meta.get("finish_reason")
    if isinstance(finish, Mapping):
        finish = finish.get("type")
    if finish not in {"stop", "length", "abort"}:
        raise ValueError(f"Unsupported SGLang finish reason: {finish!r}")
    return finish


def _output_tokens(output: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    meta = output["meta_info"]
    token_logprobs = meta.get("output_token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError(
            "SGLang output_token_logprobs are required for aligned slime training"
        )
    token_ids: list[int] = []
    logprobs: list[float] = []
    for item in token_logprobs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("Malformed SGLang output_token_logprobs entry")
        logprobs.append(float(item[0]))
        token_ids.append(int(item[1]))
    return token_ids, logprobs


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _append_recorded(
    args: Any,
    sample: Any,
    tokenizer: Any,
    *,
    token_ids: list[int],
    text: str,
    trainable: bool,
    logprobs: list[float] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> bool:
    """Append up to the original fork's 4,096 response-token cap."""
    remaining = MAX_RESPONSE_TOKENS - int(sample.response_length)
    if remaining <= 0:
        return bool(token_ids)
    kept_ids = token_ids[:remaining]
    was_truncated = len(kept_ids) != len(token_ids)
    if not kept_ids:
        return was_truncated
    kept_logprobs = logprobs[:remaining] if logprobs is not None else None
    kept_text = text if not was_truncated else _decode(tokenizer, kept_ids)
    sample.append_response_tokens(
        args,
        tokens=kept_ids,
        log_probs=kept_logprobs,
        trainable=trainable,
        meta_info=meta_info if not was_truncated else None,
        text=kept_text,
    )
    return was_truncated


async def _model_turn(
    runtime: SimpleNamespace,
    args: Any,
    sample: Any,
    context_ids: list[int],
    sampling_params: dict[str, Any],
    turn: int,
    trace_id: str,
) -> tuple[dict[str, Any], list[int], list[float]]:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": context_ids[-MAX_RESPONSE_TOKENS:],
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    headers = _routing_headers(args, sample)
    with common_trace_span(
        trace_id=trace_id,
        component="slime",
        operation="model_generation",
        attributes={"turn": turn, "max_new_tokens": MAX_ACTION_TOKENS},
    ) as common_span:
        with runtime.trace_span(
            sample,
            "search_r1_model_generation",
            attrs={"turn": turn, "max_new_tokens": MAX_ACTION_TOKENS},
        ) as span:
            output = await runtime.post(url, payload, headers=headers)
            span.update(runtime.build_sglang_meta_trace_attrs(output["meta_info"]))
        common_span.set_attributes(
            output_tokens=len(output["meta_info"].get("output_token_logprobs", [])),
            stop_reason=_finish_reason(output),
        )
    if _finish_reason(output) == "abort":
        return output, [], []
    token_ids, logprobs = _output_tokens(output)
    return output, token_ids, logprobs


async def _retrieve(
    runtime: SimpleNamespace,
    args: Any,
    sample: Any,
    query: str,
    turn: int,
    trace_id: str,
) -> tuple[str, str, float]:
    provider_batch_id = uuid.uuid4().hex
    payload = {"queries": [query], "topk": TOP_K, "return_scores": True}
    started = time.perf_counter()
    with common_trace_span(
        trace_id=trace_id,
        component="slime",
        operation="retrieval",
        attributes={
            "turn": turn,
            "top_k": TOP_K,
            "provider_batch_id": provider_batch_id,
            "query_characters": len(query),
        },
    ) as common_span:
        with runtime.trace_span(
            sample,
            "search_r1_retrieval",
            attrs={
                "turn": turn,
                "top_k": TOP_K,
                "provider_batch_id": provider_batch_id,
            },
        ) as span:
            async with _search_semaphore():
                result = await runtime.post(
                    _search_url(args),
                    payload,
                    max_retries=1,
                    headers={"X-NeMo-Search-Batch-ID": provider_batch_id},
                )
            elapsed = time.perf_counter() - started
            span.update(
                {
                    "provider_batch_id": provider_batch_id,
                    "latency_ms": elapsed * 1000.0,
                }
            )
        common_span.set_attributes(latency_ms=elapsed * 1000.0)
    rows = result.get("result") if isinstance(result, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("retriever response must contain one result row")
    observation = f"\n\n<information>{format_passages(rows[0]).strip()}</information>\n\n"
    return observation, provider_batch_id, elapsed


def _sampling_params(source: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(source)
    params["max_new_tokens"] = MAX_ACTION_TOKENS
    params["no_stop_trim"] = True
    existing_stop = params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    params["stop"] = list(
        dict.fromkeys([*existing_stop, "</search>", "</answer>"])
    )
    return params


async def generate(args: Any, sample: Any, sampling_params: Mapping[str, Any]) -> Any:
    """Generate four executable turns and one search-disabled final turn."""
    if getattr(args, "partial_rollout", False):
        raise ValueError("Strict Search-R1 does not support partial rollouts")
    runtime = _runtime()
    if not isinstance(sample, runtime.Sample):
        raise TypeError("sample must be a slime Sample")
    if not isinstance(sample.prompt, str):
        raise TypeError("aligned Search-R1 requires a text prompt after templating")

    state = runtime.GenerateState(args)
    prompt_ids = state.tokenizer(sample.prompt, add_special_tokens=False)[
        "input_ids"
    ][-MAX_START_TOKENS:]
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None

    metadata = dict(sample.metadata) if isinstance(sample.metadata, Mapping) else {}
    trace_id = str(runtime.bind_trace(sample).trace_id)
    metadata["observability_trace_id"] = trace_id
    sample.metadata = metadata

    context_ids = list(prompt_ids)
    params = _sampling_params(sampling_params)
    terminated = False
    record_truncated = False
    search_count = 0
    invalid_actions = 0
    provider_batch_ids: list[str] = []
    turns = 0

    for turn in range(1, MAX_EXECUTABLE_TURNS + 1):
        turns = turn
        output, token_ids, logprobs = await _model_turn(
            runtime, args, sample, context_ids, params, turn, trace_id
        )
        if _finish_reason(output) == "abort":
            sample.status = runtime.Sample.Status.ABORTED
            return sample
        response_text = output.get("text")
        if not isinstance(response_text, str):
            raise ValueError("SGLang output.text must be a string")
        context_ids.extend(token_ids)
        record_truncated |= _append_recorded(
            args,
            sample,
            state.tokenizer,
            token_ids=token_ids,
            text=response_text,
            trainable=True,
            logprobs=logprobs,
            meta_info=dict(output["meta_info"]),
        )

        action, content = parse_action(response_text)
        if action == "answer":
            terminated = True
            break
        if action == "search":
            observation, provider_batch_id, retrieval_seconds = await _retrieve(
                runtime, args, sample, content, turn, trace_id
            )
            search_count += 1
            provider_batch_ids.append(provider_batch_id)
            sample.non_generation_time += retrieval_seconds
        else:
            observation = _INVALID_OBSERVATION
            invalid_actions += 1
        observation_ids = state.tokenizer(
            observation, add_special_tokens=False
        )["input_ids"][:MAX_OBSERVATION_TOKENS]
        observation = _decode(state.tokenizer, observation_ids)
        context_ids.extend(observation_ids)
        record_truncated |= _append_recorded(
            args,
            sample,
            state.tokenizer,
            token_ids=observation_ids,
            text=observation,
            trainable=False,
        )

    if not terminated:
        turns = MAX_EXECUTABLE_TURNS + 1
        output, token_ids, logprobs = await _model_turn(
            runtime, args, sample, context_ids, params, turns, trace_id
        )
        if _finish_reason(output) == "abort":
            sample.status = runtime.Sample.Status.ABORTED
            return sample
        response_text = output.get("text")
        if not isinstance(response_text, str):
            raise ValueError("SGLang output.text must be a string")
        record_truncated |= _append_recorded(
            args,
            sample,
            state.tokenizer,
            token_ids=token_ids,
            text=response_text,
            trainable=True,
            logprobs=logprobs,
            meta_info=dict(output["meta_info"]),
        )
        action, _ = parse_action(response_text)
        terminated = action == "answer"

    metadata = dict(sample.metadata) if isinstance(sample.metadata, Mapping) else {}
    metadata.update(
        {
            "search_count": search_count,
            "invalid_action_count": invalid_actions,
            "turn_count": turns,
            "natural_termination": terminated,
            "search_errors": 0,
            "provider_batch_ids": provider_batch_ids,
            "round_number": turns,
        }
    )
    sample.metadata = metadata
    if record_truncated:
        sample.status = runtime.Sample.Status.TRUNCATED
    elif terminated:
        sample.status = runtime.Sample.Status.COMPLETED
    else:
        sample.status = runtime.Sample.Status.FAILED
    return sample
