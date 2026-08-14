# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict Search-R1 text-action rollout for current veRL's Agent Loop API."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from framework_observability import trace_span


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
_HTTP_SESSION: Any = None
_HTTP_SESSION_LOOP: asyncio.AbstractEventLoop | None = None
_SEARCH_SEMAPHORE: asyncio.Semaphore | None = None
_SEARCH_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


@dataclass
class ProtocolOutput:
    """Framework-neutral result consumed by the lazy veRL wrapper."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    num_turns: int
    metrics: dict[str, Any]
    extra_fields: dict[str, Any]


def _runtime() -> SimpleNamespace:
    """Load veRL lazily so protocol tests do not import its GPU stack."""
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
    from verl.utils.rollout_trace import rollout_trace_op

    return SimpleNamespace(
        AgentLoopBase=AgentLoopBase,
        AgentLoopMetrics=AgentLoopMetrics,
        AgentLoopOutput=AgentLoopOutput,
        rollout_trace_op=rollout_trace_op,
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
    }


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


def _flat_token_ids(value: Any, field_name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in value
    ):
        raise ValueError(f"{field_name} must be a flat integer token list")
    return value


async def _encode(agent: Any, text: str) -> list[int]:
    loop = getattr(agent, "loop", asyncio.get_running_loop())

    def tokenize() -> list[int]:
        encoded = agent.tokenizer(text, add_special_tokens=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer must return an input_ids mapping")
        return _flat_token_ids(encoded["input_ids"], "tokenizer input_ids")

    return await loop.run_in_executor(None, tokenize)


async def _decode(agent: Any, token_ids: list[int]) -> str:
    loop = getattr(agent, "loop", asyncio.get_running_loop())
    return await loop.run_in_executor(
        None,
        lambda: agent.tokenizer.decode(token_ids, skip_special_tokens=True),
    )


def _sampling_params(agent: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(source)
    params.pop("max_new_tokens", None)
    params["max_tokens"] = MAX_ACTION_TOKENS
    params["logprobs"] = True
    existing_stop = params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    params["stop"] = list(
        dict.fromkeys([*existing_stop, "</search>", "</answer>"])
    )

    backend = str(getattr(agent.rollout_config, "name", "")).lower()
    if backend == "vllm":
        params["include_stop_str_in_output"] = True
        params.pop("no_stop_trim", None)
    elif backend == "sglang":
        params["no_stop_trim"] = True
        params.pop("include_stop_str_in_output", None)
    else:
        raise ValueError(
            "Aligned Search-R1 supports current veRL vLLM or SGLang rollouts, "
            f"not {backend!r}"
        )
    return params


def _merge_engine_extra(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        if key == "min_global_steps" and value is not None:
            current = target.get(key)
            target[key] = value if current is None else min(current, value)
        elif key == "max_global_steps" and value is not None:
            current = target.get(key)
            target[key] = value if current is None else max(current, value)
        elif key.startswith("spec_num_") and value is not None:
            target[key] = int(target.get(key, 0)) + int(value)
        else:
            target.setdefault(key, value)


def _append_recorded(
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    *,
    token_ids: list[int],
    mask: int,
    logprobs: list[float],
) -> bool:
    remaining = MAX_RESPONSE_TOKENS - len(response_ids)
    if remaining <= 0:
        return bool(token_ids)
    kept_ids = token_ids[:remaining]
    kept_logprobs = logprobs[:remaining]
    response_ids.extend(kept_ids)
    response_mask.extend([mask] * len(kept_ids))
    response_logprobs.extend(kept_logprobs)
    return len(kept_ids) != len(token_ids)


async def _generate_turn(
    agent: Any,
    request_id: str,
    context_ids: list[int],
    sampling_params: dict[str, Any],
    priority: int,
    turn: int,
) -> tuple[Any, list[int], list[float], str, float]:
    started = time.perf_counter()
    with trace_span(
        trace_id=request_id,
        component="current_verl",
        operation="model_generation",
        attributes={"turn": turn, "max_new_tokens": MAX_ACTION_TOKENS},
    ) as span:
        output = await agent.server_manager.generate(
            request_id=request_id,
            prompt_ids=context_ids[-MAX_RESPONSE_TOKENS:],
            sampling_params=dict(sampling_params),
            priority=priority,
        )
        elapsed = time.perf_counter() - started
        span.set_attributes(
            output_tokens=len(output.token_ids),
            stop_reason=output.stop_reason,
        )
    token_ids = _flat_token_ids(output.token_ids, "TokenOutput.token_ids")
    if output.stop_reason == "aborted":
        return output, token_ids, [], "", elapsed
    if output.routed_experts is not None:
        raise NotImplementedError(
            "The aligned dense Qwen2.5-7B recipe does not use routed experts; "
            "a MoE adapter must align routing rows across all turns explicitly"
        )
    if output.log_probs is None:
        raise RuntimeError(
            "Selected-token rollout log probabilities are required for aligned veRL"
        )
    logprobs = list(output.log_probs)
    if len(logprobs) != len(token_ids):
        raise ValueError("TokenOutput.log_probs must align with token_ids")
    response_text = await _decode(agent, token_ids)
    return output, token_ids, [float(value) for value in logprobs], response_text, elapsed


def _finalize(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    turns: int,
    generation_seconds: float,
    retrieval_seconds: float,
    wall_seconds: float,
    num_preempted: int,
    search_count: int,
    invalid_actions: int,
    provider_batch_ids: list[str],
    terminated: bool,
    record_truncated: bool,
    engine_extra: dict[str, Any],
    status_override: str | None = None,
) -> ProtocolOutput:
    if status_override is not None:
        status = status_override
    elif record_truncated:
        status = "truncated"
    elif terminated:
        status = "completed"
    else:
        status = "failed"
    extra_fields = dict(engine_extra)
    extra_fields.update(
        {
            "status": status,
            "search_count": search_count,
            "invalid_actions": invalid_actions,
            "turn_count": turns,
            "natural_termination": terminated,
            "search_errors": 0,
            "provider_batch_ids": provider_batch_ids,
            "generated_tokens": sum(response_mask),
            "observation_tokens": len(response_mask) - sum(response_mask),
            "wall_time_seconds": wall_seconds,
            "turn_scores": [],
            "tool_rewards": [],
        }
    )
    return ProtocolOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        num_turns=turns + 1,
        metrics={
            "generate_sequences": generation_seconds,
            "tool_calls": retrieval_seconds,
            "num_preempted": num_preempted,
        },
        extra_fields=extra_fields,
    )


async def run_protocol(
    agent: Any,
    sampling_params: Mapping[str, Any],
    *,
    priority: int = 0,
    **kwargs: Any,
) -> ProtocolOutput:
    """Run the frozen Search-R1 state machine against a duck-typed veRL agent."""
    started = time.perf_counter()
    if getattr(agent, "enable_continuous_token", False):
        raise ValueError("Aligned Search-R1 requires continuous_token.enable=false")
    messages = kwargs.get("raw_prompt")
    if not isinstance(messages, (list, tuple)):
        raise TypeError("raw_prompt must be a list of chat messages")
    prompt_ids = _flat_token_ids(
        await agent.apply_chat_template(list(messages)),
        "chat-template prompt_ids",
    )[-MAX_START_TOKENS:]
    context_ids = list(prompt_ids)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    params = _sampling_params(agent, sampling_params)
    full_determinism = bool(getattr(agent.rollout_config, "full_determinism", False))
    request_id = f"det-{int(priority)}" if full_determinism else uuid.uuid4().hex

    generation_seconds = 0.0
    retrieval_seconds = 0.0
    search_count = 0
    invalid_actions = 0
    provider_batch_ids: list[str] = []
    turns = 0
    num_preempted = 0
    have_preemption_metric = False
    terminated = False
    record_truncated = False
    engine_extra: dict[str, Any] = {}

    async def generate() -> tuple[Any, list[int], list[float], str]:
        nonlocal generation_seconds, num_preempted, have_preemption_metric
        output, token_ids, logprobs, response_text, elapsed = await _generate_turn(
            agent,
            request_id,
            context_ids,
            params,
            int(priority),
            turns,
        )
        generation_seconds += elapsed
        if output.num_preempted is not None:
            num_preempted += int(output.num_preempted)
            have_preemption_metric = True
        _merge_engine_extra(engine_extra, output.extra_fields)
        return output, token_ids, logprobs, response_text

    for turn in range(1, MAX_EXECUTABLE_TURNS + 1):
        turns = turn
        output, token_ids, logprobs, response_text = await generate()
        if output.stop_reason == "aborted":
            return _finalize(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                turns=turns,
                generation_seconds=generation_seconds,
                retrieval_seconds=retrieval_seconds,
                wall_seconds=time.perf_counter() - started,
                num_preempted=num_preempted if have_preemption_metric else -1,
                search_count=search_count,
                invalid_actions=invalid_actions,
                provider_batch_ids=provider_batch_ids,
                terminated=False,
                record_truncated=record_truncated,
                engine_extra=engine_extra,
                status_override="aborted",
            )
        context_ids.extend(token_ids)
        record_truncated |= _append_recorded(
            response_ids,
            response_mask,
            response_logprobs,
            token_ids=token_ids,
            mask=1,
            logprobs=logprobs,
        )
        action, content = parse_action(response_text)
        if action == "answer":
            terminated = True
            break
        if action == "search":
            provider_batch_id = uuid.uuid4().hex
            retrieval_started = time.perf_counter()
            with trace_span(
                trace_id=request_id,
                component="current_verl",
                operation="retrieval",
                attributes={
                    "turn": turn,
                    "top_k": TOP_K,
                    "provider_batch_id": provider_batch_id,
                    "query_characters": len(content),
                },
            ) as span:
                observation = await agent.retrieve(
                    query=content,
                    provider_batch_id=provider_batch_id,
                    turn=turn,
                    trajectory_id=request_id,
                )
                elapsed = time.perf_counter() - retrieval_started
                span.set_attributes(latency_ms=elapsed * 1000.0)
            retrieval_seconds += elapsed
            search_count += 1
            provider_batch_ids.append(provider_batch_id)
        else:
            observation = _INVALID_OBSERVATION
            invalid_actions += 1
        if not isinstance(observation, str):
            raise ValueError("retrieval observation must be a string")
        observation_ids = (await _encode(agent, observation))[
            :MAX_OBSERVATION_TOKENS
        ]
        context_ids.extend(observation_ids)
        record_truncated |= _append_recorded(
            response_ids,
            response_mask,
            response_logprobs,
            token_ids=observation_ids,
            mask=0,
            logprobs=[0.0] * len(observation_ids),
        )

    if not terminated:
        turns = MAX_EXECUTABLE_TURNS + 1
        output, token_ids, logprobs, response_text = await generate()
        if output.stop_reason == "aborted":
            status_override = "aborted"
        else:
            status_override = None
            record_truncated |= _append_recorded(
                response_ids,
                response_mask,
                response_logprobs,
                token_ids=token_ids,
                mask=1,
                logprobs=logprobs,
            )
            action, _ = parse_action(response_text)
            terminated = action == "answer"
    else:
        status_override = None

    return _finalize(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        turns=turns,
        generation_seconds=generation_seconds,
        retrieval_seconds=retrieval_seconds,
        wall_seconds=time.perf_counter() - started,
        num_preempted=num_preempted if have_preemption_metric else -1,
        search_count=search_count,
        invalid_actions=invalid_actions,
        provider_batch_ids=provider_batch_ids,
        terminated=terminated,
        record_truncated=record_truncated,
        engine_extra=engine_extra,
        status_override=status_override,
    )


def _retriever_url(configured: str | None) -> str:
    value = (
        configured
        or os.environ.get("SEARCH_R1_RETRIEVER_URL")
        or os.environ.get("AI_SEARCH_RETRIEVER_URL")
        or "http://127.0.0.1:8000/retrieve"
    )
    if not value.startswith(("http://", "https://")):
        raise ValueError("Search-R1 retriever URL must be HTTP(S)")
    return value


def _search_semaphore(concurrency: int) -> asyncio.Semaphore:
    global _SEARCH_SEMAPHORE, _SEARCH_SEMAPHORE_LOOP
    loop = asyncio.get_running_loop()
    if _SEARCH_SEMAPHORE is None or _SEARCH_SEMAPHORE_LOOP is not loop:
        _SEARCH_SEMAPHORE = asyncio.Semaphore(concurrency)
        _SEARCH_SEMAPHORE_LOOP = loop
    return _SEARCH_SEMAPHORE


async def _http_session(concurrency: int, timeout_seconds: float) -> Any:
    global _HTTP_SESSION, _HTTP_SESSION_LOOP
    loop = asyncio.get_running_loop()
    if (
        _HTTP_SESSION is None
        or _HTTP_SESSION_LOOP is not loop
        or bool(getattr(_HTTP_SESSION, "closed", False))
    ):
        import aiohttp

        connector = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        _HTTP_SESSION = aiohttp.ClientSession(connector=connector, timeout=timeout)
        _HTTP_SESSION_LOOP = loop
    return _HTTP_SESSION


async def _retrieve_http(
    url: str,
    query: str,
    provider_batch_id: str,
    concurrency: int,
    timeout_seconds: float,
) -> str:
    session = await _http_session(concurrency, timeout_seconds)
    async with _search_semaphore(concurrency):
        async with session.post(
            url,
            json={"queries": [query], "topk": TOP_K, "return_scores": True},
            headers={"X-NeMo-Search-Batch-ID": provider_batch_id},
        ) as response:
            response.raise_for_status()
            result = await response.json()
    rows = result.get("result") if isinstance(result, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("retriever response must contain one result row")
    passages = format_passages(rows[0]).strip()
    return f"\n\n<information>{passages}</information>\n\n"


def build_search_r1_agent_loop(
    *args: Any,
    retriever_url: str | None = None,
    search_concurrency: int = 256,
    search_timeout_seconds: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Hydra factory loaded by current veRL's agent-loop config file."""
    if isinstance(search_concurrency, bool) or search_concurrency <= 0:
        raise ValueError("search_concurrency must be positive")
    if search_timeout_seconds <= 0:
        raise ValueError("search_timeout_seconds must be positive")
    kwargs.pop("name", None)
    runtime = _runtime()
    resolved_url = _retriever_url(retriever_url)

    class SearchR1AgentLoop(runtime.AgentLoopBase):
        def __init__(self, *loop_args: Any, **loop_kwargs: Any):
            super().__init__(*loop_args, **loop_kwargs)
            if self.processor is not None:
                raise ValueError("Aligned Search-R1 supports text-only policies")
            if self.rollout_config.prompt_length < MAX_START_TOKENS:
                raise ValueError("rollout.prompt_length must be at least 2048")
            if self.rollout_config.response_length < MAX_RESPONSE_TOKENS:
                raise ValueError("rollout.response_length must be at least 4096")

        @runtime.rollout_trace_op
        async def retrieve(
            self,
            *,
            query: str,
            provider_batch_id: str,
            turn: int,
            trajectory_id: str,
        ) -> str:
            del turn, trajectory_id
            return await _retrieve_http(
                resolved_url,
                query,
                provider_batch_id,
                search_concurrency,
                search_timeout_seconds,
            )

        @runtime.rollout_trace_op
        async def run(
            self,
            sampling_params: dict[str, Any],
            priority: int = 0,
            **run_kwargs: Any,
        ) -> Any:
            result = await run_protocol(
                self,
                sampling_params,
                priority=int(priority),
                **run_kwargs,
            )
            return runtime.AgentLoopOutput(
                prompt_ids=result.prompt_ids,
                response_ids=result.response_ids,
                response_mask=result.response_mask,
                response_logprobs=result.response_logprobs,
                num_turns=result.num_turns,
                metrics=runtime.AgentLoopMetrics(**result.metrics),
                extra_fields=result.extra_fields,
            )

    return SearchR1AgentLoop(*args, **kwargs)
