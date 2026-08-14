# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict Search-R1 rollout built on slime's hook contract."""

import asyncio
import json
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import slime_search_r1_adapter as adapter
import yaml


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


class _Sample:
    Status = _Status

    def __init__(self):
        self.prompt = "Question"
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.loss_mask = []
        self.rollout_log_probs = None
        self.rollout_top_p_token_ids = None
        self.rollout_top_p_token_offsets = None
        self.non_generation_time = 0.0
        self.metadata = {
            "example_id": "nq:test_0",
            "data_source": "nq",
            "question": "Question",
            "golden_answers": ["Answer"],
        }
        self.status = _Status.PENDING
        self.session_id = None

    def append_response_tokens(
        self,
        args,
        *,
        tokens,
        log_probs=None,
        trainable=True,
        meta_info=None,
        text=None,
    ):
        del args, meta_info
        self.tokens.extend(tokens)
        self.response_length += len(tokens)
        self.loss_mask.extend([1 if trainable else 0] * len(tokens))
        if log_probs is not None:
            if self.rollout_log_probs is None:
                self.rollout_log_probs = []
            self.rollout_log_probs.extend(log_probs)
        elif self.rollout_log_probs is not None:
            self.rollout_log_probs.extend([0.0] * len(tokens))
        if text:
            self.response += text


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


class _Span:
    def update(self, attrs):
        self.attrs = attrs


@contextmanager
def _trace_span(*args, **kwargs):
    del args, kwargs
    yield _Span()


def _output(text, finish="stop"):
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish},
            "output_token_logprobs": [(-0.5, ord(character)) for character in text],
        },
    }


def _runtime(responses, calls):
    async def post(url, payload, max_retries=60, headers=None):
        calls.append((url, payload, max_retries, headers))
        if url.endswith("/retrieve"):
            return {
                "result": [
                    [
                        {
                            "document": {
                                "id": "1",
                                "contents": "France\nParis is its capital.",
                            },
                            "score": 1.0,
                        }
                    ]
                ]
            }
        return responses.pop(0)

    def bind_trace(sample):
        if not hasattr(sample, "trace"):
            sample.trace = {"trace_id": "slime-test-trace", "events": []}
        return SimpleNamespace(trace_id=sample.trace["trace_id"])

    return SimpleNamespace(
        GenerateState=lambda args: SimpleNamespace(tokenizer=_Tokenizer()),
        Sample=_Sample,
        bind_trace=bind_trace,
        post=post,
        trace_span=_trace_span,
        build_sglang_meta_trace_attrs=lambda meta: {"finish": meta["finish_reason"]["type"]},
    )


def _args():
    return SimpleNamespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="random",
        search_r1_retriever_url="http://retriever/retrieve",
    )


def test_search_then_answer_preserves_tokens_logprobs_and_observation_mask(
    monkeypatch,
) -> None:
    calls = []
    responses = [_output("<search>capital France</search>"), _output("<answer>Answer</answer>")]
    monkeypatch.setattr(adapter, "_runtime", lambda: _runtime(responses, calls))
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE", None)
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE_LOOP", None)
    sample = asyncio.run(
        adapter.generate(
            _args(),
            _Sample(),
            {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 4096},
        )
    )

    assert sample.status == _Status.COMPLETED
    assert sample.metadata["search_count"] == 1
    assert sample.metadata["turn_count"] == 2
    assert "<information>Doc 1(Title: France) Paris is its capital." in sample.response
    assert 0 in sample.loss_mask and 1 in sample.loss_mask
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    model_calls = [call for call in calls if call[0].endswith(":30000/generate")]
    search_calls = [call for call in calls if call[0].endswith("/retrieve")]
    assert len(model_calls) == 2
    assert len(search_calls) == 1
    assert all(call[1]["sampling_params"]["max_new_tokens"] == 500 for call in model_calls)
    assert search_calls[0][1] == {
        "queries": ["capital France"],
        "topk": 3,
        "return_scores": True,
    }
    assert search_calls[0][3]["X-NeMo-Search-Batch-ID"]


def test_common_trace_links_generation_and_retrieval(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "trajectory-spans.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1")
    calls = []
    responses = [
        _output("<search>capital France</search>"),
        _output("<answer>Answer</answer>"),
    ]
    monkeypatch.setattr(adapter, "_runtime", lambda: _runtime(responses, calls))
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE", None)
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE_LOOP", None)
    sample = asyncio.run(
        adapter.generate(
            _args(), _Sample(), {"temperature": 1.0, "top_p": 1.0}
        )
    )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["operation"] for event in events] == [
        "model_generation",
        "retrieval",
        "model_generation",
    ]
    assert {event["component"] for event in events} == {"slime"}
    assert {event["trace_id"] for event in events} == {
        sample.metadata["observability_trace_id"]
    }
    retrieval_call = next(call for call in calls if call[0].endswith("/retrieve"))
    assert events[1]["attributes"]["provider_batch_id"] == (
        retrieval_call[3]["X-NeMo-Search-Batch-ID"]
    )


def test_four_searches_get_one_final_search_disabled_generation(monkeypatch) -> None:
    calls = []
    responses = [_output(f"<search>query {index}</search>") for index in range(5)]
    monkeypatch.setattr(adapter, "_runtime", lambda: _runtime(responses, calls))
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE", None)
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE_LOOP", None)
    sample = asyncio.run(
        adapter.generate(
            _args(), _Sample(), {"temperature": 1.0, "top_p": 1.0}
        )
    )
    model_calls = [call for call in calls if call[0].endswith(":30000/generate")]
    search_calls = [call for call in calls if call[0].endswith("/retrieve")]
    assert len(model_calls) == 5
    assert len(search_calls) == 4
    assert sample.metadata["search_count"] == 4
    assert sample.metadata["turn_count"] == 5
    assert sample.metadata["natural_termination"] is False
    assert sample.status == _Status.FAILED


def test_abort_is_returned_without_search(monkeypatch) -> None:
    calls = []
    responses = [
        {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}
    ]
    monkeypatch.setattr(adapter, "_runtime", lambda: _runtime(responses, calls))
    sample = asyncio.run(
        adapter.generate(
            _args(), _Sample(), {"temperature": 1.0, "top_p": 1.0}
        )
    )
    assert sample.status == _Status.ABORTED
    assert len(calls) == 1


def test_observation_tokens_are_capped_before_the_next_turn(monkeypatch) -> None:
    calls = []
    responses = [_output("<search>capital</search>"), _output("<answer>Answer</answer>")]
    monkeypatch.setattr(adapter, "_runtime", lambda: _runtime(responses, calls))
    monkeypatch.setattr(adapter, "MAX_OBSERVATION_TOKENS", 10)
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE", None)
    monkeypatch.setattr(adapter, "_SEARCH_SEMAPHORE_LOOP", None)
    sample = asyncio.run(
        adapter.generate(
            _args(), _Sample(), {"temperature": 1.0, "top_p": 1.0}
        )
    )
    assert sample.loss_mask.count(0) == 10
    second_model_payload = [
        call[1] for call in calls if call[0].endswith(":30000/generate")
    ][1]
    assert second_model_payload["input_ids"][-10:] == [
        ord(character) for character in "\n\n<informa"
    ]


def test_adapter_contract_is_protocol_frozen() -> None:
    assert adapter.adapter_contract() == {
        "action_protocol": "search-r1-text-v1",
        "executable_turns": 4,
        "final_answer_only_generation": True,
        "max_action_tokens": 500,
        "max_observation_tokens": 500,
        "max_start_tokens": 2048,
        "max_response_tokens": 4096,
        "retrieval_top_k": 3,
        "observation_loss_mask": 0,
        "lr_scheduler_granularity": "outer_step",
    }


class _OptimizerScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.num_steps = 0

    def get_lr(self, param_group):
        del param_group
        return float(self.num_steps)

    def get_wd(self, param_group):
        del param_group
        return 0.01

    def step(self, increment):
        self.num_steps += increment
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(param_group)
            param_group["weight_decay"] = self.get_wd(param_group)


def test_outer_step_lr_hook_holds_all_minibatches_at_one_lr() -> None:
    for outer_batch_size, optimizer_batch_size in ((40, 40), (2560, 256)):
        args = SimpleNamespace(
            rollout_batch_size=outer_batch_size // 5,
            n_samples_per_prompt=5,
            global_batch_size=optimizer_batch_size,
        )
        optimizer = SimpleNamespace(param_groups=[{"lr": -1.0, "wd_mult": 1.0}])
        scheduler = _OptimizerScheduler(optimizer)
        updates_per_outer = outer_batch_size // optimizer_batch_size

        for rollout_id in range(3):
            for step_id in range(updates_per_outer):
                adapter.align_outer_step_lr(
                    args,
                    rollout_id,
                    step_id,
                    model=None,
                    optimizer=optimizer,
                    opt_param_scheduler=scheduler,
                )
                assert optimizer.param_groups[0]["lr"] == float(
                    rollout_id * outer_batch_size
                )
                assert optimizer.param_groups[0]["weight_decay"] == 0.01
                scheduler.step(optimizer_batch_size)
            assert scheduler.num_steps == (rollout_id + 1) * outer_batch_size


def test_example_eval_config_uses_aligned_public_hooks() -> None:
    config = yaml.safe_load(
        (
            Path(__file__).parent
            / "adapters"
            / "slime-search-r1-eval.example.yaml"
        ).read_text(encoding="utf-8")
    )
    defaults = config["eval"]["defaults"]
    assert defaults["n_samples_per_eval_prompt"] == 1
    assert defaults["custom_generate_function_path"] == (
        "slime_search_r1_adapter.generate"
    )
    assert defaults["custom_rm_path"] == "framework_eval_adapters.slime_reward"
