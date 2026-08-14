# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict Search-R1 rollout built on current veRL Agent Loop."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import verl_search_r1_adapter as adapter
import yaml


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


def _output(text, *, stop_reason="completed", logprobs=True):
    token_ids = [ord(character) for character in text]
    return SimpleNamespace(
        token_ids=token_ids,
        log_probs=[-0.5] * len(token_ids) if logprobs else None,
        routed_experts=None,
        stop_reason=stop_reason,
        num_preempted=1,
        extra_fields={"min_global_steps": 7, "max_global_steps": 7},
    )


class _Server:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _Agent:
    def __init__(self, responses, *, backend="vllm", prompt="Question"):
        self.tokenizer = _Tokenizer()
        self.server_manager = _Server(responses)
        self.rollout_config = SimpleNamespace(
            name=backend,
            full_determinism=False,
            seed=42,
        )
        self.enable_continuous_token = False
        self.prompt = prompt
        self.searches = []

    async def apply_chat_template(self, messages):
        assert messages == [{"role": "user", "content": "Question"}]
        return [ord(character) for character in self.prompt]

    async def retrieve(
        self,
        *,
        query,
        provider_batch_id,
        turn,
        trajectory_id,
    ):
        self.searches.append(
            (query, provider_batch_id, turn, trajectory_id)
        )
        return (
            "\n\n<information>Doc 1(Title: France) "
            "Paris is its capital.</information>\n\n"
        )


def _run(agent, sampling_params=None, **kwargs):
    run_kwargs = {
        "raw_prompt": [{"role": "user", "content": "Question"}],
        **kwargs,
    }
    return asyncio.run(
        adapter.run_protocol(
            agent,
            sampling_params or {"temperature": 1.0, "top_p": 1.0},
            **run_kwargs,
        )
    )


def test_search_then_answer_preserves_logprobs_and_observation_mask() -> None:
    agent = _Agent(
        [_output("<search>capital France</search>"), _output("<answer>Answer</answer>")]
    )
    result = _run(agent)

    assert result.extra_fields["status"] == "completed"
    assert result.extra_fields["search_count"] == 1
    assert result.extra_fields["turn_count"] == 2
    assert result.extra_fields["generated_tokens"] == sum(result.response_mask)
    assert result.extra_fields["observation_tokens"] == result.response_mask.count(0)
    assert 0 in result.response_mask and 1 in result.response_mask
    assert len(result.response_ids) == len(result.response_mask)
    assert len(result.response_ids) == len(result.response_logprobs)
    assert all(
        logprob == 0.0
        for logprob, mask in zip(result.response_logprobs, result.response_mask)
        if mask == 0
    )
    assert len(agent.server_manager.calls) == 2
    assert len(agent.searches) == 1
    assert agent.searches[0][0] == "capital France"
    assert agent.searches[0][1]
    params = agent.server_manager.calls[0]["sampling_params"]
    assert params["max_tokens"] == 500
    assert params["logprobs"] is True
    assert params["stop"] == ["</search>", "</answer>"]
    assert params["include_stop_str_in_output"] is True


def test_controlled_work_uses_raw_prompt_and_greedy_requests(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_R1_WORKLOAD_MODE", "controlled")
    agent = _Agent([_output("<answer>Answer</answer>")], prompt="wrapped")

    result = _run(agent, index=7, session_id=3, global_steps=2)

    assert result.prompt_ids == [ord(character) for character in "Question"]
    params = agent.server_manager.calls[0]["sampling_params"]
    assert params["temperature"] == 0.0
    assert params["top_p"] == 1.0
    assert params["top_k"] == -1
    assert params["seed"] == 42 + 7 * 1024 + 3
    assert agent.server_manager.calls[0]["request_id"] == (
        "det-step-2-sample-7-rollout-3"
    )


def test_controlled_identity_is_unique_for_all_40_trajectories() -> None:
    identities = {
        adapter._controlled_request_identity(
            0,
            {"index": sample_index, "session_id": session_id, "global_steps": 1},
        )[0]
        for sample_index in range(8)
        for session_id in range(5)
    }

    assert len(identities) == 40


def test_controlled_identity_falls_back_to_legacy_priority(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_R1_WORKLOAD_MODE", "controlled")
    agent = _Agent([_output("<answer>Answer</answer>")])

    _run(agent, priority=9)

    params = agent.server_manager.calls[0]["sampling_params"]
    assert params["seed"] == 51
    assert agent.server_manager.calls[0]["request_id"] == "det-priority-9"


def test_unknown_workload_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_R1_WORKLOAD_MODE", "unknown")
    agent = _Agent([_output("<answer>Answer</answer>")])

    with pytest.raises(ValueError, match="must be training or controlled"):
        _run(agent)


def test_common_trace_links_generation_and_retrieval(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "trajectory-spans.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1")
    agent = _Agent(
        [_output("<search>capital France</search>"), _output("<answer>Answer</answer>")]
    )
    _run(agent)

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["operation"] for event in events] == [
        "model_generation",
        "retrieval",
        "model_generation",
    ]
    assert {event["component"] for event in events} == {"current_verl"}
    assert len({event["trace_id"] for event in events}) == 1
    assert events[0]["attributes"]["input_tokens"] == len(agent.prompt)
    retrieval = events[1]
    assert retrieval["attributes"]["provider_batch_id"] == agent.searches[0][1]


def test_four_searches_get_one_final_search_disabled_generation() -> None:
    agent = _Agent(
        [_output(f"<search>query {index}</search>") for index in range(5)]
    )
    result = _run(agent)

    assert len(agent.server_manager.calls) == 5
    assert len(agent.searches) == 4
    assert result.extra_fields["search_count"] == 4
    assert result.extra_fields["turn_count"] == 5
    assert result.extra_fields["natural_termination"] is False
    assert result.extra_fields["status"] == "failed"


def test_sglang_uses_its_stop_delimiter_retention_flag() -> None:
    agent = _Agent([_output("<answer>Answer</answer>")], backend="sglang")
    _run(agent)
    params = agent.server_manager.calls[0]["sampling_params"]
    assert params["no_stop_trim"] is True
    assert "include_stop_str_in_output" not in params


def test_invalid_action_adds_masked_retry_observation() -> None:
    agent = _Agent([_output("not an action"), _output("<answer>Answer</answer>")])
    result = _run(agent)
    response = agent.tokenizer.decode(result.response_ids)

    assert "My previous action is invalid." in response
    assert result.extra_fields["invalid_actions"] == 1
    assert not agent.searches
    assert 0 in result.response_mask


def test_observation_and_starting_prompt_are_capped(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "MAX_OBSERVATION_TOKENS", 10)
    long_prompt = "x" * 3000
    agent = _Agent(
        [_output("<search>capital</search>"), _output("<answer>Answer</answer>")],
        prompt=long_prompt,
    )
    result = _run(agent)

    assert len(result.prompt_ids) == 2048
    assert result.response_mask.count(0) == 10
    second_prompt = agent.server_manager.calls[1]["prompt_ids"]
    assert second_prompt[-10:] == [ord(character) for character in "\n\n<informa"]


def test_missing_selected_token_logprobs_is_rejected() -> None:
    agent = _Agent([_output("<answer>Answer</answer>", logprobs=False)])
    with pytest.raises(RuntimeError, match="log probabilities are required"):
        _run(agent)


def test_aborted_generation_is_reported_without_logprobs() -> None:
    agent = _Agent([_output("", stop_reason="aborted", logprobs=False)])
    result = _run(agent)
    assert result.extra_fields["status"] == "aborted"
    assert result.response_ids == []


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
    }


def test_example_agent_loop_config_uses_public_hydra_factory() -> None:
    config = yaml.safe_load(
        (
            Path(__file__).parent
            / "adapters"
            / "current-verl-search-r1-agent-loop.example.yaml"
        ).read_text(encoding="utf-8")
    )
    assert len(config) == 1
    assert config[0]["name"] == "search_r1_text"
    assert config[0]["_target_"] == (
        "verl_search_r1_adapter.build_search_r1_agent_loop"
    )
    assert config[0]["search_concurrency"] == 256
