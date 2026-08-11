# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the framework-neutral Search-R1 evaluator."""

import json

import pytest

from evaluate_search_r1 import (
    compare_predictions,
    evaluate_record,
    normalize_answer,
    parse_response,
    read_predictions,
    summarize_records,
)


def _record(example_id, answer, *, golden="Paris", source="nq"):
    return {
        "example_id": example_id,
        "data_source": source,
        "golden_answers": [golden],
        "response": answer,
        "status": "completed",
    }


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_normalize_answer_matches_search_r1() -> None:
    assert normalize_answer(" The, Eiffel-Tower! ") == "eiffeltower"


def test_parse_response_tracks_searches_and_protocol() -> None:
    response = (
        "reason <search>capital of France</search>"
        "<information>Paris is the capital.</information>"
        "again <search>capital of France</search>"
        "<information>Paris</information>"
        "<answer>Paris</answer>"
    )
    parsed = parse_response(response)
    assert parsed["answer"] == "Paris"
    assert parsed["protocol_valid"] is True
    assert parsed["search_count"] == 2
    assert parsed["repeated_queries"] == 1


def test_evaluate_record_computes_quality_and_retrieval_metrics() -> None:
    record = _record(
        "nq-1",
        "<search>capital of France</search>"
        "<information>Paris is the capital of France.</information>"
        "<answer>The Paris.</answer>",
    ) | {"generated_tokens": 12, "search_errors": 0}
    scored = evaluate_record(record)
    assert scored["exact_match"] == 1.0
    assert scored["answer_f1"] == 1.0
    assert scored["retrieval_answer_hit"] == 1.0
    assert scored["protocol_valid"] == 1.0


def test_evaluate_record_rejects_malformed_optional_metrics() -> None:
    with pytest.raises(ValueError, match="generated_tokens"):
        evaluate_record(
            _record("nq-1", "<answer>Paris</answer>") | {"generated_tokens": -1}
        )


def test_summary_and_paired_bootstrap_use_identical_examples(tmp_path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(
        first_path,
        [
            _record("1", "<answer>Paris</answer>"),
            _record("2", "<answer>London</answer>", source="hotpotqa"),
        ],
    )
    _write_jsonl(
        second_path,
        [
            _record("1", "<answer>London</answer>"),
            _record("2", "<answer>Paris</answer>", source="hotpotqa"),
        ],
    )
    first = read_predictions(first_path)
    second = read_predictions(second_path)
    summary = summarize_records(first)
    assert summary["exact_match"] == 0.5
    assert summary["by_source"]["nq"]["exact_match"] == 1.0

    report = compare_predictions(
        {"first": first, "second": second},
        bootstrap_replicates=200,
        seed=7,
    )["first__vs__second"]
    assert report["exact_match_difference"]["first_minus_second"] == 0.0
    assert report["exact_match_pairs"] == {
        "both_correct": 0,
        "first_only_correct": 1,
        "second_only_correct": 1,
        "neither_correct": 0,
    }


def test_paired_comparison_rejects_missing_ids(tmp_path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(first_path, [_record("1", "<answer>Paris</answer>")])
    _write_jsonl(second_path, [_record("2", "<answer>Paris</answer>")])
    with pytest.raises(ValueError, match="Prediction IDs differ"):
        compare_predictions(
            {
                "first": read_predictions(first_path),
                "second": read_predictions(second_path),
            },
            bootstrap_replicates=10,
            seed=1,
        )
