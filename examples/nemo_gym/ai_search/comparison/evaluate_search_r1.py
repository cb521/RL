# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate framework-neutral Search-R1 prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import string
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


_ACTION_PATTERN = re.compile(r"<(search|information|answer)>(.*?)</\1>", re.DOTALL)
_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b")
_ERROR_PATTERN = re.compile(
    r"(?:search|retrieval|request)\s+(?:error|failed|timed out)", re.IGNORECASE
)


def normalize_answer(value: str) -> str:
    """Apply the normalization used by Search-R1's exact-match reward."""
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLE_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, golden_answers: Iterable[str]) -> float:
    """Return one when the normalized prediction matches any golden answer."""
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(
            normalized_prediction == normalize_answer(answer)
            for answer in golden_answers
        )
    )


def _token_f1(prediction: str, golden_answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    golden_tokens = normalize_answer(golden_answer).split()
    if not prediction_tokens or not golden_tokens:
        return float(prediction_tokens == golden_tokens)
    common = Counter(prediction_tokens) & Counter(golden_tokens)
    shared = sum(common.values())
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(golden_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, golden_answers: Iterable[str]) -> float:
    """Return the maximum SQuAD-style token F1 over golden answers."""
    scores = [_token_f1(prediction, answer) for answer in golden_answers]
    return max(scores, default=0.0)


def _balanced_tags(response: str) -> bool:
    return all(
        response.count(f"<{tag}>") == response.count(f"</{tag}>")
        for tag in ("search", "information", "answer")
    )


def parse_response(response: str) -> dict[str, Any]:
    """Extract actions and validate the response-only Search-R1 tag sequence."""
    matches = list(_ACTION_PATTERN.finditer(response))
    actions = [match.group(1) for match in matches]
    queries = [
        match.group(2).strip() for match in matches if match.group(1) == "search"
    ]
    information = [
        match.group(2).strip() for match in matches if match.group(1) == "information"
    ]
    answers = [
        match.group(2).strip() for match in matches if match.group(1) == "answer"
    ]

    protocol_valid = _balanced_tags(response) and bool(answers)
    state = "action"
    if protocol_valid:
        for action in actions:
            if state == "action" and action == "search":
                state = "information"
            elif state == "information" and action == "information":
                state = "action"
            elif state == "action" and action == "answer":
                state = "terminal"
            else:
                protocol_valid = False
                break
        protocol_valid = protocol_valid and state == "terminal"

    normalized_queries = [normalize_answer(query) for query in queries]
    repeated_queries = len(normalized_queries) - len(set(normalized_queries))
    return {
        "answer": answers[-1] if answers else "",
        "answer_tag_valid": _balanced_tags(response) and bool(answers),
        "protocol_valid": protocol_valid,
        "natural_termination": bool(answers),
        "queries": queries,
        "information": information,
        "search_count": len(queries),
        "repeated_queries": repeated_queries,
        "inferred_search_errors": len(_ERROR_PATTERN.findall(response)),
    }


def _as_nonnegative_int(record: dict[str, Any], name: str) -> int | None:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer when provided")
    return value


def _golden_answers(record: dict[str, Any]) -> list[str]:
    value = record.get("golden_answers")
    if isinstance(value, dict):
        value = value.get("target")
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError("golden_answers must be a non-empty string list")
    return value


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and score one common-schema prediction record."""
    for name in ("example_id", "data_source", "response"):
        if name not in record:
            raise ValueError(f"Missing required field: {name}")
    example_id = str(record["example_id"])
    data_source = record["data_source"]
    response = record["response"]
    if not example_id:
        raise ValueError("example_id cannot be empty")
    if not isinstance(data_source, str) or not data_source:
        raise ValueError("data_source must be a non-empty string")
    if not isinstance(response, str):
        raise ValueError("response must be a string")

    golden_answers = _golden_answers(record)
    parsed = parse_response(response)
    information_text = " ".join(parsed["information"])
    normalized_information = normalize_answer(information_text)
    retrieval_hit = bool(parsed["search_count"]) and any(
        normalize_answer(answer) in normalized_information
        for answer in golden_answers
        if normalize_answer(answer)
    )

    status = record.get("status", "completed")
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    search_errors = _as_nonnegative_int(record, "search_errors")
    invalid_actions = _as_nonnegative_int(record, "invalid_actions")
    generated_tokens = _as_nonnegative_int(record, "generated_tokens")
    observation_tokens = _as_nonnegative_int(record, "observation_tokens")
    wall_time_seconds = record.get("wall_time_seconds")
    if wall_time_seconds is not None:
        if (
            isinstance(wall_time_seconds, bool)
            or not isinstance(wall_time_seconds, (int, float))
            or not math.isfinite(float(wall_time_seconds))
            or wall_time_seconds < 0
        ):
            raise ValueError("wall_time_seconds must be a finite non-negative number")
        wall_time_seconds = float(wall_time_seconds)

    return {
        "example_id": example_id,
        "data_source": data_source,
        "golden_answers": golden_answers,
        "answer": parsed["answer"],
        "exact_match": exact_match(parsed["answer"], golden_answers),
        "answer_f1": answer_f1(parsed["answer"], golden_answers),
        "answer_tag_valid": float(parsed["answer_tag_valid"]),
        "protocol_valid": float(parsed["protocol_valid"]),
        "natural_termination": float(parsed["natural_termination"]),
        "retrieval_answer_hit": float(retrieval_hit),
        "search_count": parsed["search_count"],
        "repeated_queries": parsed["repeated_queries"],
        "search_errors": (
            search_errors
            if search_errors is not None
            else parsed["inferred_search_errors"]
        ),
        "search_errors_reported": search_errors is not None,
        "invalid_actions": invalid_actions,
        "generated_tokens": generated_tokens,
        "observation_tokens": observation_tokens,
        "wall_time_seconds": wall_time_seconds,
        "status": status,
    }


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Read, validate, and score one prediction JSONL file."""
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            try:
                scored = evaluate_record(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid record at {path}:{line_number}: {error}"
                ) from error
            example_id = scored["example_id"]
            if example_id in records:
                raise ValueError(f"Duplicate example_id {example_id!r} in {path}")
            records[example_id] = scored
    if not records:
        raise ValueError(f"Prediction file is empty: {path}")
    return records


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float | int | None]) -> dict[str, float | int]:
    """Summarize available finite values with common units."""
    present = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not present:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(present),
        "mean": statistics.fmean(present),
        "p50": _percentile(present, 50),
        "p95": _percentile(present, 95),
        "max": max(present),
    }


def _mean(records: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records]
    return statistics.fmean(values) if values else 0.0


def summarize_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall and per-source Search-R1 quality metrics."""
    values = list(records.values())
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in values:
        by_source.setdefault(record["data_source"], []).append(record)

    searched = [record for record in values if record["search_count"] > 0]
    total_searches = sum(record["search_count"] for record in values)
    total_repeated = sum(record["repeated_queries"] for record in values)
    total_search_errors = sum(record["search_errors"] for record in values)
    invalid_actions = [
        record["invalid_actions"]
        for record in values
        if record["invalid_actions"] is not None
    ]
    status_counts = Counter(record["status"] for record in values)

    def source_summary(source_records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(source_records),
            "exact_match": _mean(source_records, "exact_match"),
            "answer_f1": _mean(source_records, "answer_f1"),
            "answer_tag_valid_rate": _mean(source_records, "answer_tag_valid"),
            "protocol_valid_rate": _mean(source_records, "protocol_valid"),
            "natural_termination_rate": _mean(source_records, "natural_termination"),
        }

    return {
        **source_summary(values),
        "status_counts": dict(sorted(status_counts.items())),
        "retrieval_answer_recall": (
            _mean(searched, "retrieval_answer_hit") if searched else 0.0
        ),
        "retrieval_answer_recall_examples": len(searched),
        "search": {
            "total": total_searches,
            "mean_per_example": total_searches / len(values),
            "no_search_rate": 1 - len(searched) / len(values),
            "repeated": total_repeated,
            "repeated_query_rate": (
                total_repeated / total_searches if total_searches else 0.0
            ),
            "errors": total_search_errors,
            "error_rate": (
                total_search_errors / total_searches if total_searches else 0.0
            ),
            "reported_error_examples": sum(
                record["search_errors_reported"] for record in values
            ),
        },
        "invalid_actions": {
            "reported_examples": len(invalid_actions),
            "total": sum(invalid_actions),
        },
        "generated_tokens": summarize(record["generated_tokens"] for record in values),
        "observation_tokens": summarize(
            record["observation_tokens"] for record in values
        ),
        "wall_time_seconds": summarize(
            record["wall_time_seconds"] for record in values
        ),
        "by_source": {
            source: source_summary(source_records)
            for source, source_records in sorted(by_source.items())
        },
    }


def _bootstrap_difference(
    first: list[float],
    second: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if len(first) != len(second) or not first:
        raise ValueError("Paired bootstrap requires equal non-empty inputs")
    observed = statistics.fmean(a - b for a, b in zip(first, second))
    generator = random.Random(seed)
    differences = []
    for _ in range(replicates):
        indices = [generator.randrange(len(first)) for _ in first]
        differences.append(
            statistics.fmean(first[index] - second[index] for index in indices)
        )
    return {
        "examples": len(first),
        "first_minus_second": observed,
        "bootstrap_replicates": replicates,
        "ci95_low": _percentile(differences, 2.5),
        "ci95_high": _percentile(differences, 97.5),
    }


def compare_predictions(
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_replicates: int,
    seed: int,
    require_identical_ids: bool = True,
) -> dict[str, Any]:
    """Compute paired accuracy differences for every implementation pair."""
    comparisons: dict[str, Any] = {}
    for pair_index, (first_name, second_name) in enumerate(
        combinations(predictions, 2)
    ):
        first_records = predictions[first_name]
        second_records = predictions[second_name]
        first_ids = set(first_records)
        second_ids = set(second_records)
        if require_identical_ids and first_ids != second_ids:
            raise ValueError(
                f"Prediction IDs differ for {first_name} and {second_name}: "
                f"{len(first_ids - second_ids)} only in first, "
                f"{len(second_ids - first_ids)} only in second"
            )
        common_ids = sorted(first_ids & second_ids)
        for example_id in common_ids:
            first = first_records[example_id]
            second = second_records[example_id]
            if (
                first["golden_answers"] != second["golden_answers"]
                or first["data_source"] != second["data_source"]
            ):
                raise ValueError(
                    f"Ground truth differs for {example_id!r} between "
                    f"{first_name} and {second_name}"
                )

        first_em = [first_records[key]["exact_match"] for key in common_ids]
        second_em = [second_records[key]["exact_match"] for key in common_ids]
        first_f1 = [first_records[key]["answer_f1"] for key in common_ids]
        second_f1 = [second_records[key]["answer_f1"] for key in common_ids]
        discordance = Counter(
            (
                int(first_records[key]["exact_match"]),
                int(second_records[key]["exact_match"]),
            )
            for key in common_ids
        )
        pair_seed = seed + pair_index * 10_003
        comparisons[f"{first_name}__vs__{second_name}"] = {
            "first": first_name,
            "second": second_name,
            "common_examples": len(common_ids),
            "missing_from_first": len(second_ids - first_ids),
            "missing_from_second": len(first_ids - second_ids),
            "exact_match_difference": _bootstrap_difference(
                first_em,
                second_em,
                replicates=bootstrap_replicates,
                seed=pair_seed,
            ),
            "answer_f1_difference": _bootstrap_difference(
                first_f1,
                second_f1,
                replicates=bootstrap_replicates,
                seed=pair_seed + 1,
            ),
            "exact_match_pairs": {
                "both_correct": discordance[(1, 1)],
                "first_only_correct": discordance[(1, 0)],
                "second_only_correct": discordance[(0, 1)],
                "neither_correct": discordance[(0, 0)],
            },
        }
    return comparisons


def _prediction_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/predictions.jsonl")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Prediction name and path cannot be empty")
    return name, Path(raw_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        action="append",
        type=_prediction_argument,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--allow-missing-ids", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    named_paths = dict(args.predictions)
    if len(named_paths) != len(args.predictions):
        raise ValueError("Prediction names must be unique")
    for path in named_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = {name: read_predictions(path) for name, path in named_paths.items()}
    report = {
        "schema_version": 1,
        "bootstrap_seed": args.seed,
        "implementations": {
            name: summarize_records(records) for name, records in predictions.items()
        },
        "comparisons": compare_predictions(
            predictions,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            require_identical_ids=not args.allow_missing_ids,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
