# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strictly compare the executed work in two controlled Search-R1 steps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkEvent:
    start_unix_ns: int
    trace_id: str
    kind: str
    turn: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    query_characters: int | None = None


def _normalized_event(raw: dict[str, Any], framework: str) -> WorkEvent | None:
    attributes = raw.get("attributes")
    if not isinstance(attributes, dict):
        return None
    component = raw.get("component")
    operation = raw.get("operation")

    if framework == "current":
        if component != "current_verl":
            return None
        if operation == "model_generation":
            kind = "model"
            turn = attributes.get("turn")
            input_tokens = attributes.get("input_tokens")
            output_tokens = attributes.get("output_tokens")
            query_characters = None
        elif operation == "retrieval":
            kind = "search"
            turn = attributes.get("turn")
            input_tokens = output_tokens = None
            query_characters = attributes.get("query_characters")
        else:
            return None
    elif framework == "nemo":
        if component != "search_r1_agent":
            return None
        if operation == "model_generate":
            kind = "model"
            turn_index = attributes.get("turn_index")
            turn = None if turn_index is None else int(turn_index) + 1
            input_tokens = attributes.get("input_tokens")
            output_tokens = attributes.get("output_tokens")
            query_characters = None
        elif operation == "search_tool":
            kind = "search"
            turn_index = attributes.get("turn_index")
            turn = None if turn_index is None else int(turn_index) + 1
            input_tokens = output_tokens = None
            query_characters = attributes.get("query_chars")
        else:
            return None
    else:  # pragma: no cover - argparse and compare_steps validate this
        raise ValueError(f"Unsupported framework: {framework}")

    start_unix_ns = raw.get("start_unix_ns")
    trace_id = raw.get("trace_id")
    required_values = [start_unix_ns, trace_id, turn]
    if kind == "model":
        required_values.extend([input_tokens, output_tokens])
    else:
        required_values.append(query_characters)
    if any(value is None for value in required_values):
        raise ValueError(f"Incomplete controlled-work span: {raw}")
    return WorkEvent(
        start_unix_ns=int(start_unix_ns),
        trace_id=str(trace_id),
        kind=kind,
        turn=int(turn),
        input_tokens=None if input_tokens is None else int(input_tokens),
        output_tokens=None if output_tokens is None else int(output_tokens),
        query_characters=(None if query_characters is None else int(query_characters)),
    )


def load_work_events(path: Path, framework: str) -> list[WorkEvent]:
    if framework not in {"current", "nemo"}:
        raise ValueError(f"Unsupported framework: {framework}")
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
        event = _normalized_event(raw, framework)
        if event is not None:
            events.append(event)
    if not events:
        raise ValueError(f"No controlled Search-R1 spans found in {path}")
    return sorted(events, key=lambda event: event.start_unix_ns)


def select_step(
    events: list[WorkEvent], *, step: int, trajectories_per_step: int
) -> list[WorkEvent]:
    if step < 1:
        raise ValueError("step must be at least one")
    if trajectories_per_step < 1:
        raise ValueError("trajectories_per_step must be positive")
    first_turns = [
        event for event in events if event.kind == "model" and event.turn == 1
    ]
    if len(first_turns) % trajectories_per_step:
        raise ValueError(
            "First-turn span count is not divisible by trajectories_per_step: "
            f"{len(first_turns)} vs {trajectories_per_step}"
        )
    step_count = len(first_turns) // trajectories_per_step
    if step > step_count:
        raise ValueError(
            f"Requested step {step}, but trace contains {step_count} steps"
        )

    first_index = (step - 1) * trajectories_per_step
    last_index = first_index + trajectories_per_step
    selected_first_turns = first_turns[first_index:last_index]
    start_ns = selected_first_turns[0].start_unix_ns
    end_ns = (
        first_turns[last_index].start_unix_ns if last_index < len(first_turns) else None
    )
    selected = [
        event
        for event in events
        if event.start_unix_ns >= start_ns
        and (end_ns is None or event.start_unix_ns < end_ns)
    ]
    expected_trace_ids = {event.trace_id for event in selected_first_turns}
    if len(expected_trace_ids) != trajectories_per_step:
        raise ValueError("First-turn trace IDs are not unique within the selected step")
    unexpected = {event.trace_id for event in selected} - expected_trace_ids
    if unexpected:
        raise ValueError(
            f"Events crossed the inferred step boundary for trace IDs {sorted(unexpected)}"
        )
    return selected


def summarize_step(events: list[WorkEvent]) -> dict[str, Any]:
    trajectories: dict[str, list[WorkEvent]] = defaultdict(list)
    for event in events:
        trajectories[event.trace_id].append(event)

    signature_counts: Counter[str] = Counter()
    for trajectory_events in trajectories.values():
        signature = []
        for event in sorted(
            trajectory_events,
            key=lambda item: (item.turn, 0 if item.kind == "model" else 1),
        ):
            if event.kind == "model":
                signature.append(
                    ["model", event.turn, event.input_tokens, event.output_tokens]
                )
            else:
                signature.append(["search", event.turn, event.query_characters])
        signature_counts[json.dumps(signature, separators=(",", ":"))] += 1

    model_events = [event for event in events if event.kind == "model"]
    search_events = [event for event in events if event.kind == "search"]
    return {
        "trajectories": len(trajectories),
        "model_calls": len(model_events),
        "model_input_tokens": sum(event.input_tokens or 0 for event in model_events),
        "generated_tokens": sum(event.output_tokens or 0 for event in model_events),
        "searches": len(search_events),
        "query_characters": sum(event.query_characters or 0 for event in search_events),
        "trajectory_signature_counts": dict(sorted(signature_counts.items())),
    }


def compare_steps(
    *,
    left_trace: Path,
    left_framework: str,
    left_step: int,
    right_trace: Path,
    right_framework: str,
    right_step: int,
    trajectories_per_step: int = 40,
) -> dict[str, Any]:
    left = summarize_step(
        select_step(
            load_work_events(left_trace, left_framework),
            step=left_step,
            trajectories_per_step=trajectories_per_step,
        )
    )
    right = summarize_step(
        select_step(
            load_work_events(right_trace, right_framework),
            step=right_step,
            trajectories_per_step=trajectories_per_step,
        )
    )
    left_signatures = Counter(left["trajectory_signature_counts"])
    right_signatures = Counter(right["trajectory_signature_counts"])
    scalar_keys = (
        "trajectories",
        "model_calls",
        "model_input_tokens",
        "generated_tokens",
        "searches",
        "query_characters",
    )
    scalar_differences = {
        key: {"left": left[key], "right": right[key], "delta": right[key] - left[key]}
        for key in scalar_keys
        if left[key] != right[key]
    }
    missing_on_right = left_signatures - right_signatures
    extra_on_right = right_signatures - left_signatures
    exact_match = not scalar_differences and not missing_on_right and not extra_on_right
    return {
        "exact_match": exact_match,
        "comparison": {
            "left": {
                "trace": str(left_trace),
                "framework": left_framework,
                "step": left_step,
            },
            "right": {
                "trace": str(right_trace),
                "framework": right_framework,
                "step": right_step,
            },
            "trajectories_per_step": trajectories_per_step,
        },
        "left_summary": left,
        "right_summary": right,
        "differences": {
            "scalars": scalar_differences,
            "trajectory_signatures_missing_on_right": dict(missing_on_right),
            "trajectory_signatures_extra_on_right": dict(extra_on_right),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-trace", type=Path, required=True)
    parser.add_argument("--left-framework", choices=("current", "nemo"), required=True)
    parser.add_argument("--left-step", type=int, default=2)
    parser.add_argument("--right-trace", type=Path, required=True)
    parser.add_argument("--right-framework", choices=("current", "nemo"), required=True)
    parser.add_argument("--right-step", type=int, default=2)
    parser.add_argument("--trajectories-per-step", type=int, default=40)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Report a mismatch without returning a failing exit status.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare_steps(
        left_trace=args.left_trace,
        left_framework=args.left_framework,
        left_step=args.left_step,
        right_trace=args.right_trace,
        right_framework=args.right_framework,
        right_step=args.right_step,
        trajectories_per_step=args.trajectories_per_step,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["exact_match"] or args.allow_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
