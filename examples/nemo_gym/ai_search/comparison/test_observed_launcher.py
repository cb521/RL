# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the observed NeMo RL launcher."""

from pathlib import Path


def test_local_swanlab_installs_matching_dashboard_extra() -> None:
    ai_search_dir = Path(__file__).parents[1]
    launcher = (ai_search_dir / "run_ai_search_observed.sh").read_text()
    preparation = (ai_search_dir / "prepare_ai_search.sh").read_text()

    assert 'export SWANLAB_MODE="${SWANLAB_MODE:-local}"' in launcher
    assert '[[ "${SWANLAB_MODE:-}" == "local" ]]' in preparation
    assert "from importlib.metadata import version" in preparation
    assert '"swanlab[dashboard]==${AI_SEARCH_SWANLAB_VERSION}"' in preparation
    assert '"peewee==3.19.0"' in preparation
    assert 'version("peewee") == "3.19.0"' in preparation
    assert "import swanboard" in preparation


def test_baseline_mode_disables_optional_observers() -> None:
    launcher = (Path(__file__).parents[1] / "run_ai_search_observed.sh").read_text()

    assert "baseline)" in launcher
    assert "logger.swanlab_enabled=false" in launcher
    assert "logger.monitor_gpus=false" in launcher
    assert "++policy.generation.vllm_cfg.enable_vllm_metrics_logger=false" in launcher
    assert "unset AI_SEARCH_TRACE_PATH AI_SEARCH_TRACE_SAMPLE_RATE" in launcher
    assert "unset SWANLAB_MODE SWANLAB_LOGDIR" in launcher


def test_observed_modes_enable_native_vllm_metrics() -> None:
    ai_search_dir = Path(__file__).parents[1]
    launcher = (ai_search_dir / "run_ai_search_observed.sh").read_text()
    worker = (
        ai_search_dir.parents[2] / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    ).read_text()

    assert "++policy.generation.vllm_cfg.enable_vllm_metrics_logger=true" in launcher
    assert "++policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5" in launcher
    assert "vllm-http-servers" in launcher
    assert '@app.get("/metrics", include_in_schema=False)' in worker
    assert "content=generate_latest()" in worker


def test_profile_mode_disables_unavailable_cpu_sampling() -> None:
    launcher = (Path(__file__).parents[1] / "run_ai_search_observed.sh").read_text()

    assert '\\"sample\\":\\"none\\"' in launcher
    assert '\\"cpuctxsw\\":\\"none\\"' in launcher
