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
    assert "-c 'import swanboard'" in preparation
