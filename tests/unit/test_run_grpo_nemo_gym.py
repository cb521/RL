# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

from examples.nemo_gym.run_grpo_nemo_gym import shutdown_runtime


@patch("examples.nemo_gym.run_grpo_nemo_gym.ray")
def test_shutdown_runtime_closes_every_component(mock_ray) -> None:
    nemo_gym = MagicMock()
    shutdown_ref = object()
    nemo_gym.shutdown.remote.return_value = shutdown_ref
    policy = MagicMock()
    policy_generation = MagicMock()
    cluster = MagicMock()
    logger = MagicMock()

    shutdown_runtime(
        nemo_gym=nemo_gym,
        policy=policy,
        policy_generation=policy_generation,
        cluster=cluster,
        logger=logger,
    )

    logger.finish.assert_called_once_with()
    nemo_gym.shutdown.remote.assert_called_once_with()
    mock_ray.get.assert_called_once_with(shutdown_ref, timeout=10)
    policy_generation.shutdown.assert_called_once_with()
    policy.shutdown.assert_called_once_with()
    cluster.shutdown.assert_called_once_with()
    mock_ray.shutdown.assert_called_once_with()


@patch("examples.nemo_gym.run_grpo_nemo_gym.ray")
def test_shutdown_runtime_continues_after_cleanup_error(mock_ray) -> None:
    nemo_gym = MagicMock()
    policy = MagicMock()
    policy_generation = MagicMock()
    cluster = MagicMock()
    logger = MagicMock()
    logger.finish.side_effect = RuntimeError("logger failure")

    shutdown_runtime(
        nemo_gym=nemo_gym,
        policy=policy,
        policy_generation=policy_generation,
        cluster=cluster,
        logger=logger,
    )

    policy_generation.shutdown.assert_called_once_with()
    policy.shutdown.assert_called_once_with()
    cluster.shutdown.assert_called_once_with()
    mock_ray.shutdown.assert_called_once_with()
