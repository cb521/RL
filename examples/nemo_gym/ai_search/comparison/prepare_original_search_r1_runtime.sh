#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_input="${comparison_dir}/adapters/original-search-r1-runtime.in"
runtime_lock="${comparison_dir}/adapters/original-search-r1-runtime.lock"
venv_dir="${ORIGINAL_SEARCH_R1_VENV:?ORIGINAL_SEARCH_R1_VENV must be an absolute new or previously prepared venv path}"
bootstrap_python="${ORIGINAL_SEARCH_R1_BOOTSTRAP_PYTHON:-python3.10}"
require_gpu="${ORIGINAL_SEARCH_R1_REQUIRE_GPU:-1}"
expected_input_sha256=c6262fbcd6b5d39cb59589e96a675103fb5da39d0bdccecc3171eb9b34019c49
expected_lock_sha256=11a7246f36c9ea844e14b030631d8f8b1489cd245f663a5864bc8b3074d5f269
expected_uv_version='uv 0.12.3 (x86_64-unknown-linux-gnu)'

if (( $# != 0 )); then
  echo "Use environment variables; positional arguments are not accepted." >&2
  exit 1
fi
if [[ "${venv_dir}" != /* || "${venv_dir}" == / || "${venv_dir}" == /home || "${venv_dir}" == /tmp ]]; then
  echo "ORIGINAL_SEARCH_R1_VENV must be a specific absolute environment path." >&2
  exit 1
fi
if [[ "$(uname -m)" != x86_64 ]]; then
  echo "The frozen original Search-R1 runtime supports x86_64 only." >&2
  exit 1
fi
if [[ "${require_gpu}" != 0 && "${require_gpu}" != 1 ]]; then
  echo "ORIGINAL_SEARCH_R1_REQUIRE_GPU must be 0 or 1." >&2
  exit 1
fi
for artifact in "${runtime_input}" "${runtime_lock}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "Missing original Search-R1 runtime artifact: ${artifact}" >&2
    exit 1
  fi
done
printf '%s  %s\n' "${expected_input_sha256}" "${runtime_input}" | sha256sum --check --status
printf '%s  %s\n' "${expected_lock_sha256}" "${runtime_lock}" | sha256sum --check --status

uv_bin="${UV_BIN:-}"
if [[ -z "${uv_bin}" ]]; then
  uv_bin=$(command -v uv || true)
fi
if [[ -z "${uv_bin}" || ! -x "${uv_bin}" ]]; then
  echo "Set UV_BIN to the pinned uv 0.12.3 executable." >&2
  exit 1
fi
actual_uv_version=$("${uv_bin}" --version)
if [[ "${actual_uv_version}" != "${expected_uv_version}" ]]; then
  echo "Expected ${expected_uv_version}, found ${actual_uv_version}." >&2
  exit 1
fi
marker="${venv_dir}/.original-search-r1-runtime"
runtime_python="${venv_dir}/bin/python"

verify_runtime() {
  "${uv_bin}" pip check --no-config --python "${runtime_python}"
  ORIGINAL_SEARCH_R1_REQUIRE_GPU="${require_gpu}" "${runtime_python}" - <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import version

import flash_attn
import ray
import torch
import transformers
import vllm

expected = {
    "accelerate": "1.2.1",
    "datasets": "2.21.0",
    "flash-attn": "2.6.3",
    "prometheus-client": "0.21.1",
    "ray": "2.10.0",
    "tensorboard": "2.17.1",
    "tensordict": "0.5.0",
    "torch": "2.4.0+cu121",
    "torchvision": "0.19.0+cu121",
    "transformers": "4.47.1",
    "vllm": "0.6.3",
}
actual = {package: version(package) for package in expected}
if actual != expected:
    raise RuntimeError(f"Original Search-R1 runtime drift: {actual!r}")
if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"Expected CPython 3.10, found {sys.version}")
if platform.machine() != "x86_64":
    raise RuntimeError(f"Expected x86_64, found {platform.machine()}")
if torch.version.cuda != "12.1":
    raise RuntimeError(f"Expected CUDA 12.1 Torch build, found {torch.version.cuda}")
if torch._C._GLIBCXX_USE_CXX11_ABI is not False:
    raise RuntimeError("The pinned FlashAttention wheel requires the old C++ ABI")
if os.environ["ORIGINAL_SEARCH_R1_REQUIRE_GPU"] == "1" and not torch.cuda.is_available():
    raise RuntimeError("The formal runtime check requires a visible CUDA GPU")
print(
    json.dumps(
        {
            "packages": actual,
            "python": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "visible_gpus": torch.cuda.device_count(),
            "flash_attn_import": flash_attn.__version__,
            "ray_import": ray.__version__,
            "transformers_import": transformers.__version__,
            "vllm_import": vllm.__version__,
        },
        sort_keys=True,
    )
)
PY
}

runtime_freeze_sha256() {
  "${uv_bin}" pip freeze --no-config --python "${runtime_python}" \
    | LC_ALL=C sort \
    | sha256sum \
    | awk '{print $1}'
}

if [[ -e "${venv_dir}" ]]; then
  if [[ ! -x "${runtime_python}" || ! -f "${marker}" ]]; then
    echo "Refusing to overwrite an unverified existing path: ${venv_dir}" >&2
    exit 1
  fi
  grep -Fxq "input_sha256=${expected_input_sha256}" "${marker}"
  grep -Fxq "lock_sha256=${expected_lock_sha256}" "${marker}"
  grep -Fxq "uv_version=${expected_uv_version}" "${marker}"
  current_freeze_sha256=$(runtime_freeze_sha256)
  grep -Fxq "freeze_sha256=${current_freeze_sha256}" "${marker}"
  verify_runtime
  echo "ORIGINAL_SEARCH_R1_RUNTIME_REUSE_PASS"
  exit 0
fi

if ! command -v "${bootstrap_python}" >/dev/null 2>&1 && [[ ! -x "${bootstrap_python}" ]]; then
  echo "Cannot execute ORIGINAL_SEARCH_R1_BOOTSTRAP_PYTHON=${bootstrap_python}." >&2
  exit 1
fi
if [[ "$("${bootstrap_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != 3.10 ]]; then
  echo "The frozen runtime requires CPython 3.10." >&2
  exit 1
fi

"${uv_bin}" venv \
  --no-project \
  --no-config \
  --python "${bootstrap_python}" \
  "${venv_dir}"
"${uv_bin}" pip install \
  --no-config \
  --python "${runtime_python}" \
  --torch-backend cu121 \
  --require-hashes \
  --requirement "${runtime_lock}"
verify_runtime
freeze_sha256=$(runtime_freeze_sha256)

temporary_marker="${marker}.tmp.$$"
{
  printf 'contract=original-search-r1-runtime-v1\n'
  printf 'input_sha256=%s\n' "${expected_input_sha256}"
  printf 'lock_sha256=%s\n' "${expected_lock_sha256}"
  printf 'uv_version=%s\n' "${expected_uv_version}"
  printf 'freeze_sha256=%s\n' "${freeze_sha256}"
  printf 'python=%s\n' "$("${runtime_python}" --version 2>&1)"
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
} > "${temporary_marker}"
mv "${temporary_marker}" "${marker}"
echo "ORIGINAL_SEARCH_R1_RUNTIME_CREATE_PASS"
