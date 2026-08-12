#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nemo_root=$(cd -- "${comparison_dir}/../../../.." && pwd)
slime_root="${SLIME_ROOT:?SLIME_ROOT must point to the frozen patched slime checkout}"
megatron_root="${SLIME_MEGATRON_ROOT:?SLIME_MEGATRON_ROOT must point to Megatron-LM}"
model_path="${SEARCH_R1_MODEL_PATH:?SEARCH_R1_MODEL_PATH must point to the frozen Qwen2.5-7B snapshot}"
ref_load="${SEARCH_R1_SLIME_REF_LOAD:?SEARCH_R1_SLIME_REF_LOAD must be the output checkpoint directory}"
runtime_image="${SEARCH_R1_SLIME_IMAGE:?SEARCH_R1_SLIME_IMAGE must identify the pinned runtime image}"
python_bin="${SLIME_PYTHON:-python3}"

expected_slime_base=a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e
expected_slime_patched_head=b4de90946ad5bbf137c6b542d8275b5c9feb902d
expected_slime_patch_sha256=ec1849f5087befa622bc6fb27d8003dbc72595f7cce809f4896678714f4c40ff
expected_model_revision=d149729398750b98c0af14eb82c78cfe92750796
expected_runtime_image='slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f'

if (( $# != 0 )); then
  echo "The frozen slime checkpoint preparer does not accept positional arguments." >&2
  exit 1
fi
if [[ "${ref_load}" != /* ]]; then
  echo "SEARCH_R1_SLIME_REF_LOAD must be an absolute path." >&2
  exit 1
fi
ref_load=$(readlink -m "${ref_load}")
if [[ "${ref_load}" == / ]]; then
  echo "SEARCH_R1_SLIME_REF_LOAD must not be the filesystem root." >&2
  exit 1
fi
if [[ "${runtime_image}" != "${expected_runtime_image}" ]]; then
  echo "slime runtime image must be ${expected_runtime_image}." >&2
  exit 1
fi
if [[ "$(basename -- "$(readlink -f "${model_path}")")" != "${expected_model_revision}" ]]; then
  echo "SEARCH_R1_MODEL_PATH does not resolve to revision ${expected_model_revision}." >&2
  exit 1
fi
for required_path in \
  "${model_path}/model.safetensors.index.json" \
  "${megatron_root}" \
  "${slime_root}/tools/convert_hf_to_torch_dist.py" \
  "${slime_root}/tools/convert_torch_dist_to_hf.py" \
  "${slime_root}/scripts/models/qwen2.5-7B.sh" \
  "${comparison_dir}/adapters/slime-search-r1-comparison.patch" \
  "${comparison_dir}/verify_safetensors_equivalence.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing slime conversion artifact: ${required_path}" >&2
    exit 1
  fi
done
actual_slime_head=$(git -C "${slime_root}" rev-parse HEAD)
if [[ "${actual_slime_head}" != "${expected_slime_patched_head}" ]]; then
  echo "slime checkout moved from ${expected_slime_patched_head}." >&2
  exit 1
fi
if ! git -C "${slime_root}" merge-base --is-ancestor \
  "${expected_slime_base}" "${actual_slime_head}"; then
  echo "slime patch head does not descend from ${expected_slime_base}." >&2
  exit 1
fi
actual_slime_patch_sha256="$({
  git -C "${slime_root}" diff --full-index --binary \
    "${expected_slime_base}..${actual_slime_head}"
} | sha256sum | cut -d ' ' -f 1)"
if [[ "${actual_slime_patch_sha256}" != "${expected_slime_patch_sha256}" ]]; then
  echo "slime comparison diff does not match ${expected_slime_patch_sha256}." >&2
  exit 1
fi
printf '%s  %s\n' \
  "${expected_slime_patch_sha256}" \
  "${comparison_dir}/adapters/slime-search-r1-comparison.patch" \
  | sha256sum --check --status
if ! git -C "${slime_root}" diff --quiet || ! git -C "${slime_root}" diff --cached --quiet; then
  echo "slime checkout has uncommitted tracked changes." >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null; then
  echo "Cannot execute SLIME_PYTHON=${python_bin}." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${slime_root}/scripts/models/qwen2.5-7B.sh"
conversion_command=(
  "${python_bin}" "${slime_root}/tools/convert_hf_to_torch_dist.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${model_path}"
  --ckpt-format torch_dist
  --save "${ref_load}.partial"
)
roundtrip_command=(
  "${python_bin}" "${slime_root}/tools/convert_torch_dist_to_hf.py"
  --input-dir "${ref_load}.partial/release"
  --output-dir "${ref_load}.roundtrip"
  --origin-hf-dir "${model_path}"
  --vocab-size 152064
)
if [[ "${SEARCH_R1_PRINT_COMMAND:-0}" == 1 ]]; then
  printf '%q ' "${conversion_command[@]}"
  printf '\n'
  printf '%q ' "${roundtrip_command[@]}"
  printf '\n'
  exit 0
fi

if ! git -C "${nemo_root}" diff --quiet || ! git -C "${nemo_root}" diff --cached --quiet; then
  echo "NeMo comparison code has uncommitted tracked changes." >&2
  exit 1
fi
if [[ -e "${ref_load}" ]]; then
  if [[ ! -f "${ref_load}/conversion-manifest.json" ]] || \
    [[ ! -f "${ref_load}/safetensors-equivalence.json" ]] || \
    [[ ! -f "${ref_load}/checkpoint-files.sha256" ]]; then
    echo "Existing SEARCH_R1_SLIME_REF_LOAD is not a verified conversion: ${ref_load}" >&2
    exit 1
  fi
  (
    cd "${ref_load}"
    sha256sum --check --quiet checkpoint-files.sha256
  )
  "${python_bin}" - \
    "${ref_load}/conversion-manifest.json" \
    "${expected_runtime_image}" \
    "$(readlink -f "${model_path}")" \
    "${expected_slime_patched_head}" \
    "${expected_slime_patch_sha256}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["contract"] == "search-r1-slime-checkpoint-conversion-v1"
assert manifest["runtime_image"] == sys.argv[2]
assert manifest["model_path"] == sys.argv[3]
assert manifest["slime_source_head"] == sys.argv[4]
assert manifest["slime_patch_sha256"] == sys.argv[5]
assert manifest["verification"]["bitwise_equal"] is True
PY
  chmod -R a+rX -- "${ref_load}"
  echo "SEARCH_R1_SLIME_CHECKPOINT_REUSE_PASS ${ref_load}"
  exit 0
fi

output_parent=$(dirname -- "${ref_load}")
output_name=$(basename -- "${ref_load}")
mkdir -p "${output_parent}"
staging_dir=$(mktemp -d "${output_parent}/.${output_name}.partial.XXXXXX")
roundtrip_root=$(mktemp -d "${output_parent}/.${output_name}.roundtrip.XXXXXX")
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${staging_dir:-}" && -d "${staging_dir}" && \
    "${staging_dir}" == "${output_parent}/.${output_name}.partial."* ]]; then
    rm -rf -- "${staging_dir}"
  fi
  if [[ -n "${roundtrip_root:-}" && -d "${roundtrip_root}" && \
    "${roundtrip_root}" == "${output_parent}/.${output_name}.roundtrip."* ]]; then
    rm -rf -- "${roundtrip_root}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

visible_gpus=$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${visible_gpus}" != 1 ]]; then
  echo "Checkpoint conversion requires exactly one visible GPU, found ${visible_gpus}." >&2
  exit 1
fi
export PYTHONPATH="${megatron_root}:${slime_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${slime_root}"

conversion_started=$(date +%s)
"${python_bin}" "${slime_root}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${model_path}" \
  --ckpt-format torch_dist \
  --save "${staging_dir}"
conversion_seconds=$(( $(date +%s) - conversion_started ))

if [[ "$(tr -d '[:space:]' < "${staging_dir}/latest_checkpointed_iteration.txt")" != release ]]; then
  echo "Converted checkpoint tracker does not name the release checkpoint." >&2
  exit 1
fi
checkpoint_leaf="${staging_dir}/release"
distcp_file=$(find "${checkpoint_leaf}" -type f -name '*.distcp' -print -quit)
if [[ ! -f "${checkpoint_leaf}/common.pt" || -z "${distcp_file}" ]]; then
  echo "Converted checkpoint does not contain common.pt and distcp shards." >&2
  exit 1
fi

roundtrip_dir="${roundtrip_root}/hf"
roundtrip_started=$(date +%s)
"${python_bin}" "${slime_root}/tools/convert_torch_dist_to_hf.py" \
  --input-dir "${checkpoint_leaf}" \
  --output-dir "${roundtrip_dir}" \
  --origin-hf-dir "${model_path}" \
  --vocab-size 152064
roundtrip_seconds=$(( $(date +%s) - roundtrip_started ))

verification_started=$(date +%s)
"${python_bin}" "${comparison_dir}/verify_safetensors_equivalence.py" \
  --reference "${model_path}" \
  --candidate "${roundtrip_dir}" \
  --output "${staging_dir}/safetensors-equivalence.json"
verification_seconds=$(( $(date +%s) - verification_started ))

(
  cd "${staging_dir}"
  find . -type f \
    ! -name checkpoint-files.sha256 \
    ! -name conversion-manifest.json \
    ! -name safetensors-equivalence.json \
    -print0 \
    | sort -z \
    | xargs -0 --no-run-if-empty sha256sum
) > "${staging_dir}/checkpoint-files.sha256"

export SEARCH_R1_CONVERSION_SECONDS="${conversion_seconds}"
export SEARCH_R1_ROUNDTRIP_SECONDS="${roundtrip_seconds}"
export SEARCH_R1_VERIFICATION_SECONDS="${verification_seconds}"
export SEARCH_R1_CONVERSION_NODE="$(hostname)"
export SEARCH_R1_CONVERSION_IMAGE="${runtime_image}"
export SEARCH_R1_CONVERSION_MODEL="${model_path}"
export SEARCH_R1_CONVERSION_OUTPUT="${ref_load}"
export SEARCH_R1_CONVERSION_SLIME_HEAD="${actual_slime_head}"
export SEARCH_R1_CONVERSION_SLIME_PATCH_SHA="${actual_slime_patch_sha256}"
export SEARCH_R1_CONVERSION_NEMO_HEAD="$(git -C "${nemo_root}" rev-parse HEAD)"
"${python_bin}" - \
  "${staging_dir}/safetensors-equivalence.json" \
  "${staging_dir}/conversion-manifest.json" <<'PY'
import datetime
import json
import os
import sys

verification = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = {
    "schema_version": 1,
    "contract": "search-r1-slime-checkpoint-conversion-v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "node": os.environ["SEARCH_R1_CONVERSION_NODE"],
    "runtime_image": os.environ["SEARCH_R1_CONVERSION_IMAGE"],
    "model_path": os.path.realpath(os.environ["SEARCH_R1_CONVERSION_MODEL"]),
    "output_path": os.environ["SEARCH_R1_CONVERSION_OUTPUT"],
    "slime_source_head": os.environ["SEARCH_R1_CONVERSION_SLIME_HEAD"],
    "slime_patch_sha256": os.environ["SEARCH_R1_CONVERSION_SLIME_PATCH_SHA"],
    "nemo_adapter_head": os.environ["SEARCH_R1_CONVERSION_NEMO_HEAD"],
    "conversion_seconds": int(os.environ["SEARCH_R1_CONVERSION_SECONDS"]),
    "roundtrip_seconds": int(os.environ["SEARCH_R1_ROUNDTRIP_SECONDS"]),
    "verification_seconds": int(os.environ["SEARCH_R1_VERIFICATION_SECONDS"]),
    "verification": verification,
    "timed_benchmark_work": False,
}
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(manifest, output, indent=2, sort_keys=True)
    output.write("\n")
PY

mv -- "${staging_dir}" "${ref_load}"
staging_dir=""
# Conversion normally runs as root inside the pinned slime container. Keep the
# verified checkpoint immutable to non-owners, but make directories traversable
# and artifacts readable by the host-side evidence collector.
chmod -R a+rX -- "${ref_load}"
echo "SEARCH_R1_SLIME_CHECKPOINT_PREP_PASS ${ref_load}"
