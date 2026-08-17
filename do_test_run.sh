#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-reg2026:v0.6.0}"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
INTERFACES="${INTERFACES:-interf0 interf1}"
TMP_VOLUME="reg2026-test-tmp-$$"

"${SCRIPT_DIR}/do_build.sh"

gpu_args=()
if command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi >/dev/null 2>&1 \
  && docker run --rm --platform linux/amd64 --gpus all \
       --entrypoint true "${DOCKER_IMAGE_TAG}" >/dev/null 2>&1; then
  gpu_args=(--gpus all)
fi

docker volume create "${TMP_VOLUME}" >/dev/null
trap 'docker volume rm -f "${TMP_VOLUME}" >/dev/null 2>&1 || true' EXIT

validate_output() {
  local interface="$1"
  python3 - "${OUTPUT_DIR}/${interface}" "${interface}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
interface = sys.argv[2]

if interface == "interf0":
    value = json.loads((output_dir / "visual-context-response.json").read_text())
    if not isinstance(value, str) or not value:
        raise SystemExit("Interface 0 output must be a non-empty JSON string")
else:
    value = json.loads((output_dir / "chain-of-thought.json").read_text())
    if not isinstance(value, list) or not value:
        raise SystemExit("Interface 1 output must be a non-empty JSON array")
    for step in value:
        if set(step) != {"question", "answer", "next_question"}:
            raise SystemExit("Interface 1 step has an invalid schema")
        if not step["question"] or not step["answer"]:
            raise SystemExit("Interface 1 question and answer must be non-empty")
        if not isinstance(step["next_question"], str):
            raise SystemExit("Interface 1 next_question must be a string")
    if value[-1]["next_question"]:
        raise SystemExit("Interface 1 final next_question must be empty")
PY
}

run_interface() {
  local interface="$1"
  local input_path="${INPUT_DIR}/${interface}"
  local output_path="${OUTPUT_DIR}/${interface}"

  [[ -f "${input_path}/inputs.json" ]] || {
    printf 'missing test fixture: %s/inputs.json\n' "${input_path}" >&2
    exit 1
  }
  if [[ "${interface}" == "interf1" ]] \
    && ! compgen -G "${input_path}/images/whole-slide-image/*.tif*" >/dev/null; then
    printf 'missing Interface 1 test WSI under %s/images/whole-slide-image\n' \
      "${input_path}" >&2
    exit 1
  fi
  if [[ "${interface}" == "interf1" && "${#gpu_args[@]}" -eq 0 \
      && "${ALLOW_CPU_FALLBACK:-0}" != "1" ]]; then
    printf 'Interface 1 testing requires Docker GPU access; set ALLOW_CPU_FALLBACK=1 to test only the fallback path\n' >&2
    exit 1
  fi

  rm -rf "${output_path}"
  mkdir -p "${output_path}"

  docker run --rm \
    --platform linux/amd64 \
    --network none \
    "${gpu_args[@]}" \
    --user "$(id -u):$(id -g)" \
    --volume "${input_path}:/input:ro" \
    --volume "${output_path}:/output" \
    --volume "${TMP_VOLUME}:/tmp" \
    "${DOCKER_IMAGE_TAG}"

  validate_output "${interface}"
  printf 'validated %s output\n' "${interface}"
}

read -r -a selected_interfaces <<< "${INTERFACES}"
for interface in "${selected_interfaces[@]}"; do
  case "${interface}" in
    interf0|interf1) run_interface "${interface}" ;;
    *)
      printf 'unsupported interface selector: %s\n' "${interface}" >&2
      exit 1
      ;;
  esac
done
