#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-reg2026:v0.6.0}"
MODEL_DIR="${SCRIPT_DIR}/model"
LOCK="${SCRIPT_DIR}/configs/artifacts-v0.6.0.json"

command -v docker >/dev/null 2>&1 || {
  printf 'docker is required\n' >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  printf 'python3 is required\n' >&2
  exit 1
}

python3 "${SCRIPT_DIR}/scripts/verify_model_assets.py" \
  --root "${MODEL_DIR}" \
  --lock "${LOCK}"

build_args=(
  --platform linux/amd64
  --build-arg APP_VERSION=0.6.0
  --tag "${DOCKER_IMAGE_TAG}"
)
if [[ -n "${DOCKER_QUIET_BUILD:-}" ]]; then
  build_args+=(--quiet)
fi

docker build "${build_args[@]}" "${SCRIPT_DIR}"
