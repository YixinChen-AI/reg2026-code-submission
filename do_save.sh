#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-reg2026:v0.6.0}"
OUTPUT_PATH="${SCRIPT_DIR}/reg2026_v0.6.0_amd64.tar.gz"
CHECKSUM_PATH="${OUTPUT_PATH}.sha256"
PROVENANCE_PATH="${OUTPUT_PATH}.provenance.json"

DOCKER_QUIET_BUILD=1 "${SCRIPT_DIR}/do_build.sh"

tmp_path="${OUTPUT_PATH}.tmp"
trap 'rm -f "${tmp_path}"' EXIT

docker image inspect "${DOCKER_IMAGE_TAG}" >/dev/null
docker save "${DOCKER_IMAGE_TAG}" | gzip -1 > "${tmp_path}"
mv "${tmp_path}" "${OUTPUT_PATH}"
trap - EXIT

archive_sha256="$(shasum -a 256 "${OUTPUT_PATH}" | awk '{print $1}')"
image_id="$(docker image inspect --format '{{.Id}}' "${DOCKER_IMAGE_TAG}")"
commit="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"
printf '%s  %s\n' "${archive_sha256}" "$(basename "${OUTPUT_PATH}")" > "${CHECKSUM_PATH}"
python3 - "${PROVENANCE_PATH}" "${commit}" "${image_id}" "${archive_sha256}" <<'PY'
import json
import pathlib
import sys

path, commit, image_id, archive_sha256 = sys.argv[1:]
pathlib.Path(path).write_text(
    json.dumps(
        {
            "release": "0.6.0",
            "source_commit": commit,
            "image_id": image_id,
            "archive_sha256": archive_sha256,
            "artifact_lock": "configs/artifacts-v0.6.0.json",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

printf 'saved %s\n' "${OUTPUT_PATH}"
printf 'checksum %s\n' "${CHECKSUM_PATH}"
printf 'provenance %s\n' "${PROVENANCE_PATH}"
