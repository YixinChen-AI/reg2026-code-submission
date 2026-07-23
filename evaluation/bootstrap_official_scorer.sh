#!/usr/bin/env bash
set -euo pipefail

readonly REG2026_REPOSITORY="https://github.com/Haasha/REG2026.git"
readonly REG2026_COMMIT="ec9f1dcb9beb6096c6e8b634e2fee10bfcfae924"
readonly DESTINATION="${1:-.cache/REG2026}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone --filter=blob:none "${REG2026_REPOSITORY}" "${DESTINATION}"
fi

git -C "${DESTINATION}" fetch origin "${REG2026_COMMIT}"
git -C "${DESTINATION}" checkout --detach "${REG2026_COMMIT}"

printf 'Official scorer: %s/submission_evaluation_code\n' "${DESTINATION}"
printf 'Commit: %s\n' "$(git -C "${DESTINATION}" rev-parse HEAD)"
