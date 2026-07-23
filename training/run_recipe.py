from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TOKEN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def expand(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"recipe variable is not set: {key}")
        return variables[key]

    return TOKEN.sub(replace, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    variables = {key: str(value) for key, value in recipe.get("defaults", {}).items()}
    for assignment in args.set:
        if "=" not in assignment:
            parser.error(f"invalid --set value: {assignment}")
        key, value = assignment.split("=", 1)
        variables[key] = value
    missing = [
        key for key in recipe.get("required_variables", []) if not variables.get(key)
    ]
    if missing:
        parser.error(f"missing recipe variables: {', '.join(missing)}")
    active = args.from_stage is None
    found_through = args.through_stage is None
    for stage in recipe["stages"]:
        if stage["id"] == args.from_stage:
            active = True
        if not active:
            continue
        command = [
            sys.executable,
            "-m",
            stage["module"],
            *[expand(value, variables) for value in stage.get("args", [])],
        ]
        print(f"{stage['id']}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
        if stage["id"] == args.through_stage:
            found_through = True
            break
    if args.from_stage and not any(
        stage["id"] == args.from_stage for stage in recipe["stages"]
    ):
        parser.error(f"unknown --from-stage: {args.from_stage}")
    if not found_through:
        parser.error(f"unknown --through-stage: {args.through_stage}")


if __name__ == "__main__":
    main()
