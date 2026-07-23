from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from training.common import (
    GROUP_SEPARATOR,
    atomic_json,
    canonicalize_next_question,
    canonicalize_text,
    diagnosis_of,
    load_cases,
    verify_sha256,
)


def workflow(case: dict) -> list[dict]:
    for key in (
        "chain-of-thought",
        "chain_of_thought",
        "workflow",
        "workflow_steps",
        "reasoning_steps",
        "steps",
    ):
        value = case.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(f"case {case.get('id')} has no workflow")


def skeleton(case: dict) -> frozenset[tuple[str, str]]:
    return frozenset(
        (
            canonicalize_text(step.get("question")),
            canonicalize_next_question(step.get("next_question")),
        )
        for step in workflow(case)
    )


def valid_cot(cot: object) -> bool:
    return bool(
        isinstance(cot, list)
        and cot
        and all(
            isinstance(step, dict)
            and all(
                isinstance(step.get(key), str)
                for key in ("question", "answer", "next_question")
            )
            and bool(step["question"])
            and bool(step["answer"])
            for step in cot
        )
    )


def medoid(cases: list[dict]) -> list[dict]:
    groups: dict[frozenset[tuple[str, str]], list[dict]] = defaultdict(list)
    for case in cases:
        groups[skeleton(case)].append(case)
    if not groups:
        raise ValueError("cannot select a medoid from an empty group")
    selected = max(groups.values(), key=len)
    candidates = [workflow(case) for case in selected if valid_cot(workflow(case))]
    if not candidates:
        candidates = [workflow(case) for case in cases if valid_cot(workflow(case))]
    if not candidates:
        raise ValueError("group has no schema-valid workflow")
    return candidates[0]


def build_table(cases: list[dict]) -> dict:
    by_organ: dict[str, list[dict]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        organ = str(case.get("organ", "")).lower()
        diagnosis = diagnosis_of(case)
        if not organ:
            continue
        by_organ[organ].append(case)
        if diagnosis:
            by_group[(organ, diagnosis)].append(case)
    return {
        "organ_dx": {
            f"{organ}{GROUP_SEPARATOR}{diagnosis}": medoid(group)
            for (organ, diagnosis), group in sorted(by_group.items())
        },
        "organ": {organ: medoid(group) for organ, group in sorted(by_organ.items())},
        "__global__": medoid(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.cot, args.cot_sha256)
    if args.skip_existing and args.out.exists():
        print(f"SKIP {args.out}")
        return
    table = build_table(load_cases(args.cot))
    atomic_json(args.out, table)
    print(
        f"WROTE {args.out}: organ_dx={len(table['organ_dx'])} "
        f"organs={len(table['organ'])}"
    )


if __name__ == "__main__":
    main()
