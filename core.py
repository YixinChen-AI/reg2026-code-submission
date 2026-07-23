"""Grand Challenge input and output helpers."""

import json
from pathlib import Path


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def get_interface_key() -> tuple[str, ...]:
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    return tuple(sorted(entry["socket"]["slug"] for entry in inputs))


def load_json_file(location: Path):
    with location.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(location: Path, content) -> None:
    with location.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)
