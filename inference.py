"""Container entry point for the two REG2026 interfaces."""

import multiprocessing as mp
from pathlib import Path

from core import INPUT_PATH, OUTPUT_PATH, get_interface_key, write_json_file
from src.contracts import is_valid_cot
from src.interf0.model import TISSUE_ANSWER, predict_visual_context_response
from src.interf1.model import _FALLBACK, global_fallback_cot, predict_chain_of_thought


WORKER_TIMEOUT_S = 210.0


def _fallback_cot():
    fallback = global_fallback_cot()
    return fallback if is_valid_cot(fallback) else _FALLBACK


def _interf1_worker(wsi_path: str, connection) -> None:
    result = None
    try:
        result = predict_chain_of_thought(wsi_path=Path(wsi_path))
    except BaseException as error:
        print(f"[interf1-worker] {error!r}")
    try:
        connection.send(result)
    except BaseException:
        pass
    finally:
        connection.close()


def _close_worker(process, connection) -> None:
    if process is not None:
        try:
            if process.is_alive():
                process.terminate()
                process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)
        except BaseException:
            pass
        try:
            process.close()
        except BaseException:
            pass
    if connection is not None:
        try:
            connection.close()
        except BaseException:
            pass


def _interf1_isolated(wsi_path: Path):
    process = None
    parent_connection = None
    result = None
    try:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_interf1_worker,
            args=(str(wsi_path), child_connection),
            daemon=True,
        )
        process.start()
        child_connection.close()
        if parent_connection.poll(WORKER_TIMEOUT_S):
            try:
                result = parent_connection.recv()
            except (EOFError, OSError):
                pass
        else:
            print("[interf1] worker timed out")
    except BaseException as error:
        print(f"[interf1] process isolation failed: {error!r}")
    finally:
        _close_worker(process, parent_connection)
    return result if is_valid_cot(result) else _fallback_cot()


def _safe_write(path: Path, content) -> bool:
    try:
        write_json_file(path, content)
    except BaseException as error:
        print(f"[output] failed to write {path}: {error!r}")
        return False
    return True


def interf0_handler() -> int:
    try:
        answer = predict_visual_context_response(
            question_path=INPUT_PATH / "visual-context-question.json",
            roi_image_path=(
                INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg"
            ),
        )
        if not isinstance(answer, str) or not answer:
            answer = TISSUE_ANSWER
    except BaseException as error:
        print(f"[interf0] {error!r}")
        answer = TISSUE_ANSWER
    written = _safe_write(OUTPUT_PATH / "visual-context-response.json", answer)
    return 0 if written else 1


def interf1_handler() -> int:
    wsi_dir = INPUT_PATH / "images" / "whole-slide-image"
    try:
        wsi_paths = sorted(wsi_dir.glob("*.tif*"))
    except BaseException:
        wsi_paths = []
    result = _interf1_isolated(wsi_paths[0]) if wsi_paths else _fallback_cot()
    written = _safe_write(OUTPUT_PATH / "chain-of-thought.json", result)
    return 0 if written else 1


def run() -> int:
    try:
        interface_key = get_interface_key()
    except BaseException as error:
        print(f"[input] interface detection failed: {error!r}")
        interface_key = None

    handlers = {
        (
            "histopathology-region-of-interest-thumbnail",
            "visual-context-question",
        ): interf0_handler,
        ("whole-slide-image",): interf1_handler,
    }
    handler = handlers.get(interface_key)
    if handler is None:
        cot_written = _safe_write(
            OUTPUT_PATH / "chain-of-thought.json", _fallback_cot()
        )
        roi_written = _safe_write(
            OUTPUT_PATH / "visual-context-response.json", TISSUE_ANSWER
        )
        return 0 if cot_written and roi_written else 1

    try:
        return handler()
    except BaseException as error:
        print(f"[inference] {error!r}")
        if handler is interf1_handler:
            written = _safe_write(
                OUTPUT_PATH / "chain-of-thought.json", _fallback_cot()
            )
        else:
            written = _safe_write(
                OUTPUT_PATH / "visual-context-response.json", TISSUE_ANSWER
            )
        return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(run())
