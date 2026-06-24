from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _run_id(value: object) -> str:
    if hasattr(value, "run_id"):
        return str(value.run_id)
    return str(value)


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: object) -> Path:
        return self.root / _run_id(run_id)

    def task_state_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "task_state.json"

    def trace_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "trace.jsonl"

    def report_path(self, run_id: object) -> Path:
        return self.run_dir(run_id) / "report.json"

    def start_run(self, task_state) -> Path:
        run_dir = self.run_dir(task_state)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.write_task_state(task_state)
        return run_dir

    def write_task_state(self, task_state) -> Path:
        path = self.task_state_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, task_state.to_dict())
        return path

    def append_trace(self, task_state, event: dict) -> Path:
        path = self.trace_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        return path

    def write_report(self, task_state, report: dict) -> Path:
        path = self.report_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, report)
        return path

    def load_report(self, run_id: object) -> dict:
        path = self.report_path(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)
