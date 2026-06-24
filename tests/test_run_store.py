from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.run_store import RunStore
from minibot.task_state import TaskState


class RunStoreTests(unittest.TestCase):
    def test_run_store_writes_task_trace_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            state = TaskState.create(task_id="task_1", user_request="hello", run_id="run_1")
            store = RunStore(Path(temp) / ".minibot" / "runs")

            run_dir = store.start_run(state)
            store.append_trace(state, {"event": "run_started"})
            store.write_report(state, {"task_state": state.to_dict(), "ok": True})

            self.assertEqual(run_dir, Path(temp) / ".minibot" / "runs" / "run_1")
            self.assertTrue(store.task_state_path(state).exists())
            self.assertTrue(store.trace_path(state).exists())
            self.assertTrue(store.report_path(state).exists())
            self.assertIn("run_started", store.trace_path(state).read_text(encoding="utf-8"))
            self.assertTrue(store.load_report(state)["ok"])

    def test_run_store_paths_accept_run_id_or_task_state(self):
        with tempfile.TemporaryDirectory() as temp:
            state = TaskState.create(task_id="task_1", user_request="hello", run_id="run_1")
            store = RunStore(Path(temp) / "runs")

            self.assertEqual(store.run_dir(state), store.run_dir("run_1"))
            self.assertEqual(store.task_state_path(state), store.task_state_path("run_1"))
            self.assertEqual(store.trace_path(state), store.trace_path("run_1"))
            self.assertEqual(store.report_path(state), store.report_path("run_1"))

    def test_write_task_state_persists_serialized_state(self):
        with tempfile.TemporaryDirectory() as temp:
            state = TaskState.create(task_id="task_1", user_request="hello", run_id="run_1")
            state.record_attempt()
            state.record_tool("read_file")
            store = RunStore(Path(temp) / "runs")

            store.write_task_state(state)

            payload = json.loads(store.task_state_path(state).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run_1")
            self.assertEqual(payload["attempts"], 1)
            self.assertEqual(payload["tool_steps"], 1)
            self.assertEqual(payload["last_tool"], "read_file")


if __name__ == "__main__":
    unittest.main()
