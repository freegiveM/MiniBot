from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.context_manager import SECTION_ORDER, ContextManager
from minibot.task_state import TaskState


class ContextManagerTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str] | None = None) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient(outputs or []),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
        )

    def test_prompt_order_and_current_request_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            context = ContextManager(agent, total_budget=800)
            request = "keep this exact request\nwith trailing spaces   "

            prompt, metadata = context.build_prompt(request)

            self.assertEqual(metadata["section_order"], list(SECTION_ORDER))
            self.assertEqual(metadata["section_order"][-1], "current_request")
            self.assertTrue(prompt.endswith("Current user request:\n" + request))
            positions = [prompt.index(self._section_heading(name)) for name in SECTION_ORDER]
            self.assertEqual(positions, sorted(positions))
            self.assertFalse(metadata["sections"]["current_request"]["truncated"])
            self.assertTrue(metadata["sections"]["current_request"]["preserved"])

    def test_section_metadata_explains_source_size_and_truncation_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            _, metadata = agent.context_manager.build("inspect")

            self.assertEqual(set(metadata["sections"]), set(SECTION_ORDER))
            for name in SECTION_ORDER:
                section = metadata["sections"][name]
                self.assertIsInstance(section["source"], str)
                self.assertGreaterEqual(section["chars"], 0)
                self.assertGreaterEqual(section["raw_chars"], section["chars"])
                self.assertFalse(section["truncated"])
                self.assertEqual(section["truncation_reason"], "")

    def test_section_budget_can_truncate_non_current_request_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [{"role": "assistant", "content": "x" * 200}]
            context = ContextManager(agent, section_budgets={"history": 40, "current_request": 5})
            request = "current request must stay whole"

            prompt, metadata = context.build_prompt(request)

            history = metadata["sections"]["history"]
            current = metadata["sections"]["current_request"]
            self.assertTrue(history["truncated"])
            self.assertEqual(history["truncation_reason"], "section_budget_exceeded")
            self.assertLessEqual(history["chars"], 40)
            self.assertFalse(current["truncated"])
            self.assertTrue(prompt.endswith("Current user request:\n" + request))

    def test_task_state_section_renders_current_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            state = TaskState.create(task_id="task_1", user_request="inspect", run_id="run_1")
            state.record_attempt()
            state.record_tool("read_file")
            agent.current_task_state = state

            prompt, metadata = agent.context_manager.build("continue")

            self.assertIn("Task state:\n- run_id: run_1", prompt)
            self.assertIn("- task_id: task_1", prompt)
            self.assertIn("- status: running", prompt)
            self.assertIn("- tool_steps: 1", prompt)
            self.assertIn("- attempts: 1", prompt)
            self.assertIn("- last_tool: read_file", prompt)
            self.assertEqual(metadata["sections"]["task_state"]["source"], "runtime.current_task_state")

    def test_prompt_metadata_reaches_trace_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"])

            self.assertEqual(agent.ask("hello"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            run_dir = root / ".minibot" / "runs" / run_id
            events = [
                json.loads(line)
                for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            prompt_events = [event for event in events if event["event"] == "prompt_built"]
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_events[0]["prompt_metadata"]["section_order"], list(SECTION_ORDER))
            self.assertEqual(report["prompt_metadata"]["section_order"], list(SECTION_ORDER))
            self.assertIn("task_state", report["prompt_metadata"]["sections"])

    @staticmethod
    def _section_heading(name: str) -> str:
        return {
            "identity": "Identity:",
            "workspace": "Workspace:",
            "tools": "Tools:",
            "task_state": "Task state:",
            "working_memory": "Working memory:",
            "relevant_memory": "Relevant memory:",
            "memory_index": "Memory index:",
            "history": "Transcript:",
            "current_request": "Current user request:",
        }[name]


if __name__ == "__main__":
    unittest.main()
