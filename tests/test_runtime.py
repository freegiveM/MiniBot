from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.context_manager import ContextManager
from minibot.runtime import SESSION_TOOL_OBSERVATION_LIMIT
from minibot.task_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STOPPED,
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STOP_REASON_MODEL_ERROR,
    STOP_REASON_STEP_LIMIT_REACHED,
)


class RaisingModelClient:
    supports_prompt_cache = False
    last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        del prompt, max_new_tokens, kwargs
        raise RuntimeError("boom")


class RuntimeTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str], max_steps: int = 6) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient(outputs),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
            max_steps=max_steps,
        )

    def load_task_state(self, root: Path, run_id: str) -> dict:
        return json.loads((root / ".minibot" / "runs" / run_id / "task_state.json").read_text(encoding="utf-8"))

    def load_trace_events(self, root: Path, run_id: str) -> list[dict]:
        trace_path = root / ".minibot" / "runs" / run_id / "trace.jsonl"
        return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    def test_runtime_final_answer_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"])

            self.assertEqual(agent.ask("hello"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            run_dir = root / ".minibot" / "runs" / run_id
            state = self.load_task_state(root, run_id)
            self.assertTrue((run_dir / "task_state.json").exists())
            self.assertTrue((run_dir / "trace.jsonl").exists())
            self.assertTrue((run_dir / "report.json").exists())
            self.assertEqual(state["status"], STATUS_COMPLETED)
            self.assertEqual(state["stop_reason"], STOP_REASON_FINAL_ANSWER_RETURNED)

    def test_runtime_persists_bounded_session_history_for_tool_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            long_tail = "x" * (SESSION_TOOL_OBSERVATION_LIMIT + 500)
            (root / "README.md").write_text(f"# Demo\n{long_tail}\n", encoding="utf-8")
            outputs = [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
                "<final>Done.</final>",
            ]
            agent = self.build_agent(root, outputs)

            self.assertEqual(agent.ask("read README"), "Done.")
            saved = agent.session_store.load(agent.session["id"])

            self.assertEqual([item["role"] for item in saved["history"]], ["user", "assistant", "tool", "assistant"])
            self.assertEqual(saved["history"][1]["decision"], "tool")
            self.assertEqual(saved["history"][2]["name"], "read_file")
            self.assertIn("[truncated", saved["history"][2]["content"])
            self.assertTrue(saved["history"][2]["metadata"]["truncated"])
            self.assertIn("artifact_ref", saved["history"][2]["metadata"])
            self.assertNotIn("relevant_memory", saved)

    def test_runtime_trace_keeps_full_tool_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = "TRACE_FULL_RESULT_MARKER"
            long_tail = "x" * (SESSION_TOOL_OBSERVATION_LIMIT + 200) + marker
            (root / "README.md").write_text(f"# Demo\n{long_tail}\n", encoding="utf-8")
            outputs = [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
                "<final>Done.</final>",
            ]
            agent = self.build_agent(root, outputs)

            agent.ask("read README")

            run_id = agent.session["runs"]["last_run_id"]
            tool_events = [event for event in self.load_trace_events(root, run_id) if event["event"] == "tool_executed"]
            self.assertEqual(len(tool_events), 1)
            self.assertIn(marker, tool_events[0]["result"])
            self.assertGreater(tool_events[0]["result_chars"], SESSION_TOOL_OBSERVATION_LIMIT)

    def test_runtime_executes_tool_call_batch_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.py").write_text("A = 1\n", encoding="utf-8")
            (root / "b.py").write_text("B = 2\n", encoding="utf-8")
            outputs = [
                '<tool>[{"name":"read_file","args":{"path":"a.py"}},{"name":"read_file","args":{"path":"b.py"}}]</tool>',
                "<final>Done.</final>",
            ]
            agent = self.build_agent(root, outputs)

            self.assertEqual(agent.ask("read both"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            state = self.load_task_state(root, run_id)
            tool_events = [event for event in self.load_trace_events(root, run_id) if event["event"] == "tool_executed"]
            saved = agent.session_store.load(agent.session["id"])
            self.assertEqual(state["tool_steps"], 2)
            self.assertEqual([event["args"]["path"] for event in tool_events], ["a.py", "b.py"])
            self.assertEqual([item["role"] for item in saved["history"]], ["user", "assistant", "tool", "tool", "assistant"])

    def test_runtime_stops_when_step_budget_is_reached(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            outputs = ['<tool>{"name":"list_files","args":{"path":"."}}</tool>']
            agent = self.build_agent(root, outputs, max_steps=1)

            final = agent.ask("list once")

            run_id = agent.session["runs"]["last_run_id"]
            state = self.load_task_state(root, run_id)
            self.assertIn("step limit", final)
            self.assertEqual(state["status"], STATUS_STOPPED)
            self.assertEqual(state["stop_reason"], STOP_REASON_STEP_LIMIT_REACHED)
            self.assertEqual(state["tool_steps"], 1)
            self.assertEqual(state["last_tool"], "list_files")

    def test_runtime_marks_model_exception_as_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            workspace = WorkspaceContext.build(root)
            agent = MiniBot(
                model_client=RaisingModelClient(),
                workspace=workspace,
                session_store=SessionStore(root / ".minibot" / "sessions"),
                approval_policy="auto",
            )

            final = agent.ask("hello")

            run_id = agent.session["runs"]["last_run_id"]
            state = self.load_task_state(root, run_id)
            self.assertIn("model error", final)
            self.assertEqual(state["status"], STATUS_FAILED)
            self.assertEqual(state["stop_reason"], STOP_REASON_MODEL_ERROR)
            self.assertTrue(any(event["event"] == "model_error" for event in self.load_trace_events(root, run_id)))

    def test_runtime_records_context_compacted_trace_and_report_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"])
            agent.session["history"] = [{"role": "assistant", "content": "old context " + ("x" * 2500)}]
            agent.context_manager = ContextManager(agent, total_budget=2600)

            self.assertEqual(agent.ask("latest request"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            events = self.load_trace_events(root, run_id)
            compact_events = [event for event in events if event["event"] == "context_compacted"]
            report = json.loads((root / ".minibot" / "runs" / run_id / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(compact_events), 1)
            self.assertEqual(compact_events[0]["compact_summary"]["trigger"], "prompt_budget_exceeded")
            self.assertEqual(report["prompt_metadata"]["compact_summary"]["trigger"], "prompt_budget_exceeded")
            self.assertTrue(report["prompt_metadata"]["current_request_preserved"])


if __name__ == "__main__":
    unittest.main()
