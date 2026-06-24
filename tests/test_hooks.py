from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.hooks import (
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    HookManager,
    build_default_hook_manager,
)
from minibot.task_state import STOP_REASON_FINAL_ANSWER_RETURNED


class HookManagerTests(unittest.TestCase):
    def test_hook_failure_is_reported_but_does_not_stop_emit(self):
        manager = HookManager()
        seen = []

        def bad(payload):
            del payload
            raise RuntimeError("boom")

        def good(payload):
            seen.append(payload["event"])

        manager.register(EVENT_POST_TOOL_USE, bad, name="bad")
        manager.register(EVENT_POST_TOOL_USE, good, name="good")

        result = manager.emit(EVENT_POST_TOOL_USE, {"event": EVENT_POST_TOOL_USE})

        self.assertEqual(seen, [EVENT_POST_TOOL_USE])
        self.assertEqual(result.handler_count, 2)
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "bad_failed")
        self.assertEqual(result.errors[0]["message"], "boom")

    def test_hook_result_collects_metadata_updates(self):
        manager = HookManager()
        manager.register(
            EVENT_POST_TOOL_USE,
            lambda payload: {"metadata": {"tool_name": payload["tool_name"], "seen": True}},
            name="metadata",
        )

        result = manager.emit(EVENT_POST_TOOL_USE, {"tool_name": "read_file"})

        self.assertEqual(result.metadata_updates(), {"tool_name": "read_file", "seen": True})

    def test_unknown_event_is_rejected(self):
        manager = HookManager()

        with self.assertRaises(ValueError):
            manager.register("UnknownEvent", lambda payload: None)
        with self.assertRaises(ValueError):
            manager.emit("UnknownEvent", {})


class RuntimeHookTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str], hook_manager: HookManager | None = None) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient(outputs),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
            max_steps=3,
            hook_manager=hook_manager,
        )

    def load_report(self, root: Path, run_id: str) -> dict:
        return json.loads((root / ".minibot" / "runs" / run_id / "report.json").read_text(encoding="utf-8"))

    def load_trace_events(self, root: Path, run_id: str) -> list[dict]:
        trace_path = root / ".minibot" / "runs" / run_id / "trace.jsonl"
        return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    def test_runtime_runs_with_empty_hook_manager(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"], hook_manager=HookManager())

            self.assertEqual(agent.ask("hello"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            self.assertEqual(report["hooks"]["error_count"], 0)
            self.assertEqual([item["event"] for item in report["hooks"]["emissions"]], ["UserPromptSubmit", "Stop"])

    def test_post_tool_hook_failure_does_not_swallow_tool_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            manager = HookManager()

            def bad_post_tool(payload):
                del payload
                raise RuntimeError("post failed")

            manager.register(EVENT_POST_TOOL_USE, bad_post_tool, name="post_tool")
            outputs = [
                '<tool>{"name":"read_file","args":{"path":"README.md"}}</tool>',
                "<final>Done.</final>",
            ]
            agent = self.build_agent(root, outputs, hook_manager=manager)

            self.assertEqual(agent.ask("read README"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            saved = agent.session_store.load(agent.session["id"])
            report = self.load_report(root, run_id)
            tool_item = next(item for item in saved["history"] if item.get("role") == "tool")
            self.assertIn("demo", tool_item["content"])
            self.assertEqual(tool_item["metadata"]["hook_errors"][0]["code"], "post_tool_failed")
            self.assertEqual(report["hooks"]["error_count"], 1)
            self.assertTrue(any(event["event"] == "tool_executed" for event in self.load_trace_events(root, run_id)))

    def test_pre_tool_hook_cannot_bypass_permission_denial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            manager = HookManager()
            manager.register(
                EVENT_PRE_TOOL_USE,
                lambda payload: {"metadata": {"permission_behavior": "allow", "tool_name": payload["tool_name"]}},
                name="allowing_pre_hook",
            )
            agent = self.build_agent(root, [], hook_manager=manager)

            result = agent.run_tool("read_file", {"path": "../outside.txt"})

            self.assertIn("tool denied by permission policy", result)
            self.assertEqual(agent._last_tool_result_metadata["tool_status"], "rejected")
            self.assertEqual(agent._last_tool_result_metadata["permission_reason"], "path_escape")

    def test_stop_hook_extracts_explicit_memory_intent_to_pending_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Noted.</final>"])

            self.assertEqual(agent.ask("remember: always run hook tests after lifecycle changes"), "Noted.")

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            pending = agent.memory.store.load_pending()
            stop_outputs = [
                output
                for emission in report["hooks"]["emissions"]
                if emission["event"] == "Stop"
                for output in emission["outputs"]
            ]
            memory_outputs = [output["memory_extraction"] for output in stop_outputs if "memory_extraction" in output]
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["topic"], "user-preferences")
            self.assertEqual(memory_outputs[0]["candidate_count"], 1)
            self.assertEqual(agent.session["memory_maintenance"]["pending_count"], 1)

    def test_memory_extraction_hook_failure_is_reported_without_changing_final_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"], hook_manager=build_default_hook_manager())

            def fail_extraction(*args, **kwargs):
                del args, kwargs
                raise RuntimeError("extract boom")

            agent.memory.extract_memory_candidates = fail_extraction

            self.assertEqual(agent.ask("remember: always keep hook failures non-blocking"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            self.assertEqual(report["task_state"]["stop_reason"], STOP_REASON_FINAL_ANSWER_RETURNED)
            self.assertEqual(report["hooks"]["error_count"], 1)
            self.assertEqual(report["hooks"]["errors"][0]["code"], "memory_extraction_failed")
            self.assertEqual(report["hooks"]["errors"][0]["message"], "extract boom")


if __name__ == "__main__":
    unittest.main()
