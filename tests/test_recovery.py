from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.context_manager import ContextManager
from minibot.recovery import (
    ACTION_REANCHOR_PROMPT,
    ACTION_RETRY_MODEL,
    RECOVERY_CURRENT_REQUEST_TOO_LONG,
    RECOVERY_STALE_FILE_STATE,
    RECOVERY_TOOL_SCHEMA_ERROR,
    RECOVERY_TRANSIENT_MODEL_ERROR,
    RECOVERY_WORKSPACE_DRIFT,
)
from minibot.task_state import STOP_REASON_PROMPT_TOO_LONG


class FlakyModelClient(FakeModelClient):
    def __init__(self, outputs=None):
        super().__init__(outputs)
        self.failures_remaining = 1

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        if self.failures_remaining:
            self.failures_remaining -= 1
            self.prompts.append(prompt)
            raise RuntimeError("temporary outage")
        return super().complete(prompt, max_new_tokens, **kwargs)


class RecoveryTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str], model_client=None, max_steps: int = 4) -> MiniBot:
        return MiniBot(
            model_client=model_client or FakeModelClient(outputs),
            workspace=WorkspaceContext.build(root, repo_root_override=root),
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
            max_steps=max_steps,
        )

    def load_trace_events(self, root: Path, run_id: str) -> list[dict]:
        trace_path = root / ".minibot" / "runs" / run_id / "trace.jsonl"
        return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    def load_report(self, root: Path, run_id: str) -> dict:
        return json.loads((root / ".minibot" / "runs" / run_id / "report.json").read_text(encoding="utf-8"))

    def test_invalid_tool_args_return_observation_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(
                root,
                [
                    '<tool>{"name":"patch_file","args":{"path":"README.md"}}</tool>',
                    "<final>Recovered.</final>",
                ],
            )

            self.assertEqual(agent.ask("patch"), "Recovered.")

            run_id = agent.session["runs"]["last_run_id"]
            events = self.load_trace_events(root, run_id)
            report = self.load_report(root, run_id)
            tool_item = next(item for item in agent.session["history"] if item.get("role") == "tool")
            recovery_events = report["recovery"]["events"]

            self.assertIn("recoverable tool_schema_error", tool_item["content"])
            self.assertTrue(any(event["event"] == "tool_rejected" for event in events))
            self.assertTrue(any(item["kind"] == RECOVERY_TOOL_SCHEMA_ERROR for item in recovery_events))

    def test_transient_model_error_retries_once_and_reports_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            model = FlakyModelClient(["<final>Recovered after retry.</final>"])
            agent = self.build_agent(root, [], model_client=model)

            self.assertEqual(agent.ask("hello"), "Recovered after retry.")

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            recovery = report["recovery"]["events"][0]
            self.assertEqual(recovery["kind"], RECOVERY_TRANSIENT_MODEL_ERROR)
            self.assertEqual(recovery["action"], ACTION_RETRY_MODEL)
            self.assertEqual(len(model.prompts), 2)

    def test_current_request_too_long_stops_without_calling_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            model = FakeModelClient(["<final>Should not run.</final>"])
            agent = self.build_agent(root, [], model_client=model)
            agent.context_manager = ContextManager(agent, total_budget=240)

            final = agent.ask("x" * 300)

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            state = report["task_state"]
            self.assertIn("Current user request is too long", final)
            self.assertEqual(state["stop_reason"], STOP_REASON_PROMPT_TOO_LONG)
            self.assertEqual(report["recovery"]["events"][-1]["kind"], RECOVERY_CURRENT_REQUEST_TOO_LONG)
            self.assertEqual(model.prompts, [])

    def test_reactive_compaction_does_not_trim_protected_sections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("WORKSPACE_MARKER " + ("w" * 1200), encoding="utf-8")
            agent = self.build_agent(root, [])
            agent.session["history"] = [{"role": "assistant", "content": "old " + ("h" * 1000)}]
            memory_dir = root / ".minibot" / "memory"
            memory_dir.mkdir(parents=True)
            (memory_dir / "MEMORY.md").write_text("MEMORY_MARKER " + ("m" * 1000), encoding="utf-8")

            prompt, metadata = ContextManager(agent, total_budget=700).build_prompt("continue")

            compacted_sections = [event["section"] for event in metadata["compact_summary"]["events"]]
            self.assertTrue(set(compacted_sections).issubset({"history", "relevant_memory", "memory_index"}))
            self.assertFalse(metadata["sections"]["identity"]["truncated"])
            self.assertFalse(metadata["sections"]["workspace"]["truncated"])
            self.assertFalse(metadata["sections"]["tools"]["truncated"])
            self.assertFalse(metadata["sections"]["current_request"]["truncated"])
            self.assertIn("WORKSPACE_MARKER", prompt)

    def test_stale_write_target_returns_reread_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("alpha\n", encoding="utf-8")
            agent = self.build_agent(
                root,
                [
                    '<tool>{"name":"patch_file","args":{"path":"README.md","old_text":"alpha","new_text":"beta"}}</tool>',
                    "<final>Will reread first.</final>",
                ],
            )
            agent.memory.record_file_access("README.md")
            agent.memory.mark_file_stale("README.md")

            self.assertEqual(agent.ask("patch stale file"), "Will reread first.")

            run_id = agent.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            self.assertEqual((root / "README.md").read_text(encoding="utf-8"), "alpha\n")
            self.assertTrue(any(item["kind"] == RECOVERY_STALE_FILE_STATE for item in report["recovery"]["events"]))

    def test_workspace_drift_is_reported_on_resumed_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("version one\n", encoding="utf-8")
            store = SessionStore(root / ".minibot" / "sessions")
            first = MiniBot(
                model_client=FakeModelClient(["<final>First.</final>"]),
                workspace=WorkspaceContext.build(root, repo_root_override=root),
                session_store=store,
                approval_policy="auto",
            )
            self.assertEqual(first.ask("first"), "First.")
            session_id = first.session["id"]
            (root / "README.md").write_text("version two\n", encoding="utf-8")
            resumed = MiniBot.from_session(
                FakeModelClient(["<final>Second.</final>"]),
                WorkspaceContext.build(root, repo_root_override=root),
                store,
                session_id,
                approval_policy="auto",
            )

            self.assertEqual(resumed.ask("second"), "Second.")

            run_id = resumed.session["runs"]["last_run_id"]
            report = self.load_report(root, run_id)
            drift = next(item for item in report["recovery"]["events"] if item["kind"] == RECOVERY_WORKSPACE_DRIFT)
            self.assertEqual(drift["action"], ACTION_REANCHOR_PROMPT)
            self.assertTrue(report["recovery"]["workspace_drift_detected"])


if __name__ == "__main__":
    unittest.main()
