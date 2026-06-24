from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.phase import Phase, phase_after_tool


class MiniBotScaffoldTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str], approval_policy: str = "auto") -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient(outputs),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy=approval_policy,
        )

    def test_fake_model_final_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"])

            self.assertEqual(agent.ask("inspect project"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            run_dir = root / ".minibot" / "runs" / run_id
            self.assertTrue((run_dir / "task_state.json").exists())
            self.assertTrue((run_dir / "trace.jsonl").exists())
            self.assertTrue((run_dir / "report.json").exists())
            saved = json.loads(agent.session_path.read_text(encoding="utf-8"))
            self.assertNotIn("relevant_memory", json.dumps(saved))
            self.assertEqual(saved["memory"]["working"]["current_phase"], Phase.FINALIZE.value)

    def test_read_file_records_file_access_without_file_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
            outputs = [
                '<tool>{"name":"read_file","args":{"path":"app.py","start":1,"end":2}}</tool>',
                "<final>Read it.</final>",
            ]
            agent = self.build_agent(root, outputs)

            self.assertEqual(agent.ask("read app"), "Read it.")

            memory = agent.session["memory"]
            self.assertIn("app.py", memory["file_access"])
            self.assertIn("app.py", memory["working"]["recent_files"])
            self.assertNotIn("file_summaries", memory)
            self.assertEqual(memory["working"]["current_phase"], Phase.FINALIZE.value)

    def test_context_order_keeps_current_request_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, [])

            prompt, metadata = agent.context_manager.build("keep this exact request")

            self.assertEqual(metadata["section_order"][-1], "current_request")
            self.assertTrue(prompt.endswith("Current user request:\nkeep this exact request"))
            self.assertLess(prompt.index("Relevant memory:"), prompt.index("Current user request:"))

    def test_phase_after_tool_prefers_verify_for_test_commands(self):
        phase = phase_after_tool("run_shell", {"command": "python -m unittest"}, {"tool_status": "succeeded"})
        self.assertEqual(phase, Phase.VERIFY)


if __name__ == "__main__":
    unittest.main()

