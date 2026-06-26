from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext


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
            self.assertNotIn("checkpoints", saved)
            self.assertEqual(saved["runs"]["last_run_id"], run_id)

    def test_prefix_spells_out_tool_call_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, [])

            self.assertIn("Tool protocol:", agent.prefix)
            self.assertIn("Batch tools use", agent.prefix)
            self.assertIn("<final>...</final>", agent.prefix)
            self.assertIn('"name":"read_file"', agent.prefix)
            self.assertIn('"args":{"path":"README.md"', agent.prefix)
            self.assertIn("args_schema=", agent.prefix)
            self.assertIn("example_args=", agent.prefix)
            self.assertIn("schemas describe the args object", agent.prefix)
            self.assertIn("Do not put tool arguments at the top level", agent.prefix)

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

    def test_context_order_keeps_current_request_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, [])

            prompt, metadata = agent.context_manager.build("keep this exact request")

            self.assertEqual(metadata["section_order"][-1], "current_request")
            self.assertTrue(prompt.endswith("Current user request:\nkeep this exact request"))
            self.assertLess(prompt.index("Relevant memory:"), prompt.index("Current user request:"))

    def test_context_renders_truncated_tool_observation_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, [])
            agent.session["history"] = [
                {
                    "role": "tool",
                    "name": "read_file",
                    "args": {"path": "README.md", "start": 1, "end": 20},
                    "content": "preview only\n...[truncated 8000 chars]",
                    "metadata": {
                        "artifact_ref": "runs/run_1/trace.jsonl",
                        "content_chars": 9230,
                        "ignored_internal": "do not render",
                        "tool_status": "succeeded",
                        "truncated": True,
                    },
                }
            ]

            prompt, _ = agent.context_manager.build("continue")

            self.assertIn('[tool:read_file] args={"end": 20, "path": "README.md", "start": 1}', prompt)
            self.assertIn('"artifact_ref": "runs/run_1/trace.jsonl"', prompt)
            self.assertIn('"content_chars": 9230', prompt)
            self.assertIn('"truncated": true', prompt)
            self.assertIn("Observation preview:", prompt)
            self.assertIn("Re-run read_file/search", prompt)
            self.assertIn("artifact_ref is for audit, not resume context", prompt)
            self.assertNotIn("ignored_internal", prompt)

    def test_resume_uses_session_without_previous_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>First done.</final>"])

            self.assertEqual(agent.ask("first turn"), "First done.")
            session_id = agent.session["id"]
            shutil.rmtree(root / ".minibot" / "runs")

            resumed = MiniBot.from_session(
                model_client=FakeModelClient(["<final>Second done.</final>"]),
                workspace=WorkspaceContext.build(root),
                session_store=SessionStore(root / ".minibot" / "sessions"),
                session_id=session_id,
                approval_policy="auto",
            )
            prompt, metadata = resumed.context_manager.build("continue from session")

            self.assertIn("[user] first turn", prompt)
            self.assertIn("[assistant] First done.", prompt)
            self.assertEqual(metadata["section_order"][-1], "current_request")
            self.assertNotIn("checkpoints", resumed.session)

if __name__ == "__main__":
    unittest.main()
