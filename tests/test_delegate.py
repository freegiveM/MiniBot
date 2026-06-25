from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.delegate import DelegateTask, bounded_delegate_observation


class DelegateTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str]) -> MiniBot:
        return MiniBot(
            model_client=FakeModelClient(outputs),
            workspace=WorkspaceContext.build(root, repo_root_override=root),
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
            max_steps=4,
        )

    def test_delegate_parent_observation_is_bounded(self):
        raw = {
            "summary": "checked runtime",
            "evidence": [{"file": f"f{i}.py", "line": i, "note": "n"} for i in range(20)],
            "files_read": [f"f{i}.py" for i in range(20)],
            "confidence": "high",
            "open_questions": [f"q{i}" for i in range(10)],
            "recommended_next_step": "patch runtime",
        }

        observation = bounded_delegate_observation(raw)

        self.assertEqual(len(observation["evidence"]), 5)
        self.assertEqual(len(observation["files_read"]), 10)
        self.assertEqual(len(observation["open_questions"]), 5)
        self.assertEqual(observation["summary"], "checked runtime")

    def test_delegate_task_rejects_recursive_delegate_tool(self):
        with self.assertRaises(ValueError):
            DelegateTask(task="inspect", allowed_tools=("read_file", "delegate"))

    def test_delegate_runs_child_with_isolated_context_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("# Demo\nDelegate marker\n", encoding="utf-8")
            child_result = {
                "summary": "README contains the delegate marker.",
                "evidence": [{"file": "README.md", "line": 2, "note": "marker found"}],
                "files_read": ["README.md"],
                "confidence": "high",
                "open_questions": [],
                "recommended_next_step": "Use the finding in the parent answer.",
            }
            outputs = [
                '<tool>{"name":"delegate","args":{"id":"delegate_test","task":"inspect README for marker","allowed_tools":["read_file"],"max_steps":2}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":5}}</tool>',
                json.dumps(child_result),
                "<final>Parent used delegate.</final>",
            ]
            agent = self.build_agent(root, outputs)
            agent.session["history"].append({"role": "assistant", "content": "SECRET_PARENT_CONTEXT"})

            self.assertEqual(agent.ask("delegate README inspection"), "Parent used delegate.")

            artifact_path = root / ".minibot" / "delegates" / "delegate_test.json"
            self.assertTrue(artifact_path.exists())
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            parent_tool = next(item for item in agent.session["history"] if item.get("name") == "delegate")
            parent_observation = json.loads(parent_tool["content"].split("\n", 1)[1])
            report = json.loads(
                (root / ".minibot" / "runs" / agent.session["runs"]["last_run_id"] / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            delegate_prompts = [prompt for prompt in agent.model_client.prompts if "MiniBot delegate" in prompt]

            self.assertEqual(artifact["task"]["allowed_tools"], ["read_file"])
            self.assertEqual(artifact["child"]["depth"], 1)
            self.assertEqual(artifact["child"]["max_depth"], 1)
            self.assertTrue(artifact["child"]["read_only"])
            self.assertEqual(artifact["parent_observation"]["summary"], child_result["summary"])
            self.assertEqual(parent_observation["delegate_id"], "delegate_test")
            self.assertEqual(parent_tool["metadata"]["delegate_artifact"], ".minibot/delegates/delegate_test.json")
            self.assertEqual(report["delegate_artifacts"][0]["artifact_ref"], ".minibot/delegates/delegate_test.json")
            self.assertEqual(len(delegate_prompts), 2)
            self.assertTrue(all("SECRET_PARENT_CONTEXT" not in prompt for prompt in delegate_prompts))
            self.assertNotIn("- delegate:", delegate_prompts[0])


if __name__ == "__main__":
    unittest.main()
