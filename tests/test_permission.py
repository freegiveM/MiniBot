from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.permission import (
    ACTION_ALLOW,
    ACTION_ASK,
    ACTION_DENY,
    POLICY_ASK,
    POLICY_AUTO,
    POLICY_DENY_RISKY,
    RISK_LEVEL_RISKY,
    PermissionPipeline,
    PermissionRequest,
)


class PermissionPipelineTests(unittest.TestCase):
    def test_permission_denies_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            decision = PermissionPipeline(root).check(
                PermissionRequest(
                    tool_name="read_file",
                    args={"path": "../secret.txt"},
                    workspace_root=root,
                )
            )

            self.assertEqual(decision.action, ACTION_DENY)
            self.assertEqual(decision.reason, "path_escape")

    def test_permission_requires_approval_for_safe_shell_command(self):
        with tempfile.TemporaryDirectory() as temp:
            decision = PermissionPipeline(temp, approval_policy=POLICY_ASK).check(
                PermissionRequest(
                    tool_name="run_shell",
                    args={"command": "python -m unittest tests.test_tools -v"},
                    risk_level=RISK_LEVEL_RISKY,
                    workspace_root=temp,
                )
            )

            self.assertEqual(decision.action, ACTION_ASK)
            self.assertEqual(decision.reason, "shell_command_requires_approval")
            self.assertEqual(decision.metadata["approval_policy"], POLICY_ASK)
            self.assertEqual(decision.metadata["command_category"], "test")

    def test_permission_routes_risky_shell_by_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            request = PermissionRequest(
                tool_name="run_shell",
                args={"command": "rm -rf build"},
                risk_level=RISK_LEVEL_RISKY,
                workspace_root=temp,
            )

            auto = PermissionPipeline(temp, approval_policy=POLICY_AUTO).check(request)
            ask = PermissionPipeline(temp, approval_policy=POLICY_ASK).check(request)
            deny = PermissionPipeline(temp, approval_policy=POLICY_DENY_RISKY).check(request)
            legacy_never = PermissionPipeline(temp, approval_policy="never").check(request)

            self.assertEqual(auto.action, ACTION_ALLOW)
            self.assertEqual(auto.reason, "shell_auto_approved")
            self.assertEqual(auto.metadata["approval_policy"], POLICY_AUTO)
            self.assertEqual(auto.metadata["command_category"], "risky")
            self.assertEqual(ask.action, ACTION_ASK)
            self.assertEqual(ask.reason, "shell_command_requires_approval")
            self.assertEqual(ask.metadata["command_category"], "risky")
            self.assertEqual(deny.action, ACTION_DENY)
            self.assertEqual(deny.reason, "shell_command_requires_approval")
            self.assertEqual(deny.metadata["command_category"], "risky")
            self.assertEqual(legacy_never.action, ACTION_DENY)

    def test_permission_read_only_denies_risky_write_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            decision = PermissionPipeline(temp, approval_policy=POLICY_AUTO, read_only=True).check(
                PermissionRequest(
                    tool_name="write_file",
                    args={"path": "out.txt", "content": "hello"},
                    risk_level=RISK_LEVEL_RISKY,
                    workspace_root=temp,
                )
            )

            self.assertEqual(decision.action, ACTION_DENY)
            self.assertEqual(decision.reason, "read_only")

    def test_permission_read_only_denies_run_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            decision = PermissionPipeline(temp, approval_policy=POLICY_AUTO, read_only=True).check(
                PermissionRequest(
                    tool_name="run_shell",
                    args={"command": "python -m unittest discover -s tests -v"},
                    risk_level=RISK_LEVEL_RISKY,
                    workspace_root=temp,
                )
            )

            self.assertEqual(decision.action, ACTION_DENY)
            self.assertEqual(decision.reason, "read_only")

    def test_runtime_returns_permission_deny_as_tool_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            (Path(temp) / "secret.txt").write_text("secret\n", encoding="utf-8")
            agent = MiniBot(
                model_client=FakeModelClient(
                    [
                        '<tool>{"name":"read_file","args":{"path":"../secret.txt"}}</tool>',
                        "<final>Handled denial.</final>",
                    ]
                ),
                workspace=WorkspaceContext.build(root, repo_root_override=root),
                session_store=SessionStore(root / ".minibot" / "sessions"),
                approval_policy=POLICY_AUTO,
            )

            self.assertEqual(agent.ask("read secret"), "Handled denial.")

            tool_items = [item for item in agent.session["history"] if item["role"] == "tool"]
            self.assertEqual(len(tool_items), 1)
            self.assertIn("tool denied by permission policy: path_escape", tool_items[0]["content"])
            self.assertEqual(tool_items[0]["metadata"]["tool_status"], "rejected")
            self.assertEqual(tool_items[0]["metadata"]["permission_action"], ACTION_DENY)
            self.assertEqual(tool_items[0]["metadata"]["permission_reason"], "path_escape")
            run_id = agent.session["runs"]["last_run_id"]
            trace_path = root / ".minibot" / "runs" / run_id / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            tool_event = next(event for event in events if event["event"] == "tool_executed")
            self.assertEqual(tool_event["permission_action"], ACTION_DENY)
            self.assertEqual(tool_event["permission_reason"], "path_escape")


if __name__ == "__main__":
    unittest.main()
