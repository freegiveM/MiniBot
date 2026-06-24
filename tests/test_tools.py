from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.tools import (
    OBSERVATION_ERROR,
    OBSERVATION_SUCCEEDED,
    ToolRegistry,
    ToolSpec,
    normalize_tool_calls,
)


class ToolRegistryTests(unittest.TestCase):
    def build_agent(self, root: Path) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient([]),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
        )

    def test_tool_registry_validates_missing_required_args(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            with self.assertRaises(ValueError):
                agent.tools.dispatch("read_file", {}, agent)

    def test_tool_registry_rejects_unknown_tool_and_type_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            with self.assertRaises(ValueError):
                agent.tools.dispatch("unknown_tool", {}, agent)
            with self.assertRaises(ValueError):
                agent.tools.dispatch("read_file", {"path": "README.md", "start": "bad"}, agent)

    def test_tool_registry_dispatch_returns_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            observation = agent.tools.dispatch("list_files", {"path": "."}, agent)

            self.assertEqual(observation.status, OBSERVATION_SUCCEEDED)
            self.assertIn("README.md", observation.content)
            self.assertEqual(observation.metadata["affected_paths"], ["."])
            self.assertEqual(observation.error, "")

    def test_tool_registry_custom_tool_uses_schema_and_observation(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="echo", description="Echo text.", schema={"text": "str"}),
            lambda runtime_context, args: "echo:" + args["text"],
        )

        observation = registry.dispatch("echo", {"text": "hello"}, runtime_context=None)

        self.assertEqual(observation.status, OBSERVATION_SUCCEEDED)
        self.assertEqual(observation.content, "echo:hello")
        with self.assertRaises(ValueError):
            registry.dispatch("echo", {}, runtime_context=None)

    def test_tool_registry_handler_error_returns_error_observation(self):
        def boom(runtime_context, args):
            del runtime_context, args
            raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(ToolSpec(name="boom", description="Fail.", schema={}), boom)

        observation = registry.dispatch("boom", {}, runtime_context=None)

        self.assertEqual(observation.status, OBSERVATION_ERROR)
        self.assertEqual(observation.error, "boom")
        self.assertIn("tool error: boom", observation.content)

    def test_tool_call_batch_normalizes_single_and_multiple_calls(self):
        single = normalize_tool_calls({"name": "read_file", "args": {"path": "a.py"}})
        multiple = normalize_tool_calls(
            [
                {"name": "read_file", "args": {"path": "a.py"}},
                {"name": "read_file", "args": {"path": "b.py"}},
            ]
        )
        wrapped = normalize_tool_calls(
            {
                "calls": [
                    {"name": "list_files", "args": {"path": "."}},
                ]
            }
        )

        self.assertEqual(len(single), 1)
        self.assertEqual([call.args["path"] for call in multiple], ["a.py", "b.py"])
        self.assertEqual(wrapped.calls[0].name, "list_files")
        with self.assertRaises(ValueError):
            normalize_tool_calls([])


if __name__ == "__main__":
    unittest.main()
