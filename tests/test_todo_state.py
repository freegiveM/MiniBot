from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.todo_state import (
    TODO_BLOCKED,
    TODO_COMPLETED,
    TODO_IN_PROGRESS,
    TODO_PENDING,
    TodoState,
)


class TodoStateTests(unittest.TestCase):
    def test_rejects_more_than_one_in_progress_item(self):
        state = TodoState()

        with self.assertRaises(ValueError):
            state.set_items(
                [
                    {"id": "todo_1", "content": "Inspect spec", "status": TODO_IN_PROGRESS},
                    {"id": "todo_2", "content": "Patch code", "status": TODO_IN_PROGRESS},
                ]
            )

        self.assertEqual(state.to_dict(), {"items": []})

    def test_to_dict_from_dict_round_trips_items(self):
        state = TodoState()
        state.set_items(
            [
                {
                    "id": "todo_1",
                    "content": "Inspect spec",
                    "status": TODO_COMPLETED,
                    "created_at": "created",
                    "updated_at": "updated",
                },
                {
                    "id": "todo_2",
                    "content": "Follow up on blocked test",
                    "status": TODO_BLOCKED,
                    "created_at": "created2",
                    "updated_at": "updated2",
                },
            ]
        )

        restored = TodoState.from_dict(state.to_dict())

        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_update_preserves_created_at_and_refreshes_updated_at(self):
        state = TodoState(
            [
                {
                    "id": "todo_1",
                    "content": "Inspect spec",
                    "status": TODO_PENDING,
                    "created_at": "created",
                    "updated_at": "old",
                }
            ]
        )

        item = state.update("todo_1", content="Inspect spec and code", status=TODO_IN_PROGRESS)

        self.assertEqual(item.created_at, "created")
        self.assertNotEqual(item.updated_at, "old")
        self.assertEqual(item.content, "Inspect spec and code")
        self.assertEqual(item.status, TODO_IN_PROGRESS)

    def test_update_rolls_back_when_in_progress_invariant_would_break(self):
        state = TodoState(
            [
                {"id": "todo_1", "content": "Inspect spec", "status": TODO_IN_PROGRESS},
                {"id": "todo_2", "content": "Patch code", "status": TODO_PENDING},
            ]
        )

        with self.assertRaises(ValueError):
            state.update("todo_2", status=TODO_IN_PROGRESS)

        self.assertEqual(state.items[0].status, TODO_IN_PROGRESS)
        self.assertEqual(state.items[1].status, TODO_PENDING)

    def test_rejects_duplicate_ids_and_unknown_status(self):
        state = TodoState()

        with self.assertRaises(ValueError):
            state.set_items(
                [
                    {"id": "todo_1", "content": "Inspect spec", "status": TODO_PENDING},
                    {"id": "todo_1", "content": "Patch code", "status": TODO_PENDING},
                ]
            )
        with self.assertRaises(ValueError):
            state.set_items([{"id": "todo_2", "content": "Patch code", "status": "running"}])

    def test_render_for_prompt_is_bounded(self):
        state = TodoState()
        for index in range(3):
            state.append(f"Task {index}", id=f"todo_{index}")

        rendered = state.render_for_prompt(max_items=2)

        self.assertIn("Todo plan:", rendered)
        self.assertIn("todo_0", rendered)
        self.assertIn("todo_1", rendered)
        self.assertNotIn("todo_2", rendered)
        self.assertIn("1 more", rendered)


class TodoWriteRuntimeTests(unittest.TestCase):
    def build_agent(self, root: Path, model: FakeModelClient) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=model,
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
            max_steps=3,
        )

    def test_todo_write_persists_session_prompt_and_report_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            items = [
                {"id": "plan_1", "content": "Read the Stage 9 spec", "status": TODO_COMPLETED},
                {"id": "plan_2", "content": "Wire todo_state into runtime", "status": TODO_IN_PROGRESS},
                {"id": "plan_3", "content": "Run targeted tests", "status": TODO_PENDING},
            ]
            tool_payload = json.dumps(
                {
                    "name": "todo_write",
                    "args": {"items_json": json.dumps(items)},
                }
            )
            model = FakeModelClient([f"<tool>{tool_payload}</tool>", "<final>Done.</final>"])
            agent = self.build_agent(root, model)

            self.assertEqual(agent.ask("implement stage 9"), "Done.")

            saved = agent.session_store.load(agent.session["id"])
            run_id = saved["runs"]["last_run_id"]
            report = json.loads((root / ".minibot" / "runs" / run_id / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["todo_state"]["items"][1]["status"], TODO_IN_PROGRESS)
            self.assertEqual(report["todo_state"], saved["todo_state"])
            self.assertEqual(report["todo_state"]["items"][0]["id"], "plan_1")
            self.assertGreaterEqual(len(model.prompts), 2)
            self.assertIn("Todo plan:", model.prompts[1])
            self.assertIn("[in_progress] plan_2: Wire todo_state into runtime", model.prompts[1])

    def test_todo_write_rejects_invalid_plan_without_mutating_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            model = FakeModelClient([])
            agent = self.build_agent(root, model)
            invalid_items = [
                {"id": "todo_1", "content": "Inspect spec", "status": TODO_IN_PROGRESS},
                {"id": "todo_2", "content": "Patch code", "status": TODO_IN_PROGRESS},
            ]

            result = agent.run_tool("todo_write", {"items_json": json.dumps(invalid_items)})

            self.assertIn("tool error", result)
            self.assertIn("only one todo item may be in_progress", result)
            self.assertEqual(agent.todo_state.to_dict(), {"items": []})


if __name__ == "__main__":
    unittest.main()
