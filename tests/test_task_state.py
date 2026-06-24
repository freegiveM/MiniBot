from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.task_state import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STOP_REASON_MODEL_ERROR,
    STOP_REASON_STEP_LIMIT_REACHED,
    TaskState,
)


class TaskStateTests(unittest.TestCase):
    def test_create_generates_run_id_and_defaults_to_running(self):
        state = TaskState.create(task_id="task_1", user_request="fix the bug")

        self.assertTrue(state.run_id.startswith("run_"))
        self.assertEqual(state.task_id, "task_1")
        self.assertEqual(state.user_request, "fix the bug")
        self.assertEqual(state.status, STATUS_RUNNING)
        self.assertEqual(state.tool_steps, 0)
        self.assertEqual(state.attempts, 0)
        self.assertEqual(state.last_tool, "")
        self.assertEqual(state.stop_reason, "")
        self.assertEqual(state.final_answer, "")

    def test_record_attempt_and_tool_updates_evaluation_counters(self):
        state = TaskState.create(task_id="task_1", user_request="inspect", run_id="run_1")

        state.record_attempt()
        state.record_tool("read_file")

        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.tool_steps, 1)
        self.assertEqual(state.last_tool, "read_file")

    def test_finish_success_sets_completed_status_reason_and_answer(self):
        state = TaskState.create(task_id="task_1", user_request="fix", run_id="run_1")

        state.finish_success("Done")

        self.assertEqual(state.status, STATUS_COMPLETED)
        self.assertEqual(state.stop_reason, STOP_REASON_FINAL_ANSWER_RETURNED)
        self.assertEqual(state.final_answer, "Done")

    def test_stop_sets_terminal_status_reason_and_optional_answer(self):
        state = TaskState.create(task_id="task_1", user_request="fix", run_id="run_1")

        state.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer="Stopped")

        self.assertEqual(state.status, STATUS_STOPPED)
        self.assertEqual(state.stop_reason, STOP_REASON_STEP_LIMIT_REACHED)
        self.assertEqual(state.final_answer, "Stopped")

    def test_stop_can_mark_failed_or_blocked_with_stable_reason(self):
        failed = TaskState.create(task_id="task_1", user_request="fix", run_id="run_1")
        blocked = TaskState.create(task_id="task_2", user_request="fix", run_id="run_2")

        failed.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED)
        blocked.stop(STOP_REASON_MODEL_ERROR, status=STATUS_BLOCKED)

        self.assertEqual(failed.status, STATUS_FAILED)
        self.assertEqual(blocked.status, STATUS_BLOCKED)
        self.assertEqual(failed.stop_reason, STOP_REASON_MODEL_ERROR)
        self.assertEqual(blocked.stop_reason, STOP_REASON_MODEL_ERROR)

    def test_to_dict_from_dict_round_trips_serialized_state(self):
        state = TaskState.create(task_id="task_1", user_request="fix", run_id="run_1")
        state.record_attempt()
        state.record_tool("read_file")
        state.finish_success("Done")

        restored = TaskState.from_dict(state.to_dict())

        self.assertEqual(restored, state)
        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_from_dict_defaults_optional_snapshot_fields(self):
        state = TaskState.from_dict(
            {
                "run_id": "run_1",
                "task_id": "task_1",
                "user_request": "fix",
            }
        )

        self.assertEqual(state.status, STATUS_RUNNING)
        self.assertEqual(state.tool_steps, 0)
        self.assertEqual(state.attempts, 0)
        self.assertEqual(state.last_tool, "")
        self.assertEqual(state.stop_reason, "")
        self.assertEqual(state.final_answer, "")

    def test_from_dict_rejects_unknown_status_and_stop_reason(self):
        base = {
            "run_id": "run_1",
            "task_id": "task_1",
            "user_request": "fix",
        }

        with self.assertRaises(ValueError):
            TaskState.from_dict({**base, "status": "succeeded"})
        with self.assertRaises(ValueError):
            TaskState.from_dict({**base, "stop_reason": "free text"})
        with self.assertRaises(ValueError):
            TaskState.from_dict({**base, "tool_steps": -1})

    def test_stop_rejects_running_status_and_free_text_reason(self):
        state = TaskState.create(task_id="task_1", user_request="fix", run_id="run_1")

        with self.assertRaises(ValueError):
            state.stop(STOP_REASON_MODEL_ERROR, status=STATUS_RUNNING)
        with self.assertRaises(ValueError):
            state.stop("free text")


if __name__ == "__main__":
    unittest.main()
