from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.models import FakeModelClient
from minibot.repl import run_repl


class ReplTests(unittest.TestCase):
    def test_repl_runs_multiple_turns_in_one_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            stdin = io.StringIO("first turn\n/session\nsecond turn\n/exit\n")
            stdout = io.StringIO()

            code = run_repl(
                cwd=root,
                model_client=FakeModelClient(["<final>First answer.</final>", "<final>Second answer.</final>"]),
                approval_policy="auto",
                max_steps=2,
                input_stream=stdin,
                output_stream=stdout,
                error_stream=io.StringIO(),
            )

            self.assertEqual(code, 0)
            text = stdout.getvalue()
            self.assertIn("MiniBot REPL", text)
            self.assertIn("First answer.", text)
            self.assertIn("Second answer.", text)
            self.assertIn("Last run:", text)
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 1)
            saved = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["turn_count"], 2)
            self.assertEqual(len(saved["runs"]["recent_run_ids"]), 2)
            self.assertEqual([item["role"] for item in saved["history"]], ["user", "assistant", "user", "assistant"])
            run_reports = list((root / ".minibot" / "runs").glob("*/report.json"))
            self.assertEqual(len(run_reports), 2)

    def test_repl_reset_starts_new_session_without_exiting_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            stdin = io.StringIO("/session\n/reset\n/session\n/exit\n")
            stdout = io.StringIO()

            code = run_repl(
                cwd=root,
                model_client=FakeModelClient([]),
                approval_policy="auto",
                input_stream=stdin,
                output_stream=stdout,
                error_stream=io.StringIO(),
            )

            self.assertEqual(code, 0)
            self.assertIn("Session reset.", stdout.getvalue())
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 2)

    def test_repl_missing_resume_session_returns_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            stderr = io.StringIO()

            code = run_repl(
                cwd=root,
                model_client=FakeModelClient([]),
                resume="missing-session",
                input_stream=io.StringIO("/exit\n"),
                output_stream=io.StringIO(),
                error_stream=stderr,
            )

            self.assertEqual(code, 2)
            self.assertIn("could not resume session", stderr.getvalue())

    def test_repl_resume_reuses_existing_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")

            first_code = run_repl(
                cwd=root,
                model_client=FakeModelClient(["<final>First.</final>"]),
                approval_policy="auto",
                input_stream=io.StringIO("first\n/exit\n"),
                output_stream=io.StringIO(),
                error_stream=io.StringIO(),
            )
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            session_id = sessions[0].stem

            second_code = run_repl(
                cwd=root,
                model_client=FakeModelClient(["<final>Second.</final>"]),
                approval_policy="auto",
                resume=session_id,
                input_stream=io.StringIO("second\n/exit\n"),
                output_stream=io.StringIO(),
                error_stream=io.StringIO(),
            )

            self.assertEqual(first_code, 0)
            self.assertEqual(second_code, 0)
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 1)
            saved = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["id"], session_id)
            self.assertEqual(saved["turn_count"], 2)
            self.assertEqual([item["content"] for item in saved["history"]], ["first", "First.", "second", "Second."])


if __name__ == "__main__":
    unittest.main()
