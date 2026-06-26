from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.cli import main


def _capture_main(argv: list[str], stdin_text: str | None = None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    original_stdin = sys.stdin
    if stdin_text is not None:
        sys.stdin = io.StringIO(stdin_text)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
    finally:
        sys.stdin = original_stdin
    return code, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_cli_help_returns_zero_and_prints_banner(self):
        code, stdout, stderr = _capture_main(["--help"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("MiniBot", stdout)
        self.assertIn("/ o o \\", stdout)
        self.assertIn("minibot repl", stdout)
        self.assertIn("minibot benchmark", stdout)
        self.assertIn("minibot metrics", stdout)

    def test_cli_fake_model_smoke_task_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            (root / ".env").write_text("MINIBOT_MODEL_PROVIDER=fake\nMINIBOT_MODEL_NAME=fake-from-env\n", encoding="utf-8")

            code, stdout, stderr = _capture_main(
                [
                    "--cwd",
                    str(root),
                    "--approval",
                    "auto",
                    "--max-steps",
                    "2",
                    "--model-provider",
                    "fake",
                    "--fake-response",
                    "<final>CLI works.</final>",
                    "hello",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("CLI works.", stdout)
            run_root = root / ".minibot" / "runs"
            self.assertTrue(run_root.exists())
            self.assertTrue(any(path.name == "report.json" for path in run_root.rglob("report.json")))

    def test_cli_repl_command_runs_two_turn_fake_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")

            code, stdout, stderr = _capture_main(
                [
                    "repl",
                    "--cwd",
                    str(root),
                    "--approval",
                    "auto",
                    "--max-steps",
                    "2",
                    "--model-provider",
                    "fake",
                    "--fake-response",
                    "<final>CLI REPL first.</final>",
                ],
                stdin_text="first\nsecond\n/exit\n",
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("MiniBot REPL", stdout)
            self.assertIn("CLI REPL first.", stdout)
            self.assertIn("Done.", stdout)
            self.assertIn("Bye.", stdout)
            run_reports = list((root / ".minibot" / "runs").glob("*/report.json"))
            self.assertEqual(len(run_reports), 2)
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 1)
            saved = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["turn_count"], 2)

    def test_cli_repl_command_resumes_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")

            first_code, _, first_stderr = _capture_main(
                [
                    "repl",
                    "--cwd",
                    str(root),
                    "--approval",
                    "auto",
                    "--model-provider",
                    "fake",
                    "--fake-response",
                    "<final>First.</final>",
                ],
                stdin_text="first\n/exit\n",
            )
            session_id = next((root / ".minibot" / "sessions").glob("*.json")).stem

            second_code, stdout, second_stderr = _capture_main(
                [
                    "repl",
                    "--cwd",
                    str(root),
                    "--resume",
                    session_id,
                    "--approval",
                    "auto",
                    "--model-provider",
                    "fake",
                    "--fake-response",
                    "<final>Second.</final>",
                ],
                stdin_text="second\n/exit\n",
            )

            self.assertEqual(first_code, 0)
            self.assertEqual(second_code, 0)
            self.assertEqual(first_stderr, "")
            self.assertEqual(second_stderr, "")
            self.assertIn(session_id, stdout)
            sessions = list((root / ".minibot" / "sessions").glob("*.json"))
            self.assertEqual(len(sessions), 1)
            saved = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["turn_count"], 2)

    def test_cli_http_provider_reports_configuration_error_without_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")

            code, stdout, stderr = _capture_main(
                [
                    "--cwd",
                    str(root),
                    "--model-provider",
                    "http",
                    "--api-format",
                    "openai",
                    "--model-name",
                    "mini",
                    "--base-url",
                    "https://example.test/chat",
                    "hello",
                ]
            )

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("provider configuration error", stderr)
            self.assertIn("API key is required", stderr)

    def test_cli_benchmark_command_writes_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo_root = Path(__file__).resolve().parents[1]
            artifact_path = root / "harness.json"
            workspace_root = root / "workspaces"

            code, stdout, stderr = _capture_main(
                [
                    "benchmark",
                    "--benchmark-path",
                    str(repo_root / "benchmarks" / "coding_tasks.json"),
                    "--artifact-path",
                    str(artifact_path),
                    "--workspace-root",
                    str(workspace_root),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            summary = json.loads(stdout)
            self.assertGreater(summary["total_tasks"], 0)
            self.assertTrue(artifact_path.exists())

    def test_cli_real_benchmark_reports_configuration_error_without_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo_root = Path(__file__).resolve().parents[1]

            code, stdout, stderr = _capture_main(
                [
                    "benchmark",
                    "--cwd",
                    str(root),
                    "--benchmark-path",
                    str(repo_root / "benchmarks" / "coding_tasks.json"),
                    "--model-provider",
                    "http",
                    "--api-format",
                    "openai",
                    "--model-name",
                    "mini",
                    "--base-url",
                    "https://example.test/chat",
                    "--max-tasks",
                    "1",
                ]
            )

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("provider configuration error", stderr)
            self.assertIn("API key is required", stderr)

    def test_cli_real_benchmark_dry_run_writes_planned_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo_root = Path(__file__).resolve().parents[1]
            artifact_path = root / "harness-real-v1.json"

            code, stdout, stderr = _capture_main(
                [
                    "benchmark",
                    "--cwd",
                    str(root),
                    "--benchmark-path",
                    str(repo_root / "benchmarks" / "coding_tasks.json"),
                    "--artifact-path",
                    str(artifact_path),
                    "--real",
                    "--dry-run",
                    "--max-tasks",
                    "2",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            summary = json.loads(stdout)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["planned_tasks"], 2)
            self.assertEqual(summary["total_tasks"], 0)
            saved = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "real")
            self.assertEqual(saved["benchmark"]["selected_task_ids"][0], "docs_update_readme_status")
            self.assertEqual(saved["benchmark"]["selected_task_ids"][1], "text_replace_priority_label")

    def test_cli_metrics_command_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness_path = root / "harness.json"
            report_path = root / "report.md"
            harness_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "total_tasks": 1,
                            "passed": 1,
                            "failed": 0,
                            "pass_rate": 1.0,
                            "within_budget_rate": 1.0,
                            "verifier_pass_rate": 1.0,
                            "failure_category_counts": {},
                        },
                        "rows": [
                            {
                                "id": "ok",
                                "passed": True,
                                "within_budget": True,
                                "verifier_passed": True,
                                "tool_steps": 1,
                                "attempts": 2,
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            code, stdout, stderr = _capture_main(
                [
                    "--cwd",
                    str(root),
                    "metrics",
                    "--harness-artifact-path",
                    str(harness_path),
                    "--report-path",
                    str(report_path),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(report_path), stdout)
            self.assertTrue(report_path.exists())
            self.assertIn("MiniBot Benchmark Core Report", report_path.read_text(encoding="utf-8"))

    def test_cli_metrics_command_writes_methodology_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock_path = root / "harness-regression-v2.json"
            real_path = root / "harness-real-v1.json"
            report_path = root / "methodology.md"
            base_row = {
                "id": "ok",
                "category": "documentation",
                "passed": True,
                "within_budget": True,
                "verifier_passed": True,
                "tool_steps": 1,
                "attempts": 2,
                "failure_category": "",
            }
            mock_path.write_text(
                json.dumps({"mode": "mock", "summary": {"total_tasks": 1, "passed": 1}, "rows": [base_row]}),
                encoding="utf-8",
            )
            real_path.write_text(
                json.dumps(
                    {
                        "mode": "real",
                        "reproducibility": {"provider": "http", "model": "mini-real", "api_format": "openai"},
                        "summary": {"total_tasks": 1, "passed": 1},
                        "rows": [{**base_row, "mode": "real", "latency_ms": 10, "estimated_cost_usd": 0.0}],
                    }
                ),
                encoding="utf-8",
            )

            code, stdout, stderr = _capture_main(
                [
                    "--cwd",
                    str(root),
                    "metrics",
                    "--methodology-report",
                    "--harness-artifact-path",
                    str(mock_path),
                    "--real-harness-artifact-path",
                    str(real_path),
                    "--report-path",
                    str(report_path),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(report_path), stdout)
            self.assertTrue(report_path.exists())
            self.assertIn("MiniBot Benchmark Methodology Report", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
