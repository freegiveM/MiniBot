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


def _capture_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_cli_help_returns_zero_and_prints_banner(self):
        code, stdout, stderr = _capture_main(["--help"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("MiniBot", stdout)
        self.assertIn("/ o o \\", stdout)
        self.assertIn("minibot benchmark", stdout)
        self.assertIn("minibot metrics", stdout)

    def test_cli_fake_model_smoke_task_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
