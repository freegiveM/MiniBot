from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.evaluator import load_benchmark, run_fixed_benchmark


class EvaluatorTests(unittest.TestCase):
    def test_run_fixed_benchmark_uses_fresh_fixture_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "coding_tasks.json"
            artifact_path = root / "harness-regression-v2.json"
            workspace_root = root / "workspaces"

            artifact = run_fixed_benchmark(
                benchmark_path=benchmark_path,
                artifact_path=artifact_path,
                workspace_root=workspace_root,
            )

            self.assertGreater(artifact["summary"]["total_tasks"], 0)
            self.assertEqual(artifact["summary"]["passed"], 1)
            self.assertEqual(artifact["rows"][0]["fixture_copy_relpath"], "workspaces/patch_sample_text")
            self.assertTrue(artifact["rows"][0]["report_relpath"])
            self.assertTrue(artifact["rows"][0]["expected_artifact_exists"])
            self.assertTrue(artifact["rows"][0]["verifier_passed"])
            self.assertEqual(artifact["rows"][0]["stop_reason"], "final_answer_returned")
            self.assertTrue(artifact_path.exists())
            saved = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["passed"], 1)
            fixture_sample = benchmark_path.parent / "fixtures" / "bench_repo_patch" / "sample.txt"
            self.assertEqual(fixture_sample.read_text(encoding="utf-8"), "alpha\n")

    def test_load_benchmark_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixtures" / "bench_repo_patch"
            fixture.mkdir(parents=True)
            (fixture / "sample.txt").write_text("alpha", encoding="utf-8")
            base_task = {
                "id": "task_1",
                "prompt": "x",
                "fixture_repo": "fixtures/bench_repo_patch",
                "allowed_tools": ["read_file"],
                "step_budget": 1,
                "expected_artifact": "sample.txt",
                "verifier": "python -c \"exit(0)\"",
                "category": "text-edit",
            }
            cases = [
                [{**base_task, "id": "dup"}, {**base_task, "id": "dup"}],
                [{key: value for key, value in base_task.items() if key != "prompt"}],
                [{**base_task, "fixture_repo": "fixtures/missing"}],
                [{**base_task, "step_budget": 0}],
            ]
            for index, tasks in enumerate(cases):
                benchmark_path = root / f"coding_tasks_{index}.json"
                benchmark_path.write_text(
                    json.dumps({"schema_version": 1, "description": "bad", "tasks": tasks}, indent=2),
                    encoding="utf-8",
                )
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        load_benchmark(benchmark_path)


if __name__ == "__main__":
    unittest.main()
