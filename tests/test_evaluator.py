from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.evaluator import FAILURE_PROVIDER_ERROR, MODE_REAL, load_benchmark, run_fixed_benchmark
from minibot.models import FakeModelClient


class FailingProviderModel:
    supports_prompt_cache = False
    model = "failing-real"

    def __init__(self):
        self.last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        del prompt, max_new_tokens, kwargs
        self.last_completion_metadata = {
            "provider": "http",
            "model": self.model,
            "error_category": "provider_network_error",
        }
        raise RuntimeError("provider unavailable")


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

            self.assertEqual(artifact["summary"]["total_tasks"], 12)
            self.assertEqual(artifact["summary"]["passed"], 12)
            self.assertEqual(artifact["summary"]["failed"], 0)
            self.assertEqual(artifact["summary"]["category_pass_rates"]["documentation"], 1.0)
            self.assertEqual(artifact["summary"]["category_summary"]["tool-boundary"]["total_tasks"], 2)
            self.assertGreater(artifact["summary"]["median_tool_steps"], 0)
            self.assertEqual(artifact["rows"][0]["fixture_copy_relpath"], "workspaces/docs_update_readme_status")
            self.assertTrue(artifact["rows"][0]["report_relpath"])
            self.assertTrue(artifact["rows"][0]["expected_artifact_exists"])
            self.assertTrue(artifact["rows"][0]["verifier_passed"])
            self.assertEqual(artifact["rows"][0]["stop_reason"], "final_answer_returned")
            context_row = next(row for row in artifact["rows"] if row["id"] == "context_compaction_preserves_request")
            self.assertEqual(context_row["report"]["prompt_metadata"]["compact_trigger"], "prompt_budget_exceeded")
            memory_row = next(row for row in artifact["rows"] if row["id"] == "memory_read_project_decision")
            memory_trace = (root / memory_row["trace_relpath"]).read_text(encoding="utf-8")
            self.assertIn('"name": "read_memory"', memory_trace)
            self.assertTrue(artifact_path.exists())
            saved = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["passed"], 12)
            fixture_readme = benchmark_path.parent / "fixtures" / "docs_readme_status" / "README.md"
            self.assertIn("Status: draft", fixture_readme.read_text(encoding="utf-8"))

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

    def test_real_benchmark_file_uses_real_friendly_contracts(self):
        benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "real_coding_tasks.json"

        benchmark = load_benchmark(benchmark_path)

        self.assertEqual(len(benchmark.tasks), 12)
        by_id = {task.id: task for task in benchmark.tasks}
        self.assertEqual(by_id["code_fix_math_helper"].model_outputs, ())
        self.assertIn("list_files", by_id["code_fix_math_helper"].allowed_tools)
        self.assertIn("list_files", by_id["code_normalize_name"].allowed_tools)
        self.assertEqual(by_id["tool_boundary_schema_retry"].verifier, "python verify_real.py")
        self.assertEqual(by_id["recovery_stale_file_reread"].verifier, "python verify_real.py")
        self.assertEqual(by_id["delegate_hooks_bounded_summary"].expected_artifact, (".minibot/delegates",))

    def test_real_benchmark_mode_uses_injected_model_and_separate_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "coding_tasks.json"
            artifact_path = root / "harness-real-v1.json"

            artifact = run_fixed_benchmark(
                benchmark_path=benchmark_path,
                artifact_path=artifact_path,
                workspace_root=root / "real-workspaces",
                real=True,
                approval_policy="auto",
                max_tasks=1,
                temperature=0.2,
                max_new_tokens=128,
                model_client_factory=lambda task: FakeModelClient(list(task.model_outputs), model="stub-real"),
            )

            self.assertEqual(artifact["mode"], MODE_REAL)
            self.assertEqual(artifact["benchmark"]["task_count"], 1)
            self.assertEqual(artifact["benchmark"]["source_task_count"], 12)
            self.assertEqual(artifact["summary"]["passed"], 1)
            self.assertEqual(artifact["reproducibility"]["model_version"], "injected-test-double")
            self.assertEqual(artifact["reproducibility"]["approval_policy"], "auto")
            self.assertEqual(artifact["reproducibility"]["decoding"]["temperature"], 0.2)
            self.assertEqual(artifact["reproducibility"]["decoding"]["max_new_tokens"], 128)
            self.assertEqual(artifact["rows"][0]["mode"], MODE_REAL)
            self.assertEqual(artifact["rows"][0]["approval_policy"], "auto")
            self.assertEqual(artifact["rows"][0]["model_metadata"]["model"], "stub-real")
            self.assertTrue(artifact_path.exists())

    def test_benchmark_dry_run_records_custom_approval_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "coding_tasks.json"

            artifact = run_fixed_benchmark(
                benchmark_path=benchmark_path,
                artifact_path=root / "planned.json",
                workspace_root=root / "planned-workspaces",
                real=True,
                dry_run=True,
                max_tasks=2,
                approval_policy="deny_risky",
            )

            self.assertTrue(artifact["summary"]["dry_run"])
            self.assertEqual(artifact["summary"]["planned_tasks"], 2)
            self.assertEqual(artifact["reproducibility"]["approval_policy"], "deny_risky")
            self.assertEqual(artifact["rows"], [])

    def test_real_benchmark_provider_error_gets_failure_category(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "coding_tasks.json"

            artifact = run_fixed_benchmark(
                benchmark_path=benchmark_path,
                artifact_path=root / "harness-real-v1.json",
                workspace_root=root / "real-workspaces",
                real=True,
                max_tasks=1,
                model_client_factory=lambda task: FailingProviderModel(),
            )

            self.assertEqual(artifact["summary"]["failed"], 1)
            self.assertEqual(artifact["summary"]["failure_category_counts"], {FAILURE_PROVIDER_ERROR: 1})
            self.assertEqual(artifact["rows"][0]["failure_category"], FAILURE_PROVIDER_ERROR)
            self.assertEqual(artifact["rows"][0]["stop_reason"], "model_error")
            self.assertEqual(artifact["rows"][0]["model_metadata"]["error_category"], "provider_network_error")


if __name__ == "__main__":
    unittest.main()
