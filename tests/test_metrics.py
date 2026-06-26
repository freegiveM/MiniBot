from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.metrics import (
    aggregate_benchmark_artifact,
    compare_benchmark_artifacts,
    write_benchmark_core_report,
    write_benchmark_methodology_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _memory_hook_payload() -> dict:
    return {
        "hooks": {
            "emissions": [
                {
                    "event": "Stop",
                    "outputs": [
                        {
                            "memory_extraction": {
                                "candidate_count": 1,
                                "pending_results": [{"appended": True, "rejected": False}],
                                "metadata": {
                                    "candidate_count": 1,
                                    "extraction_attempt_count": 1,
                                    "extraction_success_count": 1,
                                    "schema_error_count": 0,
                                    "mini_llm_used": True,
                                },
                            }
                        }
                    ],
                }
            ]
        }
    }


def _sample_harness_artifact() -> dict:
    return {
        "schema_version": 1,
        "captured_at": "2026-06-25T00:00:00+00:00",
        "mode": "mock",
        "benchmark": {"path": "benchmarks/coding_tasks.json", "task_count": 2},
        "reproducibility": {"mode": "mock", "model_name": "FakeModelClient"},
        "summary": {
            "total_tasks": 2,
            "passed": 1,
            "failed": 1,
            "pass_rate": 0.5,
            "within_budget_rate": 0.5,
            "verifier_pass_rate": 0.5,
            "failure_category_counts": {"budget_exceeded": 1},
        },
        "rows": [
            {
                "id": "patch_ok",
                "category": "text-edit",
                "passed": True,
                "within_budget": True,
                "verifier_passed": True,
                "tool_steps": 2,
                "attempts": 3,
                "step_budget": 4,
                "expected_artifact_exists": True,
                "failure_category": "",
                "stop_reason": "final_answer_returned",
            },
            {
                "id": "patch_budget_fail",
                "category": "text-edit",
                "passed": False,
                "within_budget": False,
                "verifier_passed": False,
                "tool_steps": 6,
                "attempts": 5,
                "step_budget": 4,
                "expected_artifact_exists": True,
                "failure_category": "budget_exceeded",
                "stop_reason": "step_limit_reached",
                "verifier_exit_code": 1,
                "verifier_stdout": "",
                "verifier_stderr": "marker missing",
                "permission_events": [
                    {"permission_reason": "path_escape", "permission_action": "deny"},
                    {"permission_reason": "risky_shell_command", "permission_action": "deny"},
                ],
                "report": _memory_hook_payload(),
            },
        ],
    }


def _sample_real_harness_artifact() -> dict:
    return {
        "schema_version": 1,
        "captured_at": "2026-06-26T00:00:00+00:00",
        "mode": "real",
        "benchmark": {"path": "benchmarks/coding_tasks.json", "task_count": 1},
        "reproducibility": {
            "mode": "real",
            "provider": "http",
            "model": "deepseek-v4-pro",
            "model_name": "deepseek-v4-pro",
            "api_format": "anthropic",
        },
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
                "id": "docs_update_readme_status",
                "category": "documentation",
                "passed": True,
                "within_budget": True,
                "verifier_passed": True,
                "tool_steps": 2,
                "attempts": 3,
                "latency_ms": 1100,
                "estimated_cost_usd": 0.025,
                "failure_category": "",
                "stop_reason": "final_answer_returned",
                "model_metadata": {
                    "provider": "http",
                    "model": "deepseek-v4-pro",
                    "api_format": "anthropic",
                    "latency_ms": 300,
                    "status_code": 200,
                },
                "report": _memory_hook_payload(),
            }
        ],
    }


class MetricsTests(unittest.TestCase):
    def test_aggregate_benchmark_artifact_computes_core_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact_path = Path(temp) / "harness-regression-v2.json"
            _write_json(artifact_path, _sample_harness_artifact())

            metrics = aggregate_benchmark_artifact(artifact_path)

            self.assertEqual(metrics["mode"], "mock")
            self.assertEqual(metrics["task_count"], 2)
            self.assertEqual(metrics["passed"], 1)
            self.assertEqual(metrics["failed"], 1)
            self.assertEqual(metrics["pass_rate"], 0.5)
            self.assertEqual(metrics["within_budget_rate"], 0.5)
            self.assertEqual(metrics["verifier_pass_rate"], 0.5)
            self.assertEqual(metrics["avg_tool_steps"], 4.0)
            self.assertEqual(metrics["median_tool_steps"], 4.0)
            self.assertEqual(metrics["avg_attempts"], 4.0)
            self.assertEqual(metrics["category_pass_rates"], {"text-edit": 0.5})
            self.assertEqual(metrics["category_metrics"]["text-edit"]["median_tool_steps"], 4.0)
            self.assertEqual(metrics["failure_category_counts"], {"budget_exceeded": 1})
            self.assertEqual(metrics["failed_examples"][0]["id"], "patch_budget_fail")
            self.assertEqual(metrics["failure_examples_by_category"]["text-edit"][0]["id"], "patch_budget_fail")
            self.assertEqual(metrics["path_escape_rejection_count"], 1)
            self.assertEqual(metrics["risky_tool_block_rate"], 1.0)
            self.assertEqual(metrics["approval_denied_count"], 2)
            self.assertEqual(metrics["provider_error_rate"], 0.0)
            self.assertEqual(metrics["memory_extraction"]["candidate_count"], 1)
            self.assertEqual(metrics["memory_extraction"]["accepted_count"], 1)
            self.assertEqual(metrics["memory_extraction_accept_rate"], 1.0)
            self.assertEqual(metrics["mini_llm_schema_error_rate"], 0.0)

    def test_write_benchmark_core_report_separates_safe_and_interview_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact_path = root / "harness-regression-v2.json"
            report_path = root / "minibot-benchmark-core-report.md"
            _write_json(artifact_path, _sample_harness_artifact())

            report = write_benchmark_core_report(
                report_path=report_path,
                harness_artifact_path=artifact_path,
                context_artifact_path=root / "context-ablation-v2.json",
                memory_artifact_path=root / "memory-ablation-v2.json",
                recovery_artifact_path=root / "recovery-ablation-v2.json",
                retrieval_artifact_path=root / "retrieval-ablation-v2.json",
            )

            self.assertEqual(report_path.read_text(encoding="utf-8"), report)
            self.assertIn("可以安全写进简历的指标", report)
            self.assertIn("适合面试展开的指标", report)
            self.assertIn("不应过度声称的探索指标", report)
            self.assertIn("Category Breakdown", report)
            self.assertIn("median_tool_steps", report)
            self.assertIn("category_pass_rates", report)
            self.assertIn("Failure examples by category", report)
            self.assertIn("failure_category_counts", report)
            self.assertIn("patch_budget_fail", report)
            self.assertIn("budget_exceeded", report)
            self.assertIn("missing", report)
            self.assertIn("not_available", report)
            self.assertIn("path_escape_rejection_count", report)

    def test_write_report_includes_present_context_ablation_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact_path = root / "harness-regression-v2.json"
            context_path = root / "context-ablation-v2.json"
            report_path = root / "report.md"
            _write_json(artifact_path, _sample_harness_artifact())
            _write_json(
                context_path,
                {
                    "variants": {
                        "full": {
                            "prompt_chars": 1000,
                            "raw_prompt_chars": 1200,
                            "compression_ratio": 0.8333,
                            "current_request_preserved_rate": 1.0,
                            "compact_snapshot_created_count": 1,
                        },
                        "no_context_reduction": {
                            "prompt_chars": 1200,
                            "raw_prompt_chars": 1200,
                            "compression_ratio": 1.0,
                            "current_request_preserved_rate": 1.0,
                            "compact_snapshot_created_count": 0,
                        },
                    }
                },
            )

            report = write_benchmark_core_report(
                report_path=report_path,
                harness_artifact_path=artifact_path,
                context_artifact_path=context_path,
                memory_artifact_path=root / "memory-ablation-v2.json",
                recovery_artifact_path=root / "recovery-ablation-v2.json",
                retrieval_artifact_path=root / "retrieval-ablation-v2.json",
            )

            self.assertIn("Context ablation", report)
            self.assertIn("full", report)
            self.assertIn("no_context_reduction", report)
            self.assertIn("prompt_chars", report)
            self.assertIn("0.8333", report)

    def test_compare_benchmark_artifacts_and_methodology_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock_path = root / "harness-regression-v2.json"
            real_path = root / "harness-real-v1.json"
            report_path = root / "minibot-benchmark-methodology-report.md"
            _write_json(mock_path, _sample_harness_artifact())
            _write_json(real_path, _sample_real_harness_artifact())

            comparison = compare_benchmark_artifacts(mock_path, real_path)
            report = write_benchmark_methodology_report(
                report_path=report_path,
                mock_harness_artifact_path=mock_path,
                real_harness_artifact_path=real_path,
                context_artifact_path=root / "context-ablation-v2.json",
                memory_artifact_path=root / "memory-ablation-v2.json",
                recovery_artifact_path=root / "recovery-ablation-v2.json",
                retrieval_artifact_path=root / "retrieval-ablation-v2.json",
            )

            self.assertEqual(comparison["mock"]["status"], "present")
            self.assertEqual(comparison["real"]["status"], "present")
            self.assertEqual(comparison["real"]["metrics"]["provider"]["model"], "deepseek-v4-pro")
            self.assertEqual(comparison["real"]["metrics"]["estimated_cost_usd_total"], 0.025)
            self.assertIn("MiniBot Benchmark Methodology Report", report)
            self.assertIn("Reference Benchmark Ideas", report)
            self.assertIn("Mock Vs Real Summary", report)
            self.assertIn("Provider, Latency, And Cost", report)
            self.assertIn("Memory And miniLLM Subchain", report)
            self.assertIn("Resume-Safe Metrics", report)
            self.assertIn("Do Not Overclaim", report)
            self.assertEqual(report_path.read_text(encoding="utf-8"), report)

    def test_methodology_report_tolerates_missing_real_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mock_path = root / "harness-regression-v2.json"
            _write_json(mock_path, _sample_harness_artifact())

            report = write_benchmark_methodology_report(
                report_path=root / "methodology.md",
                mock_harness_artifact_path=mock_path,
                real_harness_artifact_path=root / "missing-real.json",
            )

            self.assertIn("`real` | `missing`", report)
            self.assertIn("not_available", report)


if __name__ == "__main__":
    unittest.main()
