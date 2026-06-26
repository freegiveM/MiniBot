from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HARNESS_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_REAL_HARNESS_ARTIFACT_PATH = Path("artifacts/harness-real-v1.json")
DEFAULT_CONTEXT_ARTIFACT_PATH = Path("artifacts/context-ablation-v2.json")
DEFAULT_MEMORY_ARTIFACT_PATH = Path("artifacts/memory-ablation-v2.json")
DEFAULT_RECOVERY_ARTIFACT_PATH = Path("artifacts/recovery-ablation-v2.json")
DEFAULT_RETRIEVAL_ARTIFACT_PATH = Path("artifacts/retrieval-ablation-v2.json")
DEFAULT_REPORT_PATH = Path("artifacts/minibot-benchmark-core-report.md")
DEFAULT_METHODOLOGY_REPORT_PATH = Path("artifacts/minibot-benchmark-methodology-report.md")

NOT_AVAILABLE = "not_available"
STATUS_MISSING = "missing"
STATUS_PRESENT = "present"
STATUS_INVALID = "invalid"

RISKY_PERMISSION_REASONS = frozenset({"risky_tool", "risky_shell_command"})


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    variants: tuple[str, ...]
    metrics: tuple[str, ...]


CONTEXT_ABLATION = AblationSpec(
    name="context",
    label="Context ablation",
    variants=("full", "no_context_reduction"),
    metrics=(
        "prompt_chars",
        "raw_prompt_chars",
        "compression_ratio",
        "current_request_preserved_rate",
        "compact_snapshot_created_count",
    ),
)
MEMORY_ABLATION = AblationSpec(
    name="memory",
    label="Memory ablation",
    variants=("memory_on", "memory_off", "memory_irrelevant"),
    metrics=("memory_hit_rate", "correct_rate", "repeated_reads", "source_reread_count", "false_positive_count"),
)
RECOVERY_ABLATION = AblationSpec(
    name="recovery",
    label="Recovery ablation",
    variants=("resume_enabled", "resume_disabled"),
    metrics=(
        "resume_success_rate",
        "stale_reanchor_rate",
        "workspace_drift_detection_rate",
        "resume_false_accept_rate",
    ),
)
RETRIEVAL_ABLATION = AblationSpec(
    name="retrieval",
    label="Retrieval ablation",
    variants=("llm_catalog_selection", "llm_plus_keyword_fallback", "keyword_fallback_only", "retrieval_off"),
    metrics=(
        "retrieval_hit_rate",
        "selected_memory_precision",
        "selection_failure_rate",
        "fallback_usage_rate",
        "memory_false_positive_count",
    ),
)

ABLATION_SPECS = (CONTEXT_ABLATION, MEMORY_ABLATION, RECOVERY_ABLATION, RETRIEVAL_ABLATION)


def aggregate_benchmark_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact = _load_json_object(artifact_path)
    rows = _artifact_rows(artifact)
    summary = artifact.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    row_count = len(rows)
    task_count = row_count or _int_value(
        summary.get("total_tasks", _nested_get(artifact, ("benchmark", "task_count"), 0)),
        default=0,
    )

    if rows:
        passed = _count_true(rows, "passed")
        within_budget = _count_true(rows, "within_budget")
        verifier_passed = _count_true(rows, "verifier_passed")
        pass_rate = _rate(passed, row_count)
        within_budget_rate = _rate(within_budget, row_count)
        verifier_pass_rate = _rate(verifier_passed, row_count)
        failure_category_counts = _failure_category_counts(rows)
    else:
        passed = _int_value(summary.get("passed"), 0)
        within_budget = int(round(_float_value(summary.get("within_budget_rate"), 0.0) * task_count))
        verifier_passed = int(round(_float_value(summary.get("verifier_pass_rate"), 0.0) * task_count))
        pass_rate = _float_value(summary.get("pass_rate"), 0.0)
        within_budget_rate = _float_value(summary.get("within_budget_rate"), 0.0)
        verifier_pass_rate = _float_value(summary.get("verifier_pass_rate"), 0.0)
        failure_category_counts = _dict_ints(summary.get("failure_category_counts", {}))

    avg_tool_steps = _average(_number(row.get("tool_steps"), 0.0) for row in rows)
    avg_attempts = _average(_number(row.get("attempts"), 0.0) for row in rows)
    median_tool_steps = _median(_number(row.get("tool_steps"), 0.0) for row in rows)
    category_metrics = _category_metrics(rows, summary)
    security = aggregate_tool_boundary_security(artifact)
    failure_category_counts = _failure_category_counts(rows) if rows else _dict_ints(summary.get("failure_category_counts", {}))
    provider = _provider_summary(rows, artifact, task_count, failure_category_counts)
    memory_extraction = _memory_extraction_summary(rows)

    result: dict[str, Any] = {
        "artifact_path": str(artifact_path),
        "mode": _text(artifact.get("mode") or _nested_get(artifact, ("reproducibility", "mode"), "unknown")),
        "captured_at": _text(artifact.get("captured_at")),
        "benchmark": artifact.get("benchmark", {}) if isinstance(artifact.get("benchmark"), dict) else {},
        "runtime": artifact.get("runtime", {}) if isinstance(artifact.get("runtime"), dict) else {},
        "reproducibility": artifact.get("reproducibility", {})
        if isinstance(artifact.get("reproducibility"), dict)
        else {},
        "task_count": task_count,
        "passed": passed,
        "failed": max(task_count - passed, 0),
        "pass_rate": pass_rate,
        "within_budget_rate": within_budget_rate,
        "verifier_pass_rate": verifier_pass_rate,
        "avg_tool_steps": avg_tool_steps,
        "median_tool_steps": median_tool_steps,
        "avg_attempts": avg_attempts,
        "provider": provider,
        "provider_error_rate": provider["provider_error_rate"],
        "latency_ms_summary": provider["latency_ms_summary"],
        "estimated_cost_usd_total": provider["estimated_cost_usd_total"],
        "estimated_cost_usd_avg": provider["estimated_cost_usd_avg"],
        "memory_extraction": memory_extraction,
        "mini_llm_schema_error_rate": memory_extraction["mini_llm_schema_error_rate"],
        "memory_extraction_accept_rate": memory_extraction["memory_extraction_accept_rate"],
        "category_metrics": category_metrics,
        "category_pass_rates": {category: item["pass_rate"] for category, item in category_metrics.items()},
        "failure_category_counts": failure_category_counts,
        "failed_examples": _failed_examples(rows),
        "failure_examples_by_category": _failure_examples_by_category(rows),
        "security": security,
    }
    result.update(
        {
            "path_escape_rejection_count": security["path_escape_rejection_count"],
            "risky_tool_block_rate": security["risky_tool_block_rate"],
            "approval_denied_count": security["approval_denied_count"],
        }
    )
    return result


def aggregate_tool_boundary_security(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = _artifact_rows(artifact)
    path_escape_rejections = 0
    risky_attempts = 0
    risky_blocks = 0
    approval_denied = 0

    for row in rows:
        for evidence in _iter_permission_evidence(row):
            reason = _permission_reason(evidence)
            action = _permission_action(evidence)
            blocked = _is_permission_block(evidence, action)
            if reason == "path_escape" and blocked:
                path_escape_rejections += 1
            if reason in RISKY_PERMISSION_REASONS:
                risky_attempts += 1
                if blocked:
                    risky_blocks += 1
            if blocked:
                approval_denied += 1

    return {
        "path_escape_rejection_count": path_escape_rejections,
        "risky_tool_attempt_count": risky_attempts,
        "risky_tool_block_count": risky_blocks,
        "risky_tool_block_rate": _rate(risky_blocks, risky_attempts),
        "approval_denied_count": approval_denied,
        "source": "permission evidence embedded in evaluator rows and run reports",
    }


def _provider_summary(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    task_count: int,
    failure_category_counts: dict[str, int],
) -> dict[str, Any]:
    reproducibility = artifact.get("reproducibility", {}) if isinstance(artifact.get("reproducibility"), dict) else {}
    provider_name = _text(reproducibility.get("provider"), NOT_AVAILABLE)
    model_name = _text(reproducibility.get("model") or reproducibility.get("model_name"), NOT_AVAILABLE)
    api_format = _text(reproducibility.get("api_format"), NOT_AVAILABLE)
    row_latencies = [_number(row.get("latency_ms"), None) for row in rows]
    provider_latencies = [
        _number(_nested_get(row, ("model_metadata", "latency_ms"), None), None)
        for row in rows
        if isinstance(row.get("model_metadata"), dict)
    ]
    costs = [_number(row.get("estimated_cost_usd"), None) for row in rows]
    costs = [value for value in costs if value is not None]
    for row in rows:
        metadata = row.get("model_metadata", {}) if isinstance(row.get("model_metadata"), dict) else {}
        provider_name = _text(metadata.get("provider"), provider_name)
        model_name = _text(metadata.get("model"), model_name)
        api_format = _text(metadata.get("api_format"), api_format)
    return {
        "provider": provider_name,
        "model": model_name,
        "api_format": api_format,
        "provider_error_count": int(failure_category_counts.get("provider_error", 0)),
        "provider_error_rate": _rate(int(failure_category_counts.get("provider_error", 0)), task_count),
        "latency_ms_summary": _numeric_summary(row_latencies),
        "provider_latency_ms_summary": _numeric_summary(provider_latencies),
        "estimated_cost_usd_total": sum(float(value) for value in costs) if costs else 0.0,
        "estimated_cost_usd_avg": _average(costs),
    }


def _memory_extraction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    extraction_attempts = 0
    extraction_successes = 0
    deterministic_fallbacks = 0
    schema_errors = 0
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    mini_llm_used_count = 0

    for payload in _iter_memory_extraction_payloads(rows):
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        extraction_attempts += _int_value(metadata.get("extraction_attempt_count"), 0)
        extraction_successes += _int_value(metadata.get("extraction_success_count"), 0)
        deterministic_fallbacks += _int_value(metadata.get("deterministic_fallback_count"), 0)
        schema_errors += _int_value(metadata.get("schema_error_count"), 0)
        candidate_count += _int_value(payload.get("candidate_count", metadata.get("candidate_count")), 0)
        if _is_true(metadata.get("mini_llm_used")):
            mini_llm_used_count += 1
        pending_results = payload.get("pending_results", [])
        if isinstance(pending_results, list):
            for result in pending_results:
                if not isinstance(result, dict):
                    continue
                if _is_true(result.get("rejected")):
                    rejected_count += 1
                if _is_true(result.get("appended")) and not _is_true(result.get("rejected")):
                    accepted_count += 1

    return {
        "extraction_attempt_count": extraction_attempts,
        "extraction_success_count": extraction_successes,
        "deterministic_fallback_count": deterministic_fallbacks,
        "schema_error_count": schema_errors,
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "mini_llm_used_count": mini_llm_used_count,
        "mini_llm_schema_error_rate": _rate(schema_errors, extraction_attempts),
        "memory_extraction_accept_rate": _rate(accepted_count, candidate_count),
    }


def _iter_memory_extraction_payloads(rows: list[dict[str, Any]]):
    for row in rows:
        report = row.get("report", {}) if isinstance(row.get("report"), dict) else {}
        hooks = report.get("hooks", {}) if isinstance(report.get("hooks"), dict) else {}
        emissions = hooks.get("emissions", [])
        if not isinstance(emissions, list):
            continue
        for emission in emissions:
            if not isinstance(emission, dict):
                continue
            outputs = emission.get("outputs", [])
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                payload = output.get("memory_extraction")
                if isinstance(payload, dict):
                    yield payload


def summarize_benchmark_artifact(path: str | Path, *, label: str = "") -> dict[str, Any]:
    artifact_path = Path(path)
    base = {
        "label": label or artifact_path.stem,
        "path": str(artifact_path),
        "status": STATUS_MISSING,
        "metrics": {},
    }
    if not artifact_path.exists():
        return base
    try:
        return {
            **base,
            "status": STATUS_PRESENT,
            "metrics": aggregate_benchmark_artifact(artifact_path),
        }
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {**base, "status": STATUS_INVALID, "error": str(exc)}


def compare_benchmark_artifacts(
    mock_harness_artifact_path: str | Path = DEFAULT_HARNESS_ARTIFACT_PATH,
    real_harness_artifact_path: str | Path = DEFAULT_REAL_HARNESS_ARTIFACT_PATH,
) -> dict[str, Any]:
    mock = summarize_benchmark_artifact(mock_harness_artifact_path, label="mock")
    real = summarize_benchmark_artifact(real_harness_artifact_path, label="real")
    comparison: dict[str, Any] = {}
    if mock["status"] == STATUS_PRESENT and real["status"] == STATUS_PRESENT:
        mock_metrics = mock["metrics"]
        real_metrics = real["metrics"]
        comparison = {
            "pass_rate_delta_real_minus_mock": _float_value(real_metrics.get("pass_rate"))
            - _float_value(mock_metrics.get("pass_rate")),
            "verifier_pass_rate_delta_real_minus_mock": _float_value(real_metrics.get("verifier_pass_rate"))
            - _float_value(mock_metrics.get("verifier_pass_rate")),
            "median_tool_steps_delta_real_minus_mock": _float_value(real_metrics.get("median_tool_steps"))
            - _float_value(mock_metrics.get("median_tool_steps")),
            "real_provider_error_rate": _float_value(real_metrics.get("provider_error_rate")),
            "real_task_count": _int_value(real_metrics.get("task_count")),
            "mock_task_count": _int_value(mock_metrics.get("task_count")),
        }
    return {"mock": mock, "real": real, "comparison": comparison}


def summarize_ablation_artifact(path: str | Path, spec: AblationSpec) -> dict[str, Any]:
    artifact_path = Path(path)
    base = {
        "name": spec.name,
        "label": spec.label,
        "path": str(artifact_path),
        "metrics": list(spec.metrics),
        "variants": {variant: {metric: NOT_AVAILABLE for metric in spec.metrics} for variant in spec.variants},
        "missing_variants": list(spec.variants),
        "missing_metrics": {variant: list(spec.metrics) for variant in spec.variants},
        "status": STATUS_MISSING,
    }
    if not artifact_path.exists():
        return base

    try:
        artifact = _load_json_object(artifact_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {**base, "status": STATUS_INVALID, "error": str(exc)}

    payloads = _extract_variant_payloads(artifact, spec)
    variants: dict[str, dict[str, Any]] = {}
    missing_variants: list[str] = []
    missing_metrics: dict[str, list[str]] = {}
    for variant in spec.variants:
        payload = payloads.get(variant, {})
        if not payload:
            missing_variants.append(variant)
        values: dict[str, Any] = {}
        missing_for_variant: list[str] = []
        for metric in spec.metrics:
            value = payload.get(metric, NOT_AVAILABLE) if isinstance(payload, dict) else NOT_AVAILABLE
            if value is None:
                value = NOT_AVAILABLE
            if value == NOT_AVAILABLE:
                missing_for_variant.append(metric)
            values[metric] = _json_safe_scalar(value)
        variants[variant] = values
        if missing_for_variant:
            missing_metrics[variant] = missing_for_variant

    return {
        **base,
        "status": STATUS_PRESENT,
        "captured_at": _text(artifact.get("captured_at")),
        "variants": variants,
        "missing_variants": missing_variants,
        "missing_metrics": missing_metrics,
    }


def write_benchmark_core_report(
    report_path: str | Path = DEFAULT_REPORT_PATH,
    harness_artifact_path: str | Path = DEFAULT_HARNESS_ARTIFACT_PATH,
    context_artifact_path: str | Path = DEFAULT_CONTEXT_ARTIFACT_PATH,
    memory_artifact_path: str | Path = DEFAULT_MEMORY_ARTIFACT_PATH,
    recovery_artifact_path: str | Path = DEFAULT_RECOVERY_ARTIFACT_PATH,
    retrieval_artifact_path: str | Path = DEFAULT_RETRIEVAL_ARTIFACT_PATH,
) -> str:
    harness = aggregate_benchmark_artifact(harness_artifact_path)
    ablations = [
        summarize_ablation_artifact(context_artifact_path, CONTEXT_ABLATION),
        summarize_ablation_artifact(memory_artifact_path, MEMORY_ABLATION),
        summarize_ablation_artifact(recovery_artifact_path, RECOVERY_ABLATION),
        summarize_ablation_artifact(retrieval_artifact_path, RETRIEVAL_ABLATION),
    ]

    report = _render_report(harness, ablations)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(report, encoding="utf-8")
    temp_path.replace(path)
    return report


def write_benchmark_methodology_report(
    report_path: str | Path = DEFAULT_METHODOLOGY_REPORT_PATH,
    mock_harness_artifact_path: str | Path = DEFAULT_HARNESS_ARTIFACT_PATH,
    real_harness_artifact_path: str | Path = DEFAULT_REAL_HARNESS_ARTIFACT_PATH,
    context_artifact_path: str | Path = DEFAULT_CONTEXT_ARTIFACT_PATH,
    memory_artifact_path: str | Path = DEFAULT_MEMORY_ARTIFACT_PATH,
    recovery_artifact_path: str | Path = DEFAULT_RECOVERY_ARTIFACT_PATH,
    retrieval_artifact_path: str | Path = DEFAULT_RETRIEVAL_ARTIFACT_PATH,
) -> str:
    benchmark_comparison = compare_benchmark_artifacts(mock_harness_artifact_path, real_harness_artifact_path)
    ablations = [
        summarize_ablation_artifact(context_artifact_path, CONTEXT_ABLATION),
        summarize_ablation_artifact(memory_artifact_path, MEMORY_ABLATION),
        summarize_ablation_artifact(recovery_artifact_path, RECOVERY_ABLATION),
        summarize_ablation_artifact(retrieval_artifact_path, RETRIEVAL_ABLATION),
    ]

    report = _render_methodology_report(benchmark_comparison, ablations)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(report, encoding="utf-8")
    temp_path.replace(path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MiniBot benchmark artifacts.")
    parser.add_argument("--harness-artifact-path", default=str(DEFAULT_HARNESS_ARTIFACT_PATH))
    parser.add_argument("--real-harness-artifact-path", default=str(DEFAULT_REAL_HARNESS_ARTIFACT_PATH))
    parser.add_argument("--context-artifact-path", default=str(DEFAULT_CONTEXT_ARTIFACT_PATH))
    parser.add_argument("--memory-artifact-path", default=str(DEFAULT_MEMORY_ARTIFACT_PATH))
    parser.add_argument("--recovery-artifact-path", default=str(DEFAULT_RECOVERY_ARTIFACT_PATH))
    parser.add_argument("--retrieval-artifact-path", default=str(DEFAULT_RETRIEVAL_ARTIFACT_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--write-core-report", action="store_true")
    parser.add_argument("--write-methodology-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_methodology_report:
        report_path = (
            DEFAULT_METHODOLOGY_REPORT_PATH
            if args.report_path == str(DEFAULT_REPORT_PATH)
            else Path(args.report_path)
        )
        write_benchmark_methodology_report(
            report_path=report_path,
            mock_harness_artifact_path=args.harness_artifact_path,
            real_harness_artifact_path=args.real_harness_artifact_path,
            context_artifact_path=args.context_artifact_path,
            memory_artifact_path=args.memory_artifact_path,
            recovery_artifact_path=args.recovery_artifact_path,
            retrieval_artifact_path=args.retrieval_artifact_path,
        )
        print(str(Path(report_path)))
        return 0
    if args.write_core_report:
        write_benchmark_core_report(
            report_path=args.report_path,
            harness_artifact_path=args.harness_artifact_path,
            context_artifact_path=args.context_artifact_path,
            memory_artifact_path=args.memory_artifact_path,
            recovery_artifact_path=args.recovery_artifact_path,
            retrieval_artifact_path=args.retrieval_artifact_path,
        )
        print(str(Path(args.report_path)))
        return 0

    metrics = aggregate_benchmark_artifact(args.harness_artifact_path)
    print(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _render_methodology_report(benchmark_comparison: dict[str, Any], ablations: list[dict[str, Any]]) -> str:
    mock = benchmark_comparison["mock"]
    real = benchmark_comparison["real"]
    comparison = benchmark_comparison.get("comparison", {})
    lines = [
        "# MiniBot Benchmark Methodology Report",
        "",
        "## Purpose",
        "",
        "This report separates harness reliability from real-model behavior. Mock benchmark results are used as regression evidence for fixtures, tools, verifiers, recovery, memory hooks, and metrics. Real benchmark results are opt-in smoke evidence for provider integration and model decision quality.",
        "",
        "## Reference Benchmark Ideas",
        "",
        "| reference | idea borrowed | MiniBot choice |",
        "| --- | --- | --- |",
        "| HumanEval / EvalPlus | executable checks over model self-judgment | verifier scripts, unit tests, and file assertions are the primary pass/fail signal |",
        "| SWE-bench | patch a repository and judge final state with tests | small fresh fixture copies stand in for heavyweight repositories |",
        "| AgentBench | record multi-step agent interaction with an environment | MiniBot records local file, shell, permission, memory, trace, and report evidence |",
        "| GAIA | short tasks can still require tool use and evidence | tasks avoid pure chat and require bounded tool evidence |",
        "| WebArena / OSWorld | evaluate action traces and final environment state | trace/report/task_state plus verifier output form the evidence chain |",
        "",
        "## Artifact Inputs",
        "",
        "| artifact | status | path | mode | tasks | captured_at |",
        "| --- | --- | --- | --- | ---: | --- |",
        _artifact_source_row(mock),
        _artifact_source_row(real),
        "",
        "## Mock Vs Real Summary",
        "",
        "| metric | mock | real | interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    summary_rows = (
        ("task_count", "task_count", "how many selected tasks ran"),
        ("pass_rate", "pass_rate", "end-to-end row.passed rate"),
        ("verifier_pass_rate", "verifier_pass_rate", "executable verifier success rate"),
        ("within_budget_rate", "within_budget_rate", "tool step budget adherence"),
        ("median_tool_steps", "median_tool_steps", "typical tool steps per task"),
        ("provider_error_rate", "provider_error_rate", "real provider or parser failures"),
        ("mini_llm_schema_error_rate", "mini_llm_schema_error_rate", "memory miniLLM schema/fallback error rate"),
        ("memory_extraction_accept_rate", "memory_extraction_accept_rate", "accepted pending memory candidates per candidate"),
        ("estimated_cost_usd_total", "estimated_cost_usd_total", "reported provider cost, if available"),
    )
    for label, key, interpretation in summary_rows:
        lines.append(
            f"| `{label}` | {_artifact_metric(mock, key)} | {_artifact_metric(real, key)} | {_md_escape(interpretation)} |"
        )
    if comparison:
        lines.extend(
            [
                "",
                "Comparison deltas are real minus mock:",
                "",
                f"- pass_rate delta: `{_format_value(comparison.get('pass_rate_delta_real_minus_mock'))}`",
                f"- verifier_pass_rate delta: `{_format_value(comparison.get('verifier_pass_rate_delta_real_minus_mock'))}`",
                f"- median_tool_steps delta: `{_format_value(comparison.get('median_tool_steps_delta_real_minus_mock'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Category Pass Rates",
            "",
            "| category | mock pass_rate | real pass_rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for category in _combined_categories(mock, real):
        lines.append(
            f"| `{_md_escape(category)}` | {_artifact_category_rate(mock, category)} | {_artifact_category_rate(real, category)} |"
        )

    lines.extend(
        [
            "",
            "## Provider, Latency, And Cost",
            "",
            "| artifact | provider | model | api_format | provider_error_rate | avg_latency_ms | total_cost_usd |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
            _provider_row(mock),
            _provider_row(real),
            "",
            "## Memory And miniLLM Subchain",
            "",
            "| artifact | mini_llm_used | extraction_attempts | schema_errors | candidates | accepted | schema_error_rate | accept_rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _memory_extraction_row(mock),
            _memory_extraction_row(real),
            "",
            "## Failure Examples",
            "",
        ]
    )
    lines.extend(_artifact_failure_lines("mock", mock))
    lines.extend(_artifact_failure_lines("real", real))

    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "### Resume-Safe Metrics",
            "",
            "- Number of deterministic fixture tasks and pass rate, when produced by committed mock artifacts.",
            "- Verifier pass rate, within-budget rate, median tool steps, category pass rates, and failure taxonomy counts.",
            "- Artifact-backed provider error rate, latency, and cost fields, when the artifact records them.",
            "",
            "### Good Interview Discussion Metrics",
            "",
            "- Why mock results prove harness and verifier reliability but not real LLM generalization.",
            "- How real smoke runs validate provider wiring and prompt/tool protocol under cost limits.",
            "- How memory extraction is evaluated through schema reliability, pending candidates, traceability, and fallback behavior.",
            "",
            "### Do Not Overclaim",
            "",
            "- Do not compare MiniBot scores to SWE-bench, GAIA, WebArena, or OSWorld as if the scales are equivalent.",
            "- Do not use a single real run as a general model-quality claim.",
            "- Do not use LLM self-judgment as the primary pass/fail signal.",
            "- Do not treat missing artifacts as zero performance; this report marks them as missing/not_available.",
            "",
            "## Reproducibility Rules",
            "",
            "- Mock runs use scripted deterministic model outputs.",
            "- Real runs are explicit opt-in and should use low temperature, max task limits, token limits, timeouts, and cost guards.",
            "- Every task uses a fresh fixture copy; verifiers run inside that copy.",
            "- Reports should cite artifact paths and preserve failure examples instead of reporting only averages.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_source_row(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    benchmark = metrics.get("benchmark", {}) if isinstance(metrics.get("benchmark"), dict) else {}
    return (
        "| "
        + " | ".join(
            (
                f"`{_md_escape(summary.get('label', 'artifact'))}`",
                f"`{_md_escape(summary.get('status', STATUS_MISSING))}`",
                f"`{_md_escape(summary.get('path', ''))}`",
                f"`{_md_escape(metrics.get('mode', NOT_AVAILABLE))}`",
                _format_value(metrics.get("task_count", NOT_AVAILABLE)),
                _md_escape(metrics.get("captured_at") or benchmark.get("captured_at") or NOT_AVAILABLE),
            )
        )
        + " |"
    )


def _artifact_metric(summary: dict[str, Any], key: str) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    if summary.get("status") != STATUS_PRESENT:
        return f"`{summary.get('status', STATUS_MISSING)}`"
    return _format_value(metrics.get(key, NOT_AVAILABLE))


def _combined_categories(*summaries: dict[str, Any]) -> list[str]:
    categories: set[str] = set()
    for summary in summaries:
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        category_metrics = metrics.get("category_metrics", {})
        if isinstance(category_metrics, dict):
            categories.update(str(category) for category in category_metrics)
    return sorted(categories)


def _artifact_category_rate(summary: dict[str, Any], category: str) -> str:
    if summary.get("status") != STATUS_PRESENT:
        return f"`{summary.get('status', STATUS_MISSING)}`"
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    category_metrics = metrics.get("category_metrics", {}) if isinstance(metrics.get("category_metrics"), dict) else {}
    payload = category_metrics.get(category)
    if not isinstance(payload, dict):
        return f"`{NOT_AVAILABLE}`"
    return _format_value(payload.get("pass_rate", NOT_AVAILABLE))


def _provider_row(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    provider = metrics.get("provider", {}) if isinstance(metrics.get("provider"), dict) else {}
    latency = provider.get("latency_ms_summary", {}) if isinstance(provider.get("latency_ms_summary"), dict) else {}
    return (
        "| "
        + " | ".join(
            (
                f"`{_md_escape(summary.get('label', 'artifact'))}`",
                f"`{_md_escape(provider.get('provider', NOT_AVAILABLE))}`",
                f"`{_md_escape(provider.get('model', NOT_AVAILABLE))}`",
                f"`{_md_escape(provider.get('api_format', NOT_AVAILABLE))}`",
                _artifact_metric(summary, "provider_error_rate"),
                _format_value(latency.get("avg", NOT_AVAILABLE)) if summary.get("status") == STATUS_PRESENT else f"`{summary.get('status', STATUS_MISSING)}`",
                _artifact_metric(summary, "estimated_cost_usd_total"),
            )
        )
        + " |"
    )


def _memory_extraction_row(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    memory = metrics.get("memory_extraction", {}) if isinstance(metrics.get("memory_extraction"), dict) else {}
    if summary.get("status") != STATUS_PRESENT:
        status = f"`{summary.get('status', STATUS_MISSING)}`"
        values = [status] * 7
    else:
        values = [
            _format_value(memory.get("mini_llm_used_count", NOT_AVAILABLE)),
            _format_value(memory.get("extraction_attempt_count", NOT_AVAILABLE)),
            _format_value(memory.get("schema_error_count", NOT_AVAILABLE)),
            _format_value(memory.get("candidate_count", NOT_AVAILABLE)),
            _format_value(memory.get("accepted_count", NOT_AVAILABLE)),
            _format_value(memory.get("mini_llm_schema_error_rate", NOT_AVAILABLE)),
            _format_value(memory.get("memory_extraction_accept_rate", NOT_AVAILABLE)),
        ]
    return f"| `{_md_escape(summary.get('label', 'artifact'))}` | " + " | ".join(values) + " |"


def _artifact_failure_lines(label: str, summary: dict[str, Any]) -> list[str]:
    if summary.get("status") != STATUS_PRESENT:
        return [f"- `{_md_escape(label)}` artifact is `{_md_escape(summary.get('status', STATUS_MISSING))}`."]
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    examples = metrics.get("failed_examples", [])
    if not examples:
        return [f"- `{_md_escape(label)}`: no failing examples."]
    lines = [
        f"### {_md_escape(label)}",
        "",
        "| id | category | failure_category | reason | stop_reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for example in examples[:8]:
        lines.append(
            "| "
            + " | ".join(
                _md_escape(_format_value(example.get(key)))
                for key in ("id", "category", "failure_category", "failure_reason", "stop_reason")
            )
            + " |"
        )
    return lines


def _render_report(harness: dict[str, Any], ablations: list[dict[str, Any]]) -> str:
    lines = [
        "# MiniBot Benchmark Core Report",
        "",
        "## Metric Sources",
        "",
        f"- Harness artifact: `{_md_escape(harness['artifact_path'])}`",
        f"- Captured at: `{_md_escape(harness.get('captured_at') or NOT_AVAILABLE)}`",
        f"- Benchmark: `{_md_escape(_nested_get(harness, ('benchmark', 'path'), NOT_AVAILABLE))}`",
        f"- Model source: `{_md_escape(_nested_get(harness, ('reproducibility', 'model_name'), NOT_AVAILABLE))}`",
        "- Metrics in this report are derived from existing evaluator and ablation JSON artifacts only.",
        "",
        "## 可以安全写进简历的指标",
        "",
        "| metric | value | source |",
        "| --- | ---: | --- |",
    ]
    safe_metrics = (
        ("task_count", harness["task_count"], "evaluator rows"),
        ("pass_rate", harness["pass_rate"], "row.passed"),
        ("within_budget_rate", harness["within_budget_rate"], "row.within_budget"),
        ("verifier_pass_rate", harness["verifier_pass_rate"], "row.verifier_passed"),
        ("avg_tool_steps", harness["avg_tool_steps"], "row.tool_steps"),
        ("median_tool_steps", harness["median_tool_steps"], "row.tool_steps"),
        ("category_pass_rates", harness["category_pass_rates"], "row.category + row.passed"),
        ("avg_attempts", harness["avg_attempts"], "row.attempts"),
        ("failure_category_counts", harness["failure_category_counts"], "row.failure_category"),
        ("path_escape_rejection_count", harness["path_escape_rejection_count"], "permission evidence"),
        ("risky_tool_block_rate", harness["risky_tool_block_rate"], "permission evidence"),
        ("approval_denied_count", harness["approval_denied_count"], "permission evidence"),
    )
    for metric, value, source in safe_metrics:
        lines.append(f"| `{metric}` | {_format_value(value)} | {_md_escape(source)} |")

    security = harness["security"]
    lines.extend(
        [
            "",
            "Tool-boundary/security aggregation:",
            "",
            f"- Risky tool block count: `{security['risky_tool_block_count']}` / `{security['risky_tool_attempt_count']}`.",
            f"- Source limitation: {_md_escape(security['source'])}.",
            "",
            "## Category Breakdown",
            "",
            "| category | tasks | passed | pass_rate | median_tool_steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, item in harness["category_metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md_escape(category)}`",
                    _format_value(item.get("total_tasks")),
                    _format_value(item.get("passed")),
                    _format_value(item.get("pass_rate")),
                    _format_value(item.get("median_tool_steps")),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 适合面试展开的指标",
            "",
        ]
    )
    for summary in ablations:
        lines.extend(_render_ablation(summary))
        lines.append("")

    lines.extend(
        [
            "## 不通过样例与失败原因",
            "",
        ]
    )
    failed_examples = harness.get("failed_examples", [])
    if failed_examples:
        lines.extend(
            [
                "| id | category | failure_category | failure_reason | stop_reason | verifier_passed | within_budget |",
                "| --- | --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        for example in failed_examples:
            lines.append(
                "| "
                + " | ".join(
                    _md_escape(_format_value(example.get(key)))
                    for key in (
                        "id",
                        "category",
                        "failure_category",
                        "failure_reason",
                        "stop_reason",
                        "verifier_passed",
                        "within_budget",
                    )
                )
                + " |"
            )
            stdout = example.get("verifier_stdout")
            stderr = example.get("verifier_stderr")
            if stdout or stderr:
                lines.append(
                    f"- `{_md_escape(str(example.get('id', 'unknown')))}` verifier output: "
                    f"stdout={_format_value(stdout)} stderr={_format_value(stderr)}"
                )
    else:
        lines.append("- No failing examples are present in this harness artifact.")

    grouped_examples = harness.get("failure_examples_by_category", {})
    if grouped_examples:
        lines.extend(["", "Failure examples by category:", ""])
        for category, examples in grouped_examples.items():
            ids = ", ".join(
                f"`{_md_escape(example.get('id', 'unknown'))}`/{_md_escape(example.get('failure_category', 'unknown'))}"
                for example in examples
            )
            lines.append(f"- `{_md_escape(category)}`: {ids}")

    lines.extend(
        [
            "",
            "## 不应过度声称的探索指标",
            "",
            f"- Current fixed benchmark task_count is `{harness['task_count']}`; treat rates as regression signals, not online quality claims.",
            "- Fake/scripted model results prove harness wiring and verifier behavior, not real-model generalization.",
            "- Missing ablation artifacts are shown as `missing` / `not_available` instead of inferred.",
            "- Security counts reflect recorded permission evidence in artifacts; they are not a full red-team suite.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_ablation(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"### {_md_escape(summary['label'])}",
        "",
        f"- Source: `{_md_escape(summary['path'])}`",
        f"- Status: `{_md_escape(summary['status'])}`",
    ]
    if summary["status"] == STATUS_MISSING:
        lines.append("- Artifact is missing; all configured metrics are `not_available`.")
    elif summary["status"] == STATUS_INVALID:
        lines.append(f"- Artifact could not be read: `{_md_escape(summary.get('error', 'invalid artifact'))}`.")
    elif summary.get("missing_variants"):
        lines.append("- Missing variants: `" + "`, `".join(_md_escape(item) for item in summary["missing_variants"]) + "`.")
    else:
        lines.append("- All configured variants are present.")

    metrics = list(summary["metrics"])
    lines.extend(["", "| variant | " + " | ".join(f"`{metric}`" for metric in metrics) + " |"])
    lines.append("| --- | " + " | ".join("---:" for _ in metrics) + " |")
    for variant, values in summary["variants"].items():
        rendered = [_format_value(values.get(metric, NOT_AVAILABLE)) for metric in metrics]
        lines.append(f"| `{_md_escape(variant)}` | " + " | ".join(rendered) + " |")
    return lines


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _artifact_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("rows", [])
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("benchmark artifact rows must be a list")
    return [row for row in rows if isinstance(row, dict)]


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _is_true(row.get(key)))


def _failure_category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = _text(row.get("failure_category")).strip()
        if category:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _category_metrics(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = _text(row.get("category"), "uncategorized")
        grouped.setdefault(category, []).append(row)

    if grouped:
        metrics = {}
        for category, items in sorted(grouped.items()):
            total = len(items)
            passed = _count_true(items, "passed")
            metrics[category] = {
                "total_tasks": total,
                "passed": passed,
                "failed": max(total - passed, 0),
                "pass_rate": _rate(passed, total),
                "median_tool_steps": _median(_number(row.get("tool_steps"), 0.0) for row in items),
            }
        return metrics

    summary_categories = summary.get("category_summary", {}) if isinstance(summary, dict) else {}
    if isinstance(summary_categories, dict):
        result = {}
        for category, payload in sorted(summary_categories.items()):
            if not isinstance(payload, dict):
                continue
            result[str(category)] = {
                "total_tasks": _int_value(payload.get("total_tasks"), 0),
                "passed": _int_value(payload.get("passed"), 0),
                "failed": _int_value(payload.get("failed"), 0),
                "pass_rate": _float_value(payload.get("pass_rate"), 0.0),
                "median_tool_steps": _float_value(payload.get("median_tool_steps"), 0.0),
            }
        return result
    return {}


def _failed_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if _is_true(row.get("passed")):
            continue
        examples.append(
            {
                "id": _text(row.get("id"), "unknown"),
                "category": _text(row.get("category")),
                "failure_category": _text(row.get("failure_category"), "unknown") or "unknown",
                "failure_reason": _failure_reason(row),
                "stop_reason": _text(row.get("stop_reason")),
                "verifier_passed": _is_true(row.get("verifier_passed")),
                "within_budget": _is_true(row.get("within_budget")),
                "expected_artifact_exists": _is_true(row.get("expected_artifact_exists")),
                "verifier_exit_code": _int_value(row.get("verifier_exit_code"), 0),
                "verifier_stdout": _clip(_text(row.get("verifier_stdout")), 240),
                "verifier_stderr": _clip(_text(row.get("verifier_stderr")), 240),
            }
        )
    return examples


def _failure_examples_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in _failed_examples(rows):
        category = example.get("category") or "uncategorized"
        grouped.setdefault(str(category), []).append(example)
    return dict(sorted(grouped.items()))


def _failure_reason(row: dict[str, Any]) -> str:
    category = _text(row.get("failure_category")).strip()
    if category == "missing_artifact":
        return "expected artifact was not created"
    if category == "budget_exceeded":
        return f"tool_steps={_int_value(row.get('tool_steps'), 0)} exceeded step_budget={_int_value(row.get('step_budget'), 0)}"
    if category == "verifier_failed":
        return f"verifier command exited with code {_int_value(row.get('verifier_exit_code'), 0)}"
    if category == "failure_stop_reason":
        return f"stop_reason={_text(row.get('stop_reason'), NOT_AVAILABLE)}"
    return category or "failure category was not recorded"


def _iter_permission_evidence(value: Any):
    if isinstance(value, dict):
        if _permission_reason(value):
            yield value
            return
        for child in value.values():
            yield from _iter_permission_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_permission_evidence(child)


def _permission_reason(evidence: dict[str, Any]) -> str:
    reason = _text(evidence.get("permission_reason")).strip()
    details = evidence.get("details")
    if not reason and isinstance(details, dict):
        reason = _text(details.get("permission_reason")).strip()
    recovery = evidence.get("recovery")
    if not reason and isinstance(recovery, dict):
        reason = _permission_reason(recovery)
    return reason


def _permission_action(evidence: dict[str, Any]) -> str:
    action = _text(evidence.get("permission_action")).strip()
    metadata = evidence.get("metadata")
    if not action and isinstance(metadata, dict):
        action = _text(metadata.get("permission_action")).strip()
    return action


def _is_permission_block(evidence: dict[str, Any], action: str) -> bool:
    if action == "deny":
        return True
    if evidence.get("kind") == "permission_denied":
        return True
    if evidence.get("event") == "tool_rejected":
        return True
    if evidence.get("tool_status") == "rejected":
        return True
    if evidence.get("stop_reason") == "approval_denied":
        return True
    return False


def _extract_variant_payloads(artifact: dict[str, Any], spec: AblationSpec) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    raw_variants = artifact.get("variants")
    if isinstance(raw_variants, dict):
        for variant, payload in raw_variants.items():
            if variant in spec.variants and isinstance(payload, dict):
                payloads[variant] = payload
    elif isinstance(raw_variants, list):
        for payload in raw_variants:
            if not isinstance(payload, dict):
                continue
            variant = _text(payload.get("variant") or payload.get("name")).strip()
            if variant in spec.variants:
                payloads[variant] = payload

    for variant in spec.variants:
        payload = artifact.get(variant)
        if variant not in payloads and isinstance(payload, dict):
            payloads[variant] = payload

    rows = artifact.get("rows", [])
    if isinstance(rows, list):
        for variant, payload in _aggregate_variant_rows(rows, spec).items():
            payloads.setdefault(variant, payload)
    return payloads


def _aggregate_variant_rows(rows: list[Any], spec: AblationSpec) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        variant = _text(row.get("variant") or row.get("name")).strip()
        if variant in spec.variants:
            grouped.setdefault(variant, []).append(row)

    aggregated: dict[str, dict[str, Any]] = {}
    for variant, items in grouped.items():
        values: dict[str, Any] = {}
        for metric in spec.metrics:
            metric_values = [_number(item.get(metric), None) for item in items]
            metric_values = [value for value in metric_values if value is not None]
            if metric_values:
                values[metric] = _average(metric_values)
        aggregated[variant] = values
    return aggregated


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _dict_ints(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _int_value(item, 0) for key, item in sorted(value.items())}


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _average(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def _median(values) -> float:
    items = sorted(float(item or 0.0) for item in values)
    if not items:
        return 0.0
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2


def _numeric_summary(values) -> dict[str, Any]:
    items = [float(item) for item in values if item is not None]
    if not items:
        return {"count": 0, "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(items),
        "avg": _average(items),
        "median": _median(items),
        "min": min(items),
        "max": max(items),
    }


def _number(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return "`" + _md_escape(json.dumps(value, sort_keys=True, ensure_ascii=False)) + "`"
    if value is None:
        return f"`{NOT_AVAILABLE}`"
    text = str(value)
    if text == NOT_AVAILABLE:
        return f"`{NOT_AVAILABLE}`"
    return _md_escape(text)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 15)].rstrip() + " [truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
