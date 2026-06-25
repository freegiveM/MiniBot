from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HARNESS_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_CONTEXT_ARTIFACT_PATH = Path("artifacts/context-ablation-v2.json")
DEFAULT_MEMORY_ARTIFACT_PATH = Path("artifacts/memory-ablation-v2.json")
DEFAULT_RECOVERY_ARTIFACT_PATH = Path("artifacts/recovery-ablation-v2.json")
DEFAULT_RETRIEVAL_ARTIFACT_PATH = Path("artifacts/retrieval-ablation-v2.json")
DEFAULT_REPORT_PATH = Path("artifacts/minibot-benchmark-core-report.md")

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
    security = aggregate_tool_boundary_security(artifact)

    result: dict[str, Any] = {
        "artifact_path": str(artifact_path),
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
        "avg_attempts": avg_attempts,
        "failure_category_counts": failure_category_counts,
        "failed_examples": _failed_examples(rows),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MiniBot benchmark artifacts.")
    parser.add_argument("--harness-artifact-path", default=str(DEFAULT_HARNESS_ARTIFACT_PATH))
    parser.add_argument("--context-artifact-path", default=str(DEFAULT_CONTEXT_ARTIFACT_PATH))
    parser.add_argument("--memory-artifact-path", default=str(DEFAULT_MEMORY_ARTIFACT_PATH))
    parser.add_argument("--recovery-artifact-path", default=str(DEFAULT_RECOVERY_ARTIFACT_PATH))
    parser.add_argument("--retrieval-artifact-path", default=str(DEFAULT_RETRIEVAL_ARTIFACT_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--write-core-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
