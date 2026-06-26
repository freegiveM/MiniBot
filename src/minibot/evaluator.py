from __future__ import annotations

import argparse
import hashlib
import json
import locale
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .context_manager import ContextManager
from .models import FakeModelClient
from .runtime import MiniBot, SessionStore
from .task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from .tools import ToolRegistry
from .workspace import WorkspaceContext


SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_WORKSPACE_ROOT = Path("artifacts/evaluator-workspaces")
REQUIRED_TASK_FIELDS = (
    "id",
    "prompt",
    "fixture_repo",
    "allowed_tools",
    "step_budget",
    "expected_artifact",
    "verifier",
    "category",
)
FAILURE_MISSING_ARTIFACT = "missing_artifact"
FAILURE_BUDGET_EXCEEDED = "budget_exceeded"
FAILURE_VERIFIER_FAILED = "verifier_failed"
FAILURE_STOP_REASON = "failure_stop_reason"
FAILURE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    prompt: str
    fixture_repo: str
    allowed_tools: tuple[str, ...]
    step_budget: int
    expected_artifact: tuple[str, ...]
    verifier: str
    category: str
    model_outputs: tuple[str, ...] = field(default_factory=tuple)
    setup: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict, benchmark_root: Path) -> "BenchmarkTask":
        if not isinstance(payload, dict):
            raise ValueError("benchmark task must be an object")
        for field_name in REQUIRED_TASK_FIELDS:
            if field_name not in payload:
                raise ValueError(f"benchmark task missing required field: {field_name}")
        task_id = _required_text(payload, "id")
        fixture_repo = _required_text(payload, "fixture_repo")
        fixture_path = (benchmark_root / fixture_repo).resolve()
        if not fixture_path.exists() or not fixture_path.is_dir():
            raise ValueError(f"fixture_repo not found for task {task_id}: {fixture_repo}")
        step_budget = _positive_int(payload.get("step_budget"), f"{task_id}.step_budget")
        allowed_tools = _string_list(payload.get("allowed_tools"), f"{task_id}.allowed_tools")
        expected_artifact = _artifact_paths(payload.get("expected_artifact"), f"{task_id}.expected_artifact")
        model_outputs = _string_list(payload.get("model_outputs", []), f"{task_id}.model_outputs")
        setup = payload.get("setup", {})
        if not isinstance(setup, dict):
            raise ValueError(f"{task_id}.setup must be an object")
        return cls(
            id=task_id,
            prompt=_required_text(payload, "prompt"),
            fixture_repo=fixture_repo,
            allowed_tools=tuple(allowed_tools),
            step_budget=step_budget,
            expected_artifact=tuple(expected_artifact),
            verifier=_required_text(payload, "verifier"),
            category=_required_text(payload, "category"),
            model_outputs=tuple(model_outputs),
            setup=dict(setup),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "fixture_repo": self.fixture_repo,
            "allowed_tools": list(self.allowed_tools),
            "step_budget": self.step_budget,
            "expected_artifact": list(self.expected_artifact),
            "verifier": self.verifier,
            "category": self.category,
            "model_outputs": list(self.model_outputs),
            "setup": dict(self.setup),
        }


@dataclass(frozen=True)
class Benchmark:
    schema_version: int
    description: str
    tasks: tuple[BenchmarkTask, ...]
    path: Path


def load_benchmark(path: str | Path) -> Benchmark:
    benchmark_path = Path(path).resolve()
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark file must contain an object")
    schema_version = _positive_int(payload.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark schema_version: {schema_version}")
    description = _required_text(payload, "description")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("benchmark tasks must be a non-empty list")
    tasks = []
    seen_ids = set()
    for item in raw_tasks:
        task = BenchmarkTask.from_dict(item, benchmark_path.parent)
        if task.id in seen_ids:
            raise ValueError(f"duplicate benchmark task id: {task.id}")
        seen_ids.add(task.id)
        tasks.append(task)
    return Benchmark(schema_version=schema_version, description=description, tasks=tuple(tasks), path=benchmark_path)


def run_fixed_benchmark(
    benchmark_path: str | Path = "benchmarks/coding_tasks.json",
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    workspace_root: str | Path = DEFAULT_WORKSPACE_ROOT,
) -> dict:
    benchmark = load_benchmark(benchmark_path)
    artifact_path = Path(artifact_path).resolve()
    workspace_root = Path(workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in benchmark.tasks:
        rows.append(_run_task(task, benchmark.path.parent, artifact_path.parent, workspace_root))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "path": _relpath(benchmark.path, artifact_path.parent),
            "description": benchmark.description,
            "task_count": len(benchmark.tasks),
        },
        "runtime": _runtime_metadata(),
        "reproducibility": {
            "fixture_snapshot_id": _benchmark_snapshot_id(benchmark.path.parent, benchmark.tasks),
            "model_name": "FakeModelClient",
            "model_version": "scripted-deterministic",
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512},
            "timezone": datetime.now(timezone.utc).astimezone().tzname() or "UTC",
            "locale": locale.setlocale(locale.LC_CTYPE, None),
        },
        "summary": _summarize_rows(rows),
        "rows": rows,
    }
    _write_json_atomic(artifact_path, artifact)
    return artifact


def _run_task(task: BenchmarkTask, benchmark_root: Path, artifact_root: Path, workspace_root: Path) -> dict:
    fixture_source = (benchmark_root / task.fixture_repo).resolve()
    fixture_copy = workspace_root / _safe_task_dir(task.id)
    if fixture_copy.exists():
        shutil.rmtree(fixture_copy)
    shutil.copytree(fixture_source, fixture_copy)

    model = FakeModelClient(list(task.model_outputs), model="fake-scripted")
    workspace = WorkspaceContext.build(fixture_copy, repo_root_override=fixture_copy)
    agent = MiniBot(
        model_client=model,
        workspace=workspace,
        session_store=SessionStore(fixture_copy / ".minibot" / "sessions"),
        approval_policy="auto",
        max_steps=task.step_budget,
        max_new_tokens=512,
    )
    agent.tools = _filter_tools(agent.tools, task.allowed_tools)
    agent.prefix = agent.build_prefix()
    _apply_task_setup(agent, task.setup)
    final_answer = agent.ask(task.prompt)

    run_id = agent.session.get("runs", {}).get("last_run_id", "")
    report_path = fixture_copy / ".minibot" / "runs" / run_id / "report.json"
    trace_path = fixture_copy / ".minibot" / "runs" / run_id / "trace.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    task_state = report.get("task_state", {})
    verifier_result = _run_verifier(task.verifier, fixture_copy)
    expected_artifact_exists = all((fixture_copy / path).exists() for path in task.expected_artifact)
    tool_steps = int(task_state.get("tool_steps", 0) or 0)
    attempts = int(task_state.get("attempts", 0) or 0)
    stop_reason = str(task_state.get("stop_reason", ""))
    within_budget = tool_steps <= task.step_budget
    passed = (
        within_budget
        and verifier_result["passed"]
        and expected_artifact_exists
        and stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
    )
    failure_category = "" if passed else _failure_category(
        expected_artifact_exists=expected_artifact_exists,
        within_budget=within_budget,
        verifier_passed=verifier_result["passed"],
        stop_reason=stop_reason,
    )
    return {
        "id": task.id,
        "prompt": task.prompt,
        "category": task.category,
        "allowed_tools": list(task.allowed_tools),
        "step_budget": task.step_budget,
        "expected_artifact": list(task.expected_artifact),
        "expected_artifact_exists": expected_artifact_exists,
        "verifier": task.verifier,
        "verifier_passed": verifier_result["passed"],
        "verifier_exit_code": verifier_result["exit_code"],
        "verifier_stdout": verifier_result["stdout"],
        "verifier_stderr": verifier_result["stderr"],
        "fixture_copy_relpath": _relpath(fixture_copy, artifact_root),
        "report_relpath": _relpath(report_path, artifact_root),
        "trace_relpath": _relpath(trace_path, artifact_root),
        "run_id": run_id,
        "session_id": agent.session.get("id", ""),
        "tool_steps": tool_steps,
        "attempts": attempts,
        "stop_reason": stop_reason,
        "within_budget": within_budget,
        "passed": passed,
        "failure_category": failure_category,
        "final_answer": final_answer,
        "report": {
            "task_state": task_state,
            "prompt_metadata": report.get("prompt_metadata", {}),
            "recovery": report.get("recovery", {}),
            "hooks": report.get("hooks", {}),
        },
    }


def _filter_tools(registry: ToolRegistry, allowed_tools: tuple[str, ...]) -> ToolRegistry:
    filtered = ToolRegistry()
    specs = registry.specs()
    handlers = getattr(registry, "_handlers", {})
    for name in allowed_tools:
        if name not in specs:
            raise ValueError(f"allowed tool is unavailable: {name}")
        filtered.register(specs[name], handlers[name])
    return filtered


def _run_verifier(command: str, cwd: Path) -> dict:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "exit_code": -1,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": "verifier timed out",
        }


def _summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    within_budget = sum(1 for row in rows if row.get("within_budget"))
    verifier_passed = sum(1 for row in rows if row.get("verifier_passed"))
    failures: dict[str, int] = {}
    for row in rows:
        category = row.get("failure_category") or ""
        if category:
            failures[category] = failures.get(category, 0) + 1
    category_summary = _category_summary(rows)
    return {
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _rate(passed, total),
        "within_budget_rate": _rate(within_budget, total),
        "verifier_pass_rate": _rate(verifier_passed, total),
        "median_tool_steps": _median(row.get("tool_steps", 0) for row in rows),
        "category_pass_rates": {category: item["pass_rate"] for category, item in category_summary.items()},
        "category_summary": category_summary,
        "failure_category_counts": failures,
        "failure_examples_by_category": _failure_examples_by_category(rows),
    }


def _apply_task_setup(agent: MiniBot, setup: dict) -> None:
    if not setup:
        return
    kind = str(setup.get("kind", "")).strip()
    if kind == "context_reduction":
        history_count = _non_negative_int(setup.get("history_count", 8), "setup.history_count")
        item_chars = _positive_int(setup.get("item_chars", 500), "setup.item_chars")
        total_budget = _positive_int(setup.get("total_budget", 2400), "setup.total_budget")
        latest_marker = str(setup.get("latest_marker", "LATEST_CONTEXT_MARKER_KEEP"))
        history = []
        for index in range(history_count):
            history.append(
                {
                    "role": "assistant",
                    "content": f"older benchmark context {index}: " + ("x" * item_chars),
                }
            )
        history.append({"role": "assistant", "content": latest_marker})
        agent.session["history"] = history
        agent.session_path = agent.session_store.save(agent.session)
        agent.context_manager = ContextManager(agent, total_budget=total_budget)
        return
    if kind == "memory_seed":
        text = str(setup.get("project_memory", "")).strip()
        if not text:
            raise ValueError("setup.project_memory must be a non-empty string")
        memory_root = agent.root / ".minibot" / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        (memory_root / "MEMORY.md").write_text(text + "\n", encoding="utf-8")
        return
    raise ValueError(f"unsupported benchmark setup kind: {kind}")


def _category_summary(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        category = str(row.get("category") or "uncategorized")
        grouped.setdefault(category, []).append(row)

    summary = {}
    for category, items in sorted(grouped.items()):
        total = len(items)
        passed = sum(1 for row in items if row.get("passed"))
        summary[category] = {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": _rate(passed, total),
            "median_tool_steps": _median(row.get("tool_steps", 0) for row in items),
        }
    return summary


def _failure_examples_by_category(rows: list[dict], limit_per_category: int = 3) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("passed"):
            continue
        category = str(row.get("category") or "uncategorized")
        examples = grouped.setdefault(category, [])
        if len(examples) >= limit_per_category:
            continue
        examples.append(
            {
                "id": str(row.get("id") or "unknown"),
                "failure_category": str(row.get("failure_category") or "unknown"),
                "stop_reason": str(row.get("stop_reason") or ""),
                "within_budget": bool(row.get("within_budget")),
                "verifier_passed": bool(row.get("verifier_passed")),
            }
        )
    return grouped


def _failure_category(
    *,
    expected_artifact_exists: bool,
    within_budget: bool,
    verifier_passed: bool,
    stop_reason: str,
) -> str:
    if not expected_artifact_exists:
        return FAILURE_MISSING_ARTIFACT
    if not within_budget:
        return FAILURE_BUDGET_EXCEEDED
    if not verifier_passed:
        return FAILURE_VERIFIER_FAILED
    if stop_reason != STOP_REASON_FINAL_ANSWER_RETURNED:
        return FAILURE_STOP_REASON
    return FAILURE_UNKNOWN


def _runtime_metadata() -> dict:
    branch = _git(["branch", "--show-current"], "-")
    commit = _git(["rev-parse", "--short", "HEAD"], "local")
    return {"commit_sha": commit or "local", "branch": branch or "-"}


def _benchmark_snapshot_id(root: Path, tasks: tuple[BenchmarkTask, ...]) -> str:
    hasher = hashlib.sha256()
    for task in sorted(tasks, key=lambda item: item.id):
        fixture = (root / task.fixture_repo).resolve()
        hasher.update(task.id.encode("utf-8"))
        for path in sorted(item for item in fixture.rglob("*") if item.is_file()):
            rel = path.relative_to(fixture).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def _safe_task_dir(task_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task_id)
    return safe or "task"


def _required_text(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return number


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a non-negative integer") from exc
    if number < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return number


def _string_list(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} must contain non-empty strings")
        result.append(item.strip())
    return result


def _artifact_paths(value: object, key: str) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(f"{key} must be a path string or list of path strings")
    result = _string_list(items, key)
    for item in result:
        candidate = Path(item)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{key} must stay within the fixture workspace")
    return result


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _median(values) -> float:
    items = sorted(float(value or 0) for value in values)
    if not items:
        return 0.0
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git(args: list[str], fallback: str = "") -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiniBot fixed benchmark tasks.")
    parser.add_argument("--benchmark-path", default="benchmarks/coding_tasks.json")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_fixed_benchmark(
        benchmark_path=args.benchmark_path,
        artifact_path=args.artifact_path,
        workspace_root=args.workspace_root,
    )
    print(json.dumps(artifact["summary"], sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
