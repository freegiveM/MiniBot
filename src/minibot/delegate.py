from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .hooks import HookManager
from .tools import OBSERVATION_SUCCEEDED, ToolObservation, ToolRegistry
from .workspace import clip, now


DELEGATE_SCHEMA_VERSION = 1
DEFAULT_ALLOWED_TOOLS = ("list_files", "read_file", "search", "read_memory")
DEFAULT_MAX_STEPS = 3
MAX_MAX_STEPS = 12
SUMMARY_LIMIT = 700
EVIDENCE_LIMIT = 5
FILES_READ_LIMIT = 10
OPEN_QUESTIONS_LIMIT = 5
NOTE_LIMIT = 220
NEXT_STEP_LIMIT = 300
CONFIDENCE_VALUES = {"low", "medium", "high", "unknown"}


def new_delegate_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"delegate_{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class DelegateTask:
    task: str
    allowed_tools: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ALLOWED_TOOLS)
    max_steps: int = DEFAULT_MAX_STEPS
    read_only: bool = True
    id: str = field(default_factory=new_delegate_id)

    def __post_init__(self) -> None:
        task = str(self.task or "").strip()
        if not task:
            raise ValueError("delegate task must not be empty")
        object.__setattr__(self, "task", task)

        delegate_id = str(self.id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", delegate_id):
            raise ValueError("delegate id contains invalid characters")
        object.__setattr__(self, "id", delegate_id)

        max_steps = _int(self.max_steps, "max_steps")
        if max_steps < 1:
            raise ValueError("delegate max_steps must be positive")
        if max_steps > MAX_MAX_STEPS:
            raise ValueError(f"delegate max_steps must be <= {MAX_MAX_STEPS}")
        object.__setattr__(self, "max_steps", max_steps)

        tools = _normalize_allowed_tools(self.allowed_tools)
        object.__setattr__(self, "allowed_tools", tools)
        object.__setattr__(self, "read_only", bool(self.read_only))

    @classmethod
    def from_args(cls, args: dict | None) -> "DelegateTask":
        args = args or {}
        if not isinstance(args, dict):
            raise ValueError("delegate args must be an object")
        raw_allowed = args.get("allowed_tools", None)
        if raw_allowed is None and "allowed_tools_json" in args:
            raw_allowed = _load_allowed_tools_json(args.get("allowed_tools_json"))
        if raw_allowed is None:
            raw_allowed = DEFAULT_ALLOWED_TOOLS
        return cls(
            task=str(args.get("task", "")),
            allowed_tools=tuple(raw_allowed),
            max_steps=args.get("max_steps", DEFAULT_MAX_STEPS),
            read_only=_bool(args.get("read_only", True), "read_only"),
            id=str(args.get("id") or new_delegate_id()),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "allowed_tools": list(self.allowed_tools),
            "max_steps": self.max_steps,
            "read_only": self.read_only,
        }


def _int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _load_allowed_tools_json(value: object) -> list[str]:
    if not isinstance(value, str):
        raise ValueError("allowed_tools_json must be a string")
    if not value.strip():
        return list(DEFAULT_ALLOWED_TOOLS)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"allowed_tools_json must be valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("allowed_tools_json must encode a list")
    return payload


def _normalize_allowed_tools(raw_tools: Sequence[object]) -> tuple[str, ...]:
    if isinstance(raw_tools, str):
        raise ValueError("delegate allowed_tools must be a list")
    if not raw_tools:
        raw_tools = DEFAULT_ALLOWED_TOOLS
    tools: list[str] = []
    seen = set()
    for raw in raw_tools:
        name = str(raw or "").strip()
        if not name:
            raise ValueError("delegate allowed_tools must not contain empty names")
        if name == "delegate":
            raise ValueError("child agents cannot use delegate")
        if name not in seen:
            tools.append(name)
            seen.add(name)
    return tuple(tools)


def compose_child_prompt(task: DelegateTask) -> str:
    return "\n".join(
        [
            "You are a MiniBot delegate running in an isolated child context.",
            "Investigate only the delegated task below. Do not assume parent conversation history exists.",
            "Use tools when repository facts are needed. The delegate tool is unavailable to you.",
            "Return only one JSON object with these keys:",
            "- summary: concise answer for the parent agent",
            "- evidence: list of objects with file, line, note",
            "- files_read: list of workspace-relative files you inspected",
            "- confidence: low, medium, high, or unknown",
            "- open_questions: list of unresolved questions",
            "- recommended_next_step: one concrete next action for the parent agent",
            "Keep the JSON concise; do not include raw tool output.",
            "",
            f"Allowed tools: {', '.join(task.allowed_tools)}",
            f"Read only: {task.read_only}",
            "",
            "Delegated task:",
            task.task,
        ]
    )


def validate_delegate_result(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("delegate result must be an object")
    required = ("summary", "evidence", "files_read", "confidence", "open_questions", "recommended_next_step")
    for key in required:
        if key not in payload:
            raise ValueError(f"missing delegate result field: {key}")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("delegate summary must be a non-empty string")
    if not isinstance(payload["evidence"], list):
        raise ValueError("delegate evidence must be a list")
    if not isinstance(payload["files_read"], list):
        raise ValueError("delegate files_read must be a list")
    if not isinstance(payload["confidence"], str):
        raise ValueError("delegate confidence must be a string")
    if not isinstance(payload["open_questions"], list):
        raise ValueError("delegate open_questions must be a list")
    if not isinstance(payload["recommended_next_step"], str):
        raise ValueError("delegate recommended_next_step must be a string")


def normalize_delegate_result(raw: object) -> tuple[dict, dict]:
    metadata = {"schema_valid": True, "schema_error": ""}
    if isinstance(raw, dict):
        payload = raw
    else:
        text = _strip_json_fence(str(raw or "").strip())
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            payload = {
                "summary": clip(str(raw or "").strip() or "Delegate returned no summary.", SUMMARY_LIMIT),
                "evidence": [],
                "files_read": [],
                "confidence": "low",
                "open_questions": ["Delegate result was not valid JSON."],
                "recommended_next_step": "Review the delegate artifact before relying on this result.",
            }
            metadata = {"schema_valid": False, "schema_error": str(exc)}
    try:
        validate_delegate_result(payload)
    except ValueError as exc:
        metadata = {"schema_valid": False, "schema_error": str(exc)}
        payload = {
            "summary": clip(str(payload.get("summary", "")) or "Delegate result did not match the schema.", SUMMARY_LIMIT),
            "evidence": payload.get("evidence", []) if isinstance(payload, dict) else [],
            "files_read": payload.get("files_read", []) if isinstance(payload, dict) else [],
            "confidence": str(payload.get("confidence", "low")) if isinstance(payload, dict) else "low",
            "open_questions": payload.get("open_questions", []) if isinstance(payload, dict) else [],
            "recommended_next_step": str(payload.get("recommended_next_step", "Review delegate artifact.")) if isinstance(payload, dict) else "Review delegate artifact.",
        }
    return _coerce_delegate_result(payload), metadata


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    return match.group(1).strip() if match else text


def _coerce_delegate_result(payload: dict) -> dict:
    confidence = str(payload.get("confidence", "unknown")).strip().lower() or "unknown"
    if confidence not in CONFIDENCE_VALUES:
        confidence = "unknown"
    return {
        "summary": clip(str(payload.get("summary", "")).strip(), SUMMARY_LIMIT),
        "evidence": [_coerce_evidence(item) for item in _list(payload.get("evidence"))],
        "files_read": [clip(str(item), NOTE_LIMIT) for item in _list(payload.get("files_read")) if str(item).strip()],
        "confidence": confidence,
        "open_questions": [clip(str(item), NOTE_LIMIT) for item in _list(payload.get("open_questions")) if str(item).strip()],
        "recommended_next_step": clip(str(payload.get("recommended_next_step", "")).strip(), NEXT_STEP_LIMIT),
    }


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _coerce_evidence(item: object) -> dict:
    if not isinstance(item, dict):
        return {"file": "", "line": None, "note": clip(str(item), NOTE_LIMIT)}
    line = item.get("line", None)
    if line in ("", None):
        line = None
    else:
        try:
            line = int(line)
        except (TypeError, ValueError):
            line = None
    return {
        "file": clip(str(item.get("file", "")), NOTE_LIMIT),
        "line": line,
        "note": clip(str(item.get("note", "")), NOTE_LIMIT),
    }


def bounded_delegate_observation(result: dict) -> dict:
    normalized = _coerce_delegate_result(result)
    return {
        "summary": normalized["summary"],
        "evidence": normalized["evidence"][:EVIDENCE_LIMIT],
        "files_read": normalized["files_read"][:FILES_READ_LIMIT],
        "confidence": normalized["confidence"],
        "open_questions": normalized["open_questions"][:OPEN_QUESTIONS_LIMIT],
        "recommended_next_step": normalized["recommended_next_step"],
    }


def filter_tool_registry(registry: ToolRegistry, allowed_tools: Sequence[str]) -> ToolRegistry:
    allowed = _normalize_allowed_tools(allowed_tools)
    filtered = ToolRegistry()
    specs = registry.specs()
    handlers = getattr(registry, "_handlers", {})
    for name in allowed:
        if name not in specs:
            raise ValueError(f"delegate allowed tool is unavailable: {name}")
        filtered.register(specs[name], handlers[name])
    return filtered


class DelegateScheduler:
    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / ".minibot" / "delegates"
        self.root.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, delegate_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(delegate_id)):
            raise ValueError("delegate id contains invalid characters")
        return self.root / f"{delegate_id}.json"

    def run(self, parent_agent, task: DelegateTask) -> ToolObservation:
        from .runtime import MiniBot

        child = MiniBot(
            model_client=parent_agent.model_client,
            workspace=parent_agent.workspace,
            session_store=parent_agent.session_store,
            run_store=parent_agent.run_store,
            approval_policy=parent_agent.approval_policy,
            max_steps=task.max_steps,
            max_new_tokens=parent_agent.max_new_tokens,
            depth=parent_agent.depth + 1,
            max_depth=parent_agent.depth + 1,
            read_only=task.read_only,
            hook_manager=HookManager(),
            strict_action_protocol=False,
        )
        child.tools = filter_tool_registry(child.tools, task.allowed_tools)
        child.prefix = child.build_prefix()

        raw_result = child.ask(compose_child_prompt(task))
        result, schema_metadata = normalize_delegate_result(raw_result)
        observation_payload = bounded_delegate_observation(result)
        artifact_path = self.artifact_path(task.id)
        artifact_ref = artifact_path.relative_to(parent_agent.root).as_posix()
        observation_payload.update(
            {
                "delegate_id": task.id,
                "artifact_ref": artifact_ref,
                "child_run_id": child.session.get("runs", {}).get("last_run_id", ""),
                "schema_valid": schema_metadata["schema_valid"],
            }
        )
        artifact = {
            "schema_version": DELEGATE_SCHEMA_VERSION,
            "id": task.id,
            "created_at": now(),
            "parent": {
                "session_id": parent_agent.session.get("id", ""),
                "run_id": parent_agent.current_task_state.run_id if parent_agent.current_task_state else "",
                "depth": parent_agent.depth,
            },
            "child": {
                "session_id": child.session.get("id", ""),
                "run_id": child.session.get("runs", {}).get("last_run_id", ""),
                "depth": child.depth,
                "max_depth": child.max_depth,
                "read_only": child.read_only,
                "allowed_tools": list(task.allowed_tools),
            },
            "task": task.to_dict(),
            "raw_result": raw_result,
            "raw_result_chars": len(str(raw_result)),
            "result": result,
            "result_metadata": schema_metadata,
            "parent_observation": observation_payload,
        }
        self._write_json_atomic(artifact_path, artifact)
        _remember_delegate(parent_agent, task.id, artifact_ref)
        return ToolObservation(
            status=OBSERVATION_SUCCEEDED,
            content="delegate_result:\n" + json.dumps(observation_payload, sort_keys=True, ensure_ascii=False),
            metadata={
                "delegate_id": task.id,
                "delegate_artifact": artifact_ref,
                "delegate_child_run_id": observation_payload["child_run_id"],
                "delegate_schema_valid": schema_metadata["schema_valid"],
                "delegate_schema_error": schema_metadata["schema_error"],
                "affected_paths": [],
            },
        )

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)


def _remember_delegate(parent_agent, delegate_id: str, artifact_ref: str) -> None:
    item = {"id": delegate_id, "artifact_ref": artifact_ref, "created_at": now()}
    delegates = parent_agent.session.setdefault("delegate_artifacts", [])
    delegates.append(item)
    parent_agent.session["delegate_artifacts"] = delegates[-20:]
    parent_agent.session["updated_at"] = now()
    parent_agent.session_path = parent_agent.session_store.save(parent_agent.session)


def run_delegate(parent_agent, args: dict) -> ToolObservation:
    task = DelegateTask.from_args(args)
    scheduler = DelegateScheduler(parent_agent.root)
    return scheduler.run(parent_agent, task)
