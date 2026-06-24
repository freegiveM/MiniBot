from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .todo_state import TodoState
from .workspace import IGNORED_PATH_NAMES, clip


RISK_LEVEL_SAFE = "safe"
RISK_LEVEL_RISKY = "risky"
OBSERVATION_SUCCEEDED = "succeeded"
OBSERVATION_ERROR = "error"

ToolHandler = Callable[[object, dict], object]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict
    risk_level: str = RISK_LEVEL_SAFE

    @property
    def risky(self) -> bool:
        return self.risk_level != RISK_LEVEL_SAFE

    def to_prompt_dict(self) -> dict:
        return {
            "description": self.description,
            "schema": self.schema,
            "risk_level": self.risk_level,
            "risky": self.risky,
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallBatch:
    calls: tuple[ToolCall, ...]

    def __iter__(self):
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)


@dataclass
class ToolObservation:
    status: str
    content: str
    metadata: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "content": self.content,
            "metadata": self.metadata,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> "ToolRegistry":
        if not spec.name:
            raise ValueError("tool name must not be empty")
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        if not callable(handler):
            raise ValueError(f"tool handler is not callable: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler
        return self

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def items(self):
        return self._specs.items()

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"unknown tool: {name}") from exc

    def specs(self) -> dict[str, ToolSpec]:
        return dict(self._specs)

    def prompt_specs(self) -> dict[str, dict]:
        return {name: spec.to_prompt_dict() for name, spec in sorted(self._specs.items())}

    def validate_tool_call(self, name: str, args: dict | None, runtime_context) -> None:
        if name not in self._specs:
            raise ValueError(f"unknown tool: {name}")
        args = _args(args)
        _validate_schema_args(self._specs[name].schema, args)
        if name in BASE_TOOL_SPECS:
            _validate_tool_call(runtime_context, name, args)

    def dispatch(self, name: str, args: dict | None, runtime_context, *, validate: bool = True) -> ToolObservation:
        args = _args(args)
        if validate:
            self.validate_tool_call(name, args, runtime_context)
        else:
            self.spec(name)
        handler = self._handlers[name]
        try:
            result = handler(runtime_context, args)
        except Exception as exc:
            return ToolObservation(
                status=OBSERVATION_ERROR,
                content=f"tool error: {exc}",
                metadata={"affected_paths": _affected_paths(args)},
                error=str(exc),
            )
        if isinstance(result, ToolObservation):
            observation = result
        else:
            content = str(result)
            status = OBSERVATION_SUCCEEDED
            first_line = content.splitlines()[0] if content.splitlines() else ""
            if name == "run_shell" and "exit_code: 0" not in first_line:
                status = OBSERVATION_ERROR
            observation = ToolObservation(
                status=status,
                content=content,
                metadata={"affected_paths": _affected_paths(args)},
            )
        observation.metadata.setdefault("affected_paths", _affected_paths(args))
        return observation


BASE_TOOL_SPECS = {
    "list_files": ToolSpec(
        name="list_files",
        description="List workspace files.",
        schema={"path": "str='.'"},
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read a UTF-8 file by line range.",
        schema={"path": "str", "start": "int=1", "end": "int=200"},
    ),
    "search": ToolSpec(
        name="search",
        description="Search the workspace.",
        schema={"pattern": "str", "path": "str='.'"},
    ),
    "read_memory": ToolSpec(
        name="read_memory",
        description="Read bounded long-term memory topics.",
        schema={"topic": "str='index'", "max_chars": "int=2000"},
    ),
    "run_shell": ToolSpec(
        name="run_shell",
        description="Run a shell command.",
        schema={"command": "str", "timeout": "int=20"},
        risk_level=RISK_LEVEL_RISKY,
    ),
    "write_file": ToolSpec(
        name="write_file",
        description="Write a text file.",
        schema={"path": "str", "content": "str"},
        risk_level=RISK_LEVEL_RISKY,
    ),
    "patch_file": ToolSpec(
        name="patch_file",
        description="Replace one exact text block.",
        schema={"path": "str", "old_text": "str", "new_text": "str"},
        risk_level=RISK_LEVEL_RISKY,
    ),
    "todo_write": ToolSpec(
        name="todo_write",
        description="Replace the current task plan. Tracks planning only and does not execute tasks.",
        schema={"items_json": "str"},
    ),
    "delegate": ToolSpec(
        name="delegate",
        description="Ask a bounded read-only child agent to investigate.",
        schema={"task": "str", "max_steps": "int=3"},
    ),
}


def build_tool_registry(agent) -> ToolRegistry:
    registry = ToolRegistry()
    for name, spec in BASE_TOOL_SPECS.items():
        if name == "delegate" and agent.depth >= agent.max_depth:
            continue
        registry.register(spec, _TOOL_RUNNERS[name])
    return registry


def tool_signature(tools: ToolRegistry | dict) -> str:
    if isinstance(tools, ToolRegistry):
        schema = {name: spec.schema for name, spec in sorted(tools.items())}
    else:
        schema = {name: spec["schema"] for name, spec in sorted(tools.items())}
    return json.dumps(schema, sort_keys=True)


def normalize_tool_calls(payload: object) -> ToolCallBatch:
    if isinstance(payload, ToolCallBatch):
        return payload
    if isinstance(payload, ToolCall):
        return ToolCallBatch((payload,))
    if isinstance(payload, dict) and "calls" in payload and "name" not in payload:
        raw_calls = payload["calls"]
    elif isinstance(payload, dict):
        raw_calls = [payload]
    elif isinstance(payload, list):
        raw_calls = payload
    else:
        raise ValueError("tool payload must be an object or list of objects")

    if not isinstance(raw_calls, list):
        raise ValueError("tool calls must be a list")
    calls = []
    for index, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            raise ValueError(f"tool call at index {index} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tool call at index {index} is missing name")
        args = item.get("args", {}) or {}
        if not isinstance(args, dict):
            raise ValueError(f"tool call args at index {index} must be an object")
        calls.append(ToolCall(name=name.strip(), args=args))
    if not calls:
        raise ValueError("tool call batch must not be empty")
    return ToolCallBatch(tuple(calls))


def validate_tool(agent, name: str, args: dict | None) -> None:
    if isinstance(getattr(agent, "tools", None), ToolRegistry):
        agent.tools.validate_tool_call(name, args, agent)
        return
    if name not in BASE_TOOL_SPECS:
        raise ValueError(f"unknown tool: {name}")
    args = _args(args)
    _validate_schema_args(BASE_TOOL_SPECS[name].schema, args)
    _validate_tool_call(agent, name, args)


def _args(args: dict | None) -> dict:
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ValueError("tool args must be an object")
    return args


def _affected_paths(args: dict) -> list[str]:
    path = args.get("path", "")
    return [path] if isinstance(path, str) and path.strip() else []


def _required_str(args: dict, key: str) -> str:
    if key not in args:
        raise ValueError(f"missing required arg: {key}")
    value = args[key]
    if not isinstance(value, str):
        raise ValueError(f"arg must be a string: {key}")
    if not value.strip():
        raise ValueError(f"arg must not be empty: {key}")
    return value


def _optional_str(args: dict, key: str, default: str) -> str:
    if key not in args:
        return default
    value = args[key]
    if not isinstance(value, str):
        raise ValueError(f"arg must be a string: {key}")
    return value


def _required_int(args: dict, key: str) -> int:
    if key not in args:
        raise ValueError(f"missing required arg: {key}")
    return _int_arg(args[key], key)


def _optional_int(args: dict, key: str, default: int) -> int:
    if key not in args:
        return default
    return _int_arg(args[key], key)


def _int_arg(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"arg must be an integer: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"arg must be an integer: {key}") from exc


def _validate_schema_args(schema: dict, args: dict) -> None:
    for key, descriptor in schema.items():
        text = str(descriptor)
        kind = text.split("=", 1)[0]
        required = "=" not in text
        if required and key not in args:
            raise ValueError(f"missing required arg: {key}")
        if key not in args:
            continue
        if kind == "str" and not isinstance(args[key], str):
            raise ValueError(f"arg must be a string: {key}")
        if kind == "int":
            _int_arg(args[key], key)


def _validate_tool_call(agent, name: str, args: dict) -> None:
    if name == "list_files":
        if not agent.path(_optional_str(args, "path", ".")).is_dir():
            raise ValueError("path is not a directory")
    elif name == "read_file":
        if not agent.path(_required_str(args, "path")).is_file():
            raise ValueError("path is not a file")
        start = _optional_int(args, "start", 1)
        end = _optional_int(args, "end", 200)
        if start < 1 or end < start:
            raise ValueError("invalid line range")
    elif name == "search":
        _required_str(args, "pattern")
        agent.path(_optional_str(args, "path", "."))
    elif name == "read_memory":
        max_chars = _optional_int(args, "max_chars", 2000)
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        _memory_path(agent, args)
    elif name == "run_shell":
        _required_str(args, "command")
        timeout = _optional_int(args, "timeout", 20)
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
    elif name == "write_file":
        path = agent.path(_required_str(args, "path"))
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing required arg: content")
        if not isinstance(args["content"], str):
            raise ValueError("arg must be a string: content")
    elif name == "patch_file":
        path = agent.path(_required_str(args, "path"))
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = _required_str(args, "old_text")
        if "new_text" not in args:
            raise ValueError("missing required arg: new_text")
        if not isinstance(args["new_text"], str):
            raise ValueError("arg must be a string: new_text")
        count = path.read_text(encoding="utf-8").count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
    elif name == "todo_write":
        TodoState().set_items(_todo_items_from_args(args))
    elif name == "delegate":
        if agent.depth >= agent.max_depth:
            raise ValueError("delegate depth exceeded")
        _required_str(args, "task")
        max_steps = _optional_int(args, "max_steps", 3)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
    else:
        raise ValueError(f"unknown tool: {name}")


def _todo_items_from_args(args: dict) -> list:
    raw = _required_str(args, "items_json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"items_json must be valid JSON: {exc}") from exc
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]
    if not isinstance(payload, list):
        raise ValueError("items_json must encode a list of todo items")
    return payload


def tool_list_files(agent, args: dict) -> str:
    path = agent.path(args.get("path", "."))
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = [f"{'[D]' if entry.is_dir() else '[F]'} {entry.relative_to(agent.root)}" for entry in entries[:200]]
    return "\n".join(lines) or "(empty)"


def tool_read_file(agent, args: dict) -> str:
    path = agent.path(args["path"])
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(agent.root)}\n{body}"


def tool_search(agent, args: dict) -> str:
    pattern = str(args.get("pattern", "")).strip()
    path = agent.path(args.get("path", "."))
    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"
    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def _memory_path(agent, args: dict) -> Path:
    topic = str(args.get("topic", "index")).strip() or "index"
    memory_root = agent.root / ".minibot" / "memory"
    if topic in {"index", "MEMORY.md", "memory"}:
        path = memory_root / "MEMORY.md"
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", topic):
            raise ValueError("invalid memory topic")
        path = memory_root / "topics" / f"{topic}.md"
    resolved = path.resolve()
    allowed = [memory_root.resolve(), (memory_root / "topics").resolve()]
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError("memory path escapes memory root")
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("memory topic not found")
    return resolved


def tool_read_memory(agent, args: dict) -> str:
    path = _memory_path(agent, args)
    max_chars = max(1, min(int(args.get("max_chars", 2000)), 4000))
    return clip(path.read_text(encoding="utf-8", errors="replace"), max_chars)


def tool_run_shell(agent, args: dict) -> str:
    timeout = int(args.get("timeout", 20))
    result = subprocess.run(
        str(args.get("command", "")).strip(),
        cwd=agent.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=agent.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(agent, args: dict) -> str:
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


def tool_patch_file(agent, args: dict) -> str:
    path = agent.path(args["path"])
    old_text = str(args.get("old_text", ""))
    new_text = str(args.get("new_text", ""))
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root)}"


def tool_todo_write(agent, args: dict) -> ToolObservation:
    items = _todo_items_from_args(args)
    agent.todo_state.set_items(items)
    agent.persist_todo_state()
    summary = agent.todo_state.summary()
    return ToolObservation(
        status=OBSERVATION_SUCCEEDED,
        content=agent.todo_state.render_for_prompt(),
        metadata={"todo_summary": summary},
    )


def tool_delegate(agent, args: dict) -> str:
    from .runtime import MiniBot

    child = MiniBot(
        model_client=agent.model_client,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="never",
        max_steps=int(args.get("max_steps", 3)),
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth + 1,
        max_depth=agent.max_depth,
        read_only=True,
    )
    result = child.ask(str(args.get("task", "")).strip())
    return "delegate_result:\n" + clip(result, 2000)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "read_memory": tool_read_memory,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "todo_write": tool_todo_write,
    "delegate": tool_delegate,
}
