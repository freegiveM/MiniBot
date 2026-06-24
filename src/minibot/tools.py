from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from functools import partial
from pathlib import Path

from .workspace import IGNORED_PATH_NAMES, clip


BASE_TOOL_SPECS = {
    "list_files": {"schema": {"path": "str='.'"}, "risky": False, "description": "List workspace files."},
    "read_file": {"schema": {"path": "str", "start": "int=1", "end": "int=200"}, "risky": False, "description": "Read a UTF-8 file by line range."},
    "search": {"schema": {"pattern": "str", "path": "str='.'"}, "risky": False, "description": "Search the workspace."},
    "read_memory": {"schema": {"topic": "str='index'", "max_chars": "int=2000"}, "risky": False, "description": "Read bounded long-term memory topics."},
    "run_shell": {"schema": {"command": "str", "timeout": "int=20"}, "risky": True, "description": "Run a shell command."},
    "write_file": {"schema": {"path": "str", "content": "str"}, "risky": True, "description": "Write a text file."},
    "patch_file": {"schema": {"path": "str", "old_text": "str", "new_text": "str"}, "risky": True, "description": "Replace one exact text block."},
    "delegate": {"schema": {"task": "str", "max_steps": "int=3"}, "risky": False, "description": "Ask a bounded read-only child agent to investigate."},
}


def build_tool_registry(agent) -> dict:
    return {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], agent)}
        for name, spec in BASE_TOOL_SPECS.items()
        if name != "delegate" or agent.depth < agent.max_depth
    }


def tool_signature(tools: dict) -> str:
    return json.dumps({name: spec["schema"] for name, spec in sorted(tools.items())}, sort_keys=True)


def validate_tool(agent, name: str, args: dict | None) -> None:
    args = args or {}
    if name not in agent.tools:
        raise ValueError(f"unknown tool: {name}")
    if name == "list_files":
        if not agent.path(args.get("path", ".")).is_dir():
            raise ValueError("path is not a directory")
    elif name == "read_file":
        if not agent.path(args["path"]).is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
    elif name == "search":
        if not str(args.get("pattern", "")).strip():
            raise ValueError("pattern must not be empty")
        agent.path(args.get("path", "."))
    elif name == "read_memory":
        _memory_path(agent, args)
    elif name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
    elif name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
    elif name == "patch_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        count = path.read_text(encoding="utf-8").count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
    elif name == "delegate":
        if agent.depth >= agent.max_depth:
            raise ValueError("delegate depth exceeded")
        if not str(args.get("task", "")).strip():
            raise ValueError("task must not be empty")


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
    "delegate": tool_delegate,
}

