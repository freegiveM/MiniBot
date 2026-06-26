from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

from .models import FakeModelClient
from .runtime import MiniBot, SessionStore
from .workspace import WorkspaceContext


REPL_BANNER = "MiniBot REPL"
REPL_HELP = "\n".join(
    [
        "Commands:",
        "  /help     Show this help.",
        "  /session  Show the current session id and latest run id.",
        "  /reset    Start a new session in this process.",
        "  /exit     Exit the REPL.",
    ]
)


def run_repl(
    *,
    cwd: str | Path = ".",
    model_client=None,
    approval_policy: str = "ask",
    max_steps: int = 6,
    resume: str | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Run a small line-oriented MiniBot REPL around the existing ask() loop."""

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    root = Path(cwd).resolve()
    if not root.exists() or not root.is_dir():
        print(f"minibot: --cwd does not exist or is not a directory: {root}", file=error_stream)
        return 2
    if max_steps <= 0:
        print("minibot: error: --max-steps must be positive", file=error_stream)
        return 2

    workspace = WorkspaceContext.build(root)
    session_store = SessionStore(Path(workspace.repo_root) / ".minibot" / "sessions")
    model_client = model_client or FakeModelClient(["<final>MiniBot REPL is running.</final>"])
    try:
        agent = _build_agent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps,
            resume=resume,
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        print(f"minibot: could not resume session {resume!r}: {exc}", file=error_stream)
        return 2

    _print_banner(agent, output_stream)
    while True:
        line = _read_line(input_stream, output_stream)
        if line == "":
            _save_session(agent)
            output_stream.write("Bye.\n")
            output_stream.flush()
            return 0

        message = line.strip()
        if not message:
            continue
        if message.startswith("/"):
            command_result = _handle_command(
                message,
                agent=agent,
                model_client=model_client,
                workspace=workspace,
                session_store=session_store,
                approval_policy=approval_policy,
                max_steps=max_steps,
                output_stream=output_stream,
            )
            if command_result == "exit":
                _save_session(agent)
                output_stream.write("Bye.\n")
                output_stream.flush()
                return 0
            if isinstance(command_result, MiniBot):
                agent = command_result
            continue

        final = agent.ask(message)
        output_stream.write(f"{final}\n")
        output_stream.flush()


def _build_agent(
    *,
    model_client,
    workspace: WorkspaceContext,
    session_store: SessionStore,
    approval_policy: str,
    max_steps: int,
    resume: str | None = None,
) -> MiniBot:
    kwargs = {"approval_policy": approval_policy, "max_steps": max_steps}
    if resume:
        return MiniBot.from_session(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session_id=resume,
            **kwargs,
        )
    return MiniBot(
        model_client=model_client,
        workspace=workspace,
        session_store=session_store,
        **kwargs,
    )


def _print_banner(agent: MiniBot, output_stream: TextIO) -> None:
    output_stream.write(f"{REPL_BANNER}\n")
    output_stream.write("Type /help for commands. Use /exit or EOF to quit.\n")
    output_stream.write(_session_line(agent) + "\n")
    output_stream.flush()


def _read_line(input_stream: TextIO, output_stream: TextIO) -> str:
    output_stream.write("minibot> ")
    output_stream.flush()
    return input_stream.readline()


def _handle_command(
    command: str,
    *,
    agent: MiniBot,
    model_client,
    workspace: WorkspaceContext,
    session_store: SessionStore,
    approval_policy: str,
    max_steps: int,
    output_stream: TextIO,
) -> str | MiniBot | None:
    normalized = command.strip().lower()
    if normalized == "/exit":
        return "exit"
    if normalized == "/help":
        output_stream.write(REPL_HELP + "\n")
        output_stream.flush()
        return None
    if normalized == "/session":
        output_stream.write(_session_line(agent) + "\n")
        output_stream.flush()
        return None
    if normalized == "/reset":
        _save_session(agent)
        new_agent = _build_agent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps,
        )
        output_stream.write(f"Session reset. {_session_line(new_agent)}\n")
        output_stream.flush()
        return new_agent

    output_stream.write(f"Unknown command: {command}. Type /help for commands.\n")
    output_stream.flush()
    return None


def _session_line(agent: MiniBot) -> str:
    runs = agent.session.get("runs", {}) if isinstance(agent.session.get("runs"), dict) else {}
    last_run_id = str(runs.get("last_run_id") or "none")
    return f"Session: {agent.session.get('id', '')}  Last run: {last_run_id}"


def _save_session(agent: MiniBot) -> None:
    agent.session_path = agent.session_store.save(agent.session)
