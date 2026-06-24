from __future__ import annotations

import argparse
from pathlib import Path

from .models import FakeModelClient
from .runtime import MiniBot, SessionStore
from .workspace import WorkspaceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibot", description="Run the MiniBot local coding agent scaffold.")
    parser.add_argument("message", nargs="*", help="Task message for the agent.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool steps.")
    parser.add_argument("--fake-response", default="<final>MiniBot scaffold is running.</final>", help="Fake model response for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cwd = Path(args.cwd).resolve()
    workspace = WorkspaceContext.build(cwd)
    state_root = Path(workspace.repo_root) / ".minibot"
    session_store = SessionStore(state_root / "sessions")
    model = FakeModelClient([args.fake_response])
    agent = MiniBot(
        model_client=model,
        workspace=workspace,
        session_store=session_store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
    )
    message = " ".join(args.message).strip()
    if not message:
        parser.print_help()
        return 0
    print(agent.ask(message))
    return 0

