from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    INTAKE = "intake"
    INSPECT = "inspect"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    DEBUG = "debug"
    FINALIZE = "finalize"
    BLOCKED = "blocked"


VERIFY_COMMAND_TOKENS = ("test", "pytest", "unittest", "build", "compile", "mvn", "gradle", "npm test", "pnpm test")


def normalize_phase(value: str | Phase | None, fallback: Phase = Phase.INTAKE) -> Phase:
    if isinstance(value, Phase):
        return value
    try:
        return Phase(str(value or "").strip())
    except ValueError:
        return fallback


def infer_initial_phase(user_request: str) -> Phase:
    text = str(user_request or "").lower()
    if any(word in text for word in ("plan", "design", "spec", "方案", "计划", "设计")):
        return Phase.PLAN
    return Phase.INSPECT


def phase_after_tool(name: str, args: dict | None = None, metadata: dict | None = None) -> Phase:
    args = args or {}
    metadata = metadata or {}
    status = str(metadata.get("tool_status", "")).strip()
    if status in {"error", "partial_success"}:
        return Phase.DEBUG
    if status in {"rejected", "blocked"}:
        return Phase.BLOCKED

    if name in {"write_file", "patch_file"}:
        return Phase.IMPLEMENT
    if name == "run_shell":
        command = str(args.get("command", "")).lower()
        if any(token in command for token in VERIFY_COMMAND_TOKENS):
            return Phase.VERIFY
        return Phase.INSPECT
    if name in {"list_files", "read_file", "search", "read_memory", "delegate"}:
        return Phase.INSPECT
    return Phase.INSPECT


def phase_after_stop(stop_reason: str) -> Phase:
    reason = str(stop_reason or "").lower()
    if not reason or reason == "final_answer_returned":
        return Phase.FINALIZE
    if any(token in reason for token in ("denied", "mismatch", "limit", "error", "blocked", "timeout")):
        return Phase.BLOCKED
    return Phase.FINALIZE

