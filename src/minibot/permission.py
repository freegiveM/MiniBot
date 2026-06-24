from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


ACTION_ALLOW = "allow"
ACTION_DENY = "deny"
ACTION_ASK = "ask"

POLICY_AUTO = "auto"
POLICY_ASK = "ask"
POLICY_DENY_RISKY = "deny_risky"
POLICY_NEVER = "never"

RISK_LEVEL_SAFE = "safe"
RISK_LEVEL_RISKY = "risky"

PATH_ARG_TOOLS = frozenset({"list_files", "read_file", "search", "write_file", "patch_file"})
WRITE_TOOLS = frozenset({"write_file", "patch_file"})


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    args: dict = field(default_factory=dict)
    risk_level: str = RISK_LEVEL_SAFE
    workspace_root: str | Path = "."
    read_only: bool = False


@dataclass(frozen=True)
class PermissionDecision:
    action: str
    reason: str
    message: str = ""
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "permission_action": self.action,
            "permission_reason": self.reason,
            **self.metadata,
        }


class PermissionPipeline:
    def __init__(self, workspace_root: str | Path, approval_policy: str = POLICY_ASK, read_only: bool = False):
        self.workspace_root = Path(workspace_root).resolve()
        self.approval_policy = _normalize_policy(approval_policy)
        self.read_only = bool(read_only)

    def check(self, request: PermissionRequest) -> PermissionDecision:
        request = PermissionRequest(
            tool_name=str(request.tool_name),
            args=dict(request.args or {}),
            risk_level=str(request.risk_level or RISK_LEVEL_SAFE),
            workspace_root=request.workspace_root or self.workspace_root,
            read_only=self.read_only or bool(request.read_only),
        )
        path_decision = self._check_path(request)
        if path_decision.action == ACTION_DENY:
            return path_decision

        if request.read_only and (request.risk_level != RISK_LEVEL_SAFE or request.tool_name in WRITE_TOOLS):
            return _deny("read_only", f"{request.tool_name} is blocked in read-only mode")

        if request.tool_name == "run_shell":
            return self._check_shell(request)

        if request.risk_level != RISK_LEVEL_SAFE:
            return self._risk_decision(request, reason="risky_tool")

        return _allow("safe_tool")

    def _check_path(self, request: PermissionRequest) -> PermissionDecision:
        if request.tool_name not in PATH_ARG_TOOLS or "path" not in request.args:
            return _allow("no_path_arg")
        raw_path = request.args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return _allow("path_schema_deferred")
        candidate = Path(raw_path)
        resolved = (candidate if candidate.is_absolute() else self.workspace_root / candidate).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return _deny(
                "path_escape",
                f"{request.tool_name} path escapes workspace",
                {"path": raw_path, "workspace_root": str(self.workspace_root)},
            )
        return _allow("path_within_workspace", {"path": str(resolved)})

    def _check_shell(self, request: PermissionRequest) -> PermissionDecision:
        command = str(request.args.get("command", "")).strip()
        if not command:
            return _allow("shell_schema_deferred")
        if _is_safe_shell_command(command):
            return _allow("safe_shell_command", {"command_category": _shell_command_category(command)})
        return self._risk_decision(request, reason="risky_shell_command")

    def _risk_decision(self, request: PermissionRequest, reason: str) -> PermissionDecision:
        if self.approval_policy == POLICY_AUTO:
            return _allow(reason, {"approval_policy": self.approval_policy})
        if self.approval_policy == POLICY_DENY_RISKY:
            return _deny(reason, f"{request.tool_name} requires approval but policy denies risky tools")
        return PermissionDecision(
            action=ACTION_ASK,
            reason=reason,
            message=f"{request.tool_name} requires approval",
            metadata={"approval_policy": self.approval_policy},
        )


def _normalize_policy(policy: str) -> str:
    value = str(policy or POLICY_ASK).strip().lower()
    if value == POLICY_NEVER:
        return POLICY_DENY_RISKY
    if value not in {POLICY_AUTO, POLICY_ASK, POLICY_DENY_RISKY}:
        raise ValueError(f"unknown approval policy: {policy}")
    return value


def _allow(reason: str, metadata: dict | None = None) -> PermissionDecision:
    return PermissionDecision(action=ACTION_ALLOW, reason=reason, metadata=metadata or {})


def _deny(reason: str, message: str, metadata: dict | None = None) -> PermissionDecision:
    return PermissionDecision(action=ACTION_DENY, reason=reason, message=message, metadata=metadata or {})


def _shell_command_category(command: str) -> str:
    text = command.strip().lower()
    if _matches_any(text, (r"^python\s+-m\s+unittest(\s|$)", r"^pytest(\s|$)")):
        return "test"
    if _matches_any(text, (r"^npm\s+(run\s+)?test(\s|$)", r"^pnpm\s+test(\s|$)", r"^yarn\s+test(\s|$)")):
        return "test"
    if _matches_any(text, (r"^cargo\s+test(\s|$)", r"^go\s+test(\s|$)", r"^dotnet\s+test(\s|$)")):
        return "test"
    if _matches_any(text, (r"^python\s+-m\s+compileall(\s|$)", r"^npm\s+run\s+build(\s|$)", r"^pnpm\s+build(\s|$)", r"^yarn\s+build(\s|$)")):
        return "build"
    if _matches_any(text, (r"^git\s+(status|diff|log|show|branch)(\s|$)", r"^rg(\s|$)", r"^dir(\s|$)", r"^ls(\s|$)")):
        return "read_only"
    return "risky"


def _is_safe_shell_command(command: str) -> bool:
    return _shell_command_category(command) != "risky"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
