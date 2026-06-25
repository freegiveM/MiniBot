from __future__ import annotations

from dataclasses import dataclass, field

from .workspace import clip, now


RECOVERY_TOOL_SCHEMA_ERROR = "tool_schema_error"
RECOVERY_PERMISSION_DENIED = "permission_denied"
RECOVERY_PROMPT_TOO_LONG = "prompt_too_long"
RECOVERY_CURRENT_REQUEST_TOO_LONG = "current_request_too_long"
RECOVERY_TRANSIENT_MODEL_ERROR = "transient_model_error"
RECOVERY_STALE_FILE_STATE = "stale_file_state"
RECOVERY_WORKSPACE_DRIFT = "workspace_drift"

ACTION_RETURN_OBSERVATION = "return_observation"
ACTION_REACTIVE_COMPACT = "reactive_compact"
ACTION_RETRY_MODEL = "retry_model"
ACTION_REANCHOR_PROMPT = "reanchor_prompt"
ACTION_STOP = "stop"


@dataclass(frozen=True)
class RecoveryEvent:
    kind: str
    action: str
    message: str
    recoverable: bool = True
    details: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "action": self.action,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
            "created_at": self.created_at,
        }

    def observation(self) -> str:
        return f"recoverable {self.kind}: {self.message}"


class RecoveryPolicy:
    def __init__(
        self,
        max_model_retries: int = 1,
        current_request_budget_ratio: float = 0.75,
        current_request_absolute_limit: int = 6000,
    ):
        self.max_model_retries = max(0, int(max_model_retries))
        self.current_request_budget_ratio = max(0.1, min(float(current_request_budget_ratio), 1.0))
        self.current_request_absolute_limit = max(1, int(current_request_absolute_limit))

    def tool_schema_error(self, tool_name: str, args: dict, error: object) -> RecoveryEvent:
        return RecoveryEvent(
            kind=RECOVERY_TOOL_SCHEMA_ERROR,
            action=ACTION_RETURN_OBSERVATION,
            message=(
                f"{tool_name} arguments were invalid. Fix the tool call schema or read the "
                "relevant context before retrying."
            ),
            details={"tool_name": str(tool_name), "args": dict(args or {}), "error": clip(str(error), 500)},
        )

    def permission_denied(self, tool_name: str, args: dict, reason: str) -> RecoveryEvent:
        return RecoveryEvent(
            kind=RECOVERY_PERMISSION_DENIED,
            action=ACTION_RETURN_OBSERVATION,
            message=f"tool denied by permission policy: {reason}. Choose a permitted path or tool.",
            details={"tool_name": str(tool_name), "args": dict(args or {}), "permission_reason": str(reason)},
        )

    def prompt_compacted(self, prompt_metadata: dict) -> RecoveryEvent:
        compact_summary = prompt_metadata.get("compact_summary", {}) if isinstance(prompt_metadata, dict) else {}
        return RecoveryEvent(
            kind=RECOVERY_PROMPT_TOO_LONG,
            action=ACTION_REACTIVE_COMPACT,
            message="Prompt exceeded the budget and was compacted in the configured order.",
            details={
                "raw_prompt_chars": compact_summary.get("raw_prompt_chars", prompt_metadata.get("raw_prompt_chars", 0)),
                "prompt_chars": prompt_metadata.get("prompt_chars", 0),
                "total_budget_chars": prompt_metadata.get("total_budget_chars"),
                "events": compact_summary.get("events", []),
            },
        )

    def prompt_still_too_long(self, prompt_metadata: dict) -> RecoveryEvent:
        current = prompt_metadata.get("current_request", {}) if isinstance(prompt_metadata, dict) else {}
        return RecoveryEvent(
            kind=RECOVERY_PROMPT_TOO_LONG,
            action=ACTION_STOP,
            message=(
                "Prompt remains too long after compacting history, relevant memory, and memory index. "
                "Shorten the request or move large content into files for MiniBot to read."
            ),
            recoverable=False,
            details={
                "prompt_chars": prompt_metadata.get("prompt_chars", 0),
                "total_budget_chars": prompt_metadata.get("total_budget_chars"),
                "current_request_chars": current.get("chars", 0),
                "compact_trigger": prompt_metadata.get("compact_trigger", ""),
            },
        )

    def current_request_too_long(self, prompt_metadata: dict) -> RecoveryEvent:
        current = prompt_metadata.get("current_request", {}) if isinstance(prompt_metadata, dict) else {}
        threshold = self.current_request_threshold(prompt_metadata)
        return RecoveryEvent(
            kind=RECOVERY_CURRENT_REQUEST_TOO_LONG,
            action=ACTION_STOP,
            message=(
                "Current user request is too long to fit the prompt budget. Shorten it or provide "
                "large content as a file path for MiniBot to read."
            ),
            recoverable=False,
            details={
                "current_request_chars": current.get("chars", 0),
                "current_request_threshold": threshold,
                "total_budget_chars": prompt_metadata.get("total_budget_chars"),
            },
        )

    def should_stop_for_current_request(self, prompt_metadata: dict) -> bool:
        current = prompt_metadata.get("current_request", {}) if isinstance(prompt_metadata, dict) else {}
        chars = int(current.get("chars", 0) or 0)
        threshold = self.current_request_threshold(prompt_metadata)
        return threshold is not None and chars > threshold

    def current_request_threshold(self, prompt_metadata: dict) -> int | None:
        budget = prompt_metadata.get("total_budget_chars") if isinstance(prompt_metadata, dict) else None
        if budget is None:
            return None
        budget = max(0, int(budget))
        return max(1, min(self.current_request_absolute_limit, int(budget * self.current_request_budget_ratio)))

    def model_error(self, error: object, retry_count: int) -> RecoveryEvent:
        can_retry = retry_count <= self.max_model_retries
        return RecoveryEvent(
            kind=RECOVERY_TRANSIENT_MODEL_ERROR,
            action=ACTION_RETRY_MODEL if can_retry else ACTION_STOP,
            message=(
                "Model call failed; retrying once because the error may be transient."
                if can_retry
                else "Model call failed after the retry budget was exhausted."
            ),
            recoverable=can_retry,
            details={
                "error": clip(str(error), 500),
                "retry_count": retry_count,
                "max_model_retries": self.max_model_retries,
            },
        )

    def stale_file_state(self, tool_name: str, path: str) -> RecoveryEvent:
        return RecoveryEvent(
            kind=RECOVERY_STALE_FILE_STATE,
            action=ACTION_RETURN_OBSERVATION,
            message=f"{path} has stale read metadata. Re-run read_file before modifying it.",
            details={"tool_name": str(tool_name), "path": str(path)},
        )

    def workspace_drift(self, previous_identity: dict, current_identity: dict) -> RecoveryEvent | None:
        if not isinstance(previous_identity, dict) or not previous_identity:
            return None
        changed = {}
        for key in ("repo_root", "workspace_fingerprint", "tool_signature", "read_only", "approval_policy"):
            before = previous_identity.get(key)
            after = current_identity.get(key)
            if before != after:
                changed[key] = {"previous": before, "current": after}
        if not changed:
            return None
        return RecoveryEvent(
            kind=RECOVERY_WORKSPACE_DRIFT,
            action=ACTION_REANCHOR_PROMPT,
            message="Runtime identity changed; using current workspace context and treating old run artifacts as audit-only.",
            details={"changed": changed},
        )
