from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"

VALID_STATUSES = frozenset(
    {
        STATUS_RUNNING,
        STATUS_COMPLETED,
        STATUS_STOPPED,
        STATUS_FAILED,
        STATUS_BLOCKED,
    }
)
TERMINAL_STATUSES = frozenset(
    {
        STATUS_COMPLETED,
        STATUS_STOPPED,
        STATUS_FAILED,
        STATUS_BLOCKED,
    }
)

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_TOOL_ERROR = "tool_error"
STOP_REASON_RUNTIME_ERROR = "runtime_error"
STOP_REASON_PROMPT_TOO_LONG = "prompt_too_long"

VALID_STOP_REASONS = frozenset(
    {
        STOP_REASON_FINAL_ANSWER_RETURNED,
        STOP_REASON_STEP_LIMIT_REACHED,
        STOP_REASON_RETRY_LIMIT_REACHED,
        STOP_REASON_MODEL_ERROR,
        STOP_REASON_APPROVAL_DENIED,
        STOP_REASON_TOOL_ERROR,
        STOP_REASON_RUNTIME_ERROR,
        STOP_REASON_PROMPT_TOO_LONG,
    }
)


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _required_text(payload: dict, key: str) -> str:
    if key not in payload:
        raise ValueError(f"missing TaskState field: {key}")
    return _text(payload[key])


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"TaskState field must be a non-negative integer: {key}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TaskState field must be a non-negative integer: {key}") from exc
    if number < 0:
        raise ValueError(f"TaskState field must be a non-negative integer: {key}")
    return number


def _validate_status(value: str) -> None:
    if value not in VALID_STATUSES:
        raise ValueError(f"unknown TaskState status: {value}")


def _validate_stop_reason(value: str, *, allow_empty: bool = True) -> None:
    if allow_empty and not value:
        return
    if value not in VALID_STOP_REASONS:
        raise ValueError(f"unknown TaskState stop_reason: {value}")


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""

    def __post_init__(self) -> None:
        self.run_id = _text(self.run_id)
        self.task_id = _text(self.task_id)
        self.user_request = _text(self.user_request)
        self.status = _text(self.status, STATUS_RUNNING)
        self.tool_steps = _non_negative_int(self.tool_steps, "tool_steps")
        self.attempts = _non_negative_int(self.attempts, "attempts")
        self.last_tool = _text(self.last_tool)
        self.stop_reason = _text(self.stop_reason)
        self.final_answer = _text(self.final_answer)
        _validate_status(self.status)
        _validate_stop_reason(self.stop_reason)

    @classmethod
    def create(cls, task_id: str, user_request: str, run_id: str = "") -> "TaskState":
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskState":
        if not isinstance(payload, dict):
            raise TypeError("TaskState.from_dict() requires a dictionary payload")
        return cls(
            run_id=_required_text(payload, "run_id"),
            task_id=_required_text(payload, "task_id"),
            user_request=_required_text(payload, "user_request"),
            status=_text(payload.get("status"), STATUS_RUNNING),
            tool_steps=_non_negative_int(payload.get("tool_steps", 0), "tool_steps"),
            attempts=_non_negative_int(payload.get("attempts", 0), "attempts"),
            last_tool=_text(payload.get("last_tool")),
            stop_reason=_text(payload.get("stop_reason")),
            final_answer=_text(payload.get("final_answer")),
        )

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_tool(self, name: str) -> None:
        self.tool_steps += 1
        self.last_tool = str(name or "")

    def finish_success(self, final_answer: str) -> None:
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)

    def stop(self, reason: str, status: str = STATUS_STOPPED, final_answer: str = "") -> None:
        status = _text(status, STATUS_STOPPED)
        reason = _text(reason)
        _validate_status(status)
        _validate_stop_reason(reason, allow_empty=False)
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"TaskState.stop() requires a terminal status, got: {status}")
        self.status = status
        self.stop_reason = reason
        self.final_answer = _text(final_answer)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
        }
