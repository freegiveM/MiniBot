from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_TOOL_ERROR = "tool_error"


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

    @classmethod
    def create(cls, task_id: str, user_request: str, run_id: str = "") -> "TaskState":
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

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
        self.status = status
        self.stop_reason = reason
        self.final_answer = final_answer

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
