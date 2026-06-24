from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import memory as memorylib
from .workspace import now


EVENT_USER_PROMPT_SUBMIT = "UserPromptSubmit"
EVENT_PRE_TOOL_USE = "PreToolUse"
EVENT_POST_TOOL_USE = "PostToolUse"
EVENT_STOP = "Stop"

CORE_HOOK_EVENTS = frozenset(
    {
        EVENT_USER_PROMPT_SUBMIT,
        EVENT_PRE_TOOL_USE,
        EVENT_POST_TOOL_USE,
        EVENT_STOP,
    }
)

HookHandler = Callable[[dict], object]


@dataclass(frozen=True)
class HookEvent:
    name: str
    payload: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.name not in CORE_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {self.name}")
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "created_at", self.created_at or now())

    def to_dict(self) -> dict:
        return {
            "event": self.name,
            "created_at": self.created_at,
            "payload_keys": sorted(str(key) for key in self.payload.keys()),
        }


@dataclass(frozen=True)
class _HookEntry:
    name: str
    handler: HookHandler


@dataclass
class HookResult:
    event: str
    handler_count: int = 0
    outputs: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        self.created_at = self.created_at or now()

    @property
    def ok(self) -> bool:
        return not self.errors

    def metadata_updates(self) -> dict:
        metadata = {}
        for output in self.outputs:
            if isinstance(output.get("metadata"), dict):
                metadata.update(output["metadata"])
        if self.errors:
            metadata["hook_errors"] = list(self.errors)
        return metadata

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "handler_count": self.handler_count,
            "outputs": list(self.outputs),
            "errors": list(self.errors),
            "ok": self.ok,
            "created_at": self.created_at,
        }


class HookManager:
    def __init__(self) -> None:
        self._handlers: dict[str, list[_HookEntry]] = {event: [] for event in CORE_HOOK_EVENTS}

    def register(self, event: str, handler: HookHandler, name: str = "") -> "HookManager":
        if event not in CORE_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event}")
        if not callable(handler):
            raise ValueError(f"hook handler is not callable: {event}")
        handler_name = str(name or getattr(handler, "__name__", "") or "hook_handler").strip()
        self._handlers[event].append(_HookEntry(name=handler_name, handler=handler))
        return self

    def handlers(self, event: str) -> list[str]:
        if event not in CORE_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event}")
        return [entry.name for entry in self._handlers[event]]

    def emit(self, event: str, payload: dict | None = None) -> HookResult:
        payload = dict(payload or {})
        payload.setdefault("event", event)
        payload.setdefault("hook_event", event)
        hook_event = HookEvent(event, payload)
        result = HookResult(event=hook_event.name)

        for entry in self._handlers[hook_event.name]:
            result.handler_count += 1
            try:
                output = entry.handler(dict(hook_event.payload))
            except Exception as exc:
                result.errors.append(
                    {
                        "handler": entry.name,
                        "code": f"{entry.name}_failed",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
                continue
            if output is None:
                continue
            if isinstance(output, dict):
                item = dict(output)
            else:
                item = {"value": str(output)}
            item.setdefault("handler", entry.name)
            result.outputs.append(item)

        return result


def tool_observation_metadata_hook(payload: dict) -> dict | None:
    if payload.get("event") != EVENT_POST_TOOL_USE:
        return None
    result = str(payload.get("result", ""))
    return {
        "metadata": {
            "hook_event": EVENT_POST_TOOL_USE,
            "hook_observation_handler": "tool_observation_metadata",
            "hook_observation_chars": len(result),
        }
    }


def stop_summary_hook(payload: dict) -> dict | None:
    if payload.get("event") != EVENT_STOP:
        return None
    task_state = payload.get("task_state", {}) if isinstance(payload.get("task_state"), dict) else {}
    return {
        "stop_summary": {
            "status": task_state.get("status", ""),
            "stop_reason": task_state.get("stop_reason", ""),
            "tool_steps": task_state.get("tool_steps", 0),
            "attempts": task_state.get("attempts", 0),
        }
    }


def memory_extraction_hook(payload: dict) -> dict | None:
    if payload.get("event") != EVENT_STOP:
        return None
    agent = payload.get("agent")
    if agent is None or not hasattr(agent, "memory"):
        return {"memory_extraction": {"skipped_reason": "missing_agent"}}

    task_state = payload.get("task_state", {}) if isinstance(payload.get("task_state"), dict) else {}
    extraction_payload = memorylib.build_extraction_payload(
        user_message=str(task_state.get("user_request") or payload.get("user_message", "")),
        final_answer=str(payload.get("final_answer", "")),
        history=list(getattr(agent, "session", {}).get("history", [])),
        tool_events=list(payload.get("tool_events", [])),
        task_state=task_state,
        source_ref=str(task_state.get("run_id", "")),
    )
    candidates, metadata = agent.memory.extract_memory_candidates(extraction_payload)
    pending_results = [agent.memory.append_pending_candidate(candidate) for candidate in candidates]

    agent.session["memory"] = agent.memory.to_dict()
    maintenance = agent.session.setdefault("memory_maintenance", {})
    maintenance["pending_count"] = len(agent.memory.store.load_pending())
    maintenance["last_extraction_at"] = now()

    return {
        "memory_extraction": {
            "candidate_count": len(candidates),
            "pending_results": pending_results,
            "metadata": metadata,
        }
    }


def build_default_hook_manager() -> HookManager:
    manager = HookManager()
    manager.register(EVENT_POST_TOOL_USE, tool_observation_metadata_hook, name="tool_observation_metadata")
    manager.register(EVENT_STOP, stop_summary_hook, name="stop_summary")
    manager.register(EVENT_STOP, memory_extraction_hook, name="memory_extraction")
    return manager
