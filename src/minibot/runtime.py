from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from . import memory as memorylib
from .context_manager import ContextManager
from .hooks import (
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    EVENT_STOP,
    EVENT_USER_PROMPT_SUBMIT,
    HookManager,
    build_default_hook_manager,
)
from .model_providers import ModelResponse
from .permission import (
    ACTION_ALLOW,
    ACTION_ASK,
    ACTION_DENY,
    PermissionPipeline,
    PermissionRequest,
)
from .recovery import (
    ACTION_RETRY_MODEL,
    RecoveryEvent,
    RecoveryPolicy,
)
from .run_store import RunStore
from .task_state import (
    STATUS_FAILED,
    STOP_REASON_APPROVAL_DENIED,
    STOP_REASON_MODEL_ERROR,
    STOP_REASON_PROMPT_TOO_LONG,
    STOP_REASON_RETRY_LIMIT_REACHED,
    STOP_REASON_STEP_LIMIT_REACHED,
    STOP_REASON_TOOL_ERROR,
    TaskState,
)
from .todo_state import TodoState
from . import tools as toolkit
from .workspace import clip, now


DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "PATH", "PWD", "TMP", "TEMP", "USER")
RECENT_RUN_LIMIT = 10
SESSION_TOOL_OBSERVATION_LIMIT = 1200


class SessionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, session: dict) -> Path:
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, session_id: str) -> dict:
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self) -> str | None:
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class MiniBot:
    def __init__(
        self,
        model_client,
        workspace,
        session_store: SessionStore,
        session: dict | None = None,
        run_store: RunStore | None = None,
        approval_policy: str = "ask",
        max_steps: int = 6,
        max_new_tokens: int = 512,
        depth: int = 0,
        max_depth: int = 1,
        read_only: bool = False,
        shell_env_allowlist: tuple[str, ...] | None = None,
        hook_manager: HookManager | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        strict_action_protocol: bool = True,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root).resolve()
        self.session_store = session_store
        self.run_store = run_store or RunStore(self.root / ".minibot" / "runs")
        self.approval_policy = approval_policy
        self.max_steps = int(max_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.depth = int(depth)
        self.max_depth = int(max_depth)
        self.read_only = bool(read_only)
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.strict_action_protocol = bool(strict_action_protocol)
        self.session = session or self._new_session()
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.todo_state = TodoState.from_dict(self.session.get("todo_state", {}))
        self.hooks = hook_manager or build_default_hook_manager()
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.tools = toolkit.build_tool_registry(self)
        self.permission_pipeline = PermissionPipeline(
            self.root,
            approval_policy=self.approval_policy,
            read_only=self.read_only,
        )
        self.prefix = self.build_prefix()
        self.context_manager = ContextManager(self)
        self.current_task_state: TaskState | None = None
        self.current_run_dir: Path | None = None
        self.last_prompt_metadata: dict = {}
        self._last_tool_result_metadata: dict = {}
        self._current_tool_events: list[dict] = []
        self.hook_results: list[dict] = []
        self.recovery_events: list[dict] = []
        self._model_error_retries = 0
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store: SessionStore, session_id: str, **kwargs) -> "MiniBot":
        return cls(model_client, workspace, session_store, session=session_store.load(session_id), **kwargs)

    def _new_session(self) -> dict:
        return {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "schema_version": 2,
            "created_at": now(),
            "updated_at": now(),
            "workspace_root": str(self.root),
            "runtime_identity": {},
            "turn_count": 0,
            "history": [],
            "memory": memorylib.default_memory_state(),
            "todo_state": {"items": []},
            "runs": {"last_run_id": "", "recent_run_ids": []},
            "memory_maintenance": {
                "pending_store": ".minibot/memory/pending.jsonl",
                "pending_count": 0,
                "last_maintenance_turn": 0,
                "last_promotion_at": "",
            },
            "pending_delegates": [],
            "delegate_artifacts": [],
            "recovery": {
                "workspace_drift_detected": False,
                "last_event": {},
            },
        }

    def _ensure_session_shape(self) -> None:
        default = self._new_session()
        for key, value in default.items():
            self.session.setdefault(key, value)
        self.session["memory"] = memorylib.normalize_memory_state(self.session.get("memory"), self.root)
        self.session["todo_state"] = TodoState.from_dict(self.session.get("todo_state", {})).to_dict()
        self.session.pop("checkpoints", None)
        self.session.setdefault("runs", {"last_run_id": "", "recent_run_ids": []})
        self.session.setdefault("pending_delegates", [])
        self.session.setdefault("delegate_artifacts", [])
        self.session.setdefault("recovery", {"workspace_drift_detected": False, "last_event": {}})

    def build_prefix(self) -> str:
        tool_lines = []
        for name, spec in sorted(self.tools.items()):
            args_schema = json.dumps(spec.schema, sort_keys=True, separators=(",", ":"))
            example_args = json.dumps(spec.example_args, sort_keys=True, separators=(",", ":"))
            tool_lines.append(
                f"- {name}: {spec.description} args_schema={args_schema} example_args={example_args} risky={spec.risky}"
            )
        return "\n".join(
            [
                "You are MiniBot, a local coding agent.",
                "Use tools when repository facts are needed.",
                "For exact source-code facts, read the source file instead of relying on memory.",
                "Relevant memory is temporary prompt context and must not be treated as session state.",
                'Tool protocol: <tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":40}}</tool>',
                'Batch tools use <tool>[{"name":"read_file","args":{"path":"README.md"}},{"name":"search","args":{"pattern":"Status","path":"."}}]</tool>; final answers use <final>...</final>.',
                "Each args object must match that tool's args_schema.",
                'Do not put tool arguments at the top level; always include both "name" and "args".',
                "Tools (schemas describe the args object, not the outer tool call):",
                *tool_lines,
            ]
        )

    def current_runtime_identity(self) -> dict:
        return {
            "repo_root": str(self.root),
            "workspace_fingerprint": self.workspace.fingerprint(),
            "tool_signature": hashlib.sha256(toolkit.tool_signature(self.tools).encode("utf-8")).hexdigest(),
            "read_only": self.read_only,
            "approval_policy": self.approval_policy,
            "model_provider": self.model_client.__class__.__name__,
            "model_name": str(getattr(self.model_client, "model", "") or getattr(getattr(self.model_client, "config", None), "model_name", "")),
        }

    def path(self, raw_path: str | Path) -> Path:
        root = self.root.resolve()
        candidate = Path(raw_path)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes workspace") from exc
        return resolved

    def shell_env(self) -> dict:
        return {key: value for key, value in os.environ.items() if key in self.shell_env_allowlist}

    def record(self, item: dict) -> None:
        self.session.setdefault("history", []).append(item)
        self.session["updated_at"] = now()
        self.session_path = self.session_store.save(self.session)

    def persist_todo_state(self) -> None:
        self.session["todo_state"] = self.todo_state.to_dict()
        self.session["updated_at"] = now()
        self.session_path = self.session_store.save(self.session)

    def emit_trace(self, task_state: TaskState, event: str, payload: dict | None = None) -> None:
        self.run_store.append_trace(
            task_state,
            {
                "event": event,
                "created_at": now(),
                **(payload or {}),
            },
        )

    def _emit_hooks(self, event: str, payload: dict | None = None, task_state: TaskState | None = None):
        task_state = task_state or self.current_task_state
        hook_payload = {
            "agent": self,
            "session_id": self.session.get("id", ""),
            **(payload or {}),
        }
        if task_state is not None:
            hook_payload.setdefault("task_state", task_state.to_dict())
        result = self.hooks.emit(event, hook_payload)
        result_payload = result.to_dict()
        self.hook_results.append(result_payload)
        if task_state is not None:
            self.emit_trace(task_state, "hook_emitted", {"hook_event": event, "hook_result": result_payload})
        return result

    @staticmethod
    def _hook_error_metadata(result) -> dict:
        if not result.errors:
            return {}
        return {"hook_errors": list(result.errors)}

    def _start_task(self, user_message: str) -> TaskState:
        previous_identity = dict(self.session.get("runtime_identity") or {})
        current_identity = self.current_runtime_identity()
        self.memory.set_task_summary(user_message)
        self.session["memory"] = self.memory.to_dict()
        self.session["runtime_identity"] = current_identity
        self.session["turn_count"] = int(self.session.get("turn_count", 0)) + 1
        self.record({"role": "user", "content": str(user_message), "created_at": now()})

        task_state = TaskState.create(task_id="task_" + uuid.uuid4().hex[:8], user_request=str(user_message))
        self.current_task_state = task_state
        self._current_tool_events = []
        self.hook_results = []
        self.recovery_events = []
        self._model_error_retries = 0
        self.current_run_dir = self.run_store.start_run(task_state)
        self._remember_run(task_state.run_id)
        self.emit_trace(task_state, "run_started", {"task_id": task_state.task_id})
        drift = self.recovery_policy.workspace_drift(previous_identity, current_identity)
        if drift is not None:
            self.session.setdefault("recovery", {})["workspace_drift_detected"] = True
            self._record_recovery(drift, task_state)
        self._emit_hooks(EVENT_USER_PROMPT_SUBMIT, {"user_message": str(user_message)}, task_state)
        return task_state

    def _record_recovery(self, event: RecoveryEvent, task_state: TaskState | None = None) -> dict:
        payload = event.to_dict()
        self.recovery_events.append(payload)
        recovery_state = self.session.setdefault("recovery", {})
        recovery_state["last_event"] = payload
        self.session["updated_at"] = now()
        if task_state is not None:
            self.emit_trace(task_state, "recovery_triggered", {"recovery": payload})
        return payload

    def _remember_run(self, run_id: str) -> None:
        runs = self.session.setdefault("runs", {"last_run_id": "", "recent_run_ids": []})
        runs["last_run_id"] = run_id
        recent_ids = [item for item in runs.get("recent_run_ids", []) if item != run_id]
        recent_ids.append(run_id)
        runs["recent_run_ids"] = recent_ids[-RECENT_RUN_LIMIT:]
        self.session_path = self.session_store.save(self.session)

    def _attempt_limit(self) -> int:
        return max(self.max_steps * 3, 4)

    def _record_assistant_decision(self, raw: str, kind: str, payload: object) -> None:
        item = {
            "role": "assistant",
            "content": str(raw or ""),
            "decision": kind,
            "created_at": now(),
        }
        if kind == "retry":
            item["parse_error"] = str(payload)
        self.record(item)

    def _record_tool_observation(self, task_state: TaskState, name: str, args: dict, result: str, metadata: dict) -> None:
        result_text = str(result)
        preview = clip(result_text, SESSION_TOOL_OBSERVATION_LIMIT)
        self.session["memory"] = self.memory.to_dict()
        self.record(
            {
                "role": "tool",
                "name": name,
                "args": args,
                "content": preview,
                "metadata": {
                    "artifact_ref": f"runs/{task_state.run_id}/trace.jsonl",
                    "content_chars": len(result_text),
                    "truncated": preview != result_text,
                    **metadata,
                },
                "created_at": now(),
            }
        )

    def _finish_run_success(self, task_state: TaskState, final: str) -> str:
        self.session["memory"] = self.memory.to_dict()
        self.record({"role": "assistant", "content": final, "created_at": now()})
        task_state.finish_success(final)
        self._emit_hooks(
            EVENT_STOP,
            {"final_answer": final, "tool_events": list(self._current_tool_events)},
            task_state,
        )
        self.session["memory"] = self.memory.to_dict()
        self.session_path = self.session_store.save(self.session)
        self.run_store.write_task_state(task_state)
        self.emit_trace(task_state, "run_finished", {"status": task_state.status, "stop_reason": task_state.stop_reason})
        self.run_store.write_report(task_state, self.build_report(task_state))
        return final

    def _stop_run(self, task_state: TaskState, reason: str, final: str, status: str | None = None) -> str:
        if status is None:
            task_state.stop(reason, final_answer=final)
        else:
            task_state.stop(reason, status=status, final_answer=final)
        self.session["memory"] = self.memory.to_dict()
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self._emit_hooks(
            EVENT_STOP,
            {"final_answer": final, "tool_events": list(self._current_tool_events)},
            task_state,
        )
        self.session["memory"] = self.memory.to_dict()
        self.session_path = self.session_store.save(self.session)
        self.run_store.write_task_state(task_state)
        self.emit_trace(task_state, "run_finished", {"status": task_state.status, "stop_reason": task_state.stop_reason})
        self.run_store.write_report(task_state, self.build_report(task_state))
        return final

    def ask(self, user_message: str) -> str:
        task_state = self._start_task(user_message)

        attempts = 0
        while task_state.tool_steps < self.max_steps and attempts < self._attempt_limit():
            attempts += 1
            task_state.record_attempt()
            prompt, prompt_metadata = self.context_manager.build(user_message)
            self.last_prompt_metadata = prompt_metadata
            self.emit_trace(task_state, "prompt_built", {"prompt_metadata": prompt_metadata})
            if prompt_metadata.get("compact_summary"):
                self.emit_trace(task_state, "context_compacted", {"compact_summary": prompt_metadata["compact_summary"]})
                self._record_recovery(self.recovery_policy.prompt_compacted(prompt_metadata), task_state)
            if prompt_metadata.get("over_total_budget"):
                if self.recovery_policy.should_stop_for_current_request(prompt_metadata):
                    event = self.recovery_policy.current_request_too_long(prompt_metadata)
                else:
                    event = self.recovery_policy.prompt_still_too_long(prompt_metadata)
                self._record_recovery(event, task_state)
                return self._stop_run(task_state, STOP_REASON_PROMPT_TOO_LONG, event.message)
            cache_kwargs = self._prompt_cache_kwargs(prompt_metadata)
            self.emit_trace(task_state, "model_requested", self._model_requested_metadata(prompt_metadata, cache_kwargs))
            try:
                raw = self.model_client.complete(prompt, self.max_new_tokens, tools=self.tools, **cache_kwargs)
            except Exception as exc:
                self._record_prompt_cache_result(prompt_metadata)
                self._model_error_retries += 1
                event = self.recovery_policy.model_error(exc, self._model_error_retries)
                self._record_recovery(event, task_state)
                self.emit_trace(
                    task_state,
                    "model_error",
                    {"error": str(exc), "recoverable": event.recoverable, "retry_count": self._model_error_retries},
                )
                if event.action == ACTION_RETRY_MODEL and attempts < self._attempt_limit():
                    continue
                final = f"Stopped after model error: {exc}"
                return self._stop_run(task_state, STOP_REASON_MODEL_ERROR, final, status=STATUS_FAILED)
            self._record_prompt_cache_result(prompt_metadata)
            self.emit_trace(
                task_state,
                "model_completed",
                {
                    "prompt_metadata": prompt_metadata,
                    "model": self._model_report(),
                },
            )
            response = self._normalize_model_response(raw)
            raw_text = self._response_record_text(response)
            kind, payload = self.parse_response(response, strict_action_protocol=self.strict_action_protocol)
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "provider_tool_call_count": len(response.tool_calls),
                    "text_protocol_action": kind if not response.tool_calls else "",
                },
            )

            if kind == "tool":
                self._record_assistant_decision(raw_text, kind, payload)
                for call in payload:
                    if task_state.tool_steps >= self.max_steps:
                        break
                    name = call.name
                    args = dict(call.args)
                    task_state.record_tool(name)
                    result = self.run_tool(name, args)
                    metadata = dict(self._last_tool_result_metadata)
                    self._record_tool_observation(task_state, name, args, result, metadata)
                    self.run_store.write_task_state(task_state)
                    tool_event = {
                        "name": name,
                        "args": args,
                        "status": metadata.get("tool_status", ""),
                        "result_chars": len(str(result)),
                        "result_preview": clip(result, 800),
                    }
                    self._current_tool_events.append(tool_event)
                    self.emit_trace(
                        task_state,
                        "tool_executed",
                        {"name": name, "args": args, "result": str(result), "result_chars": len(str(result)), **metadata},
                    )
                continue

            if kind == "retry":
                self._record_assistant_decision(raw_text, kind, payload)
                self.run_store.write_task_state(task_state)
                continue

            final = str(payload or raw_text).strip()
            return self._finish_run_success(task_state, final)

        final = (
            "Stopped after too many malformed model responses."
            if attempts >= self._attempt_limit()
            else "Stopped after reaching the step limit without a final answer."
        )
        reason = STOP_REASON_RETRY_LIMIT_REACHED if attempts >= self._attempt_limit() else STOP_REASON_STEP_LIMIT_REACHED
        return self._stop_run(task_state, reason, final)

    def _prompt_cache_kwargs(self, prompt_metadata: dict) -> dict:
        plan = prompt_metadata.get("prompt_cache", {}) if isinstance(prompt_metadata, dict) else {}
        if not isinstance(plan, dict):
            return {}
        if not bool(getattr(self.model_client, "supports_prompt_cache", False)):
            return {}
        if not bool(plan.get("eligible")):
            return {}
        key = str(plan.get("prompt_cache_key") or "")
        if not key:
            return {}
        return {
            "prompt_cache_key": key,
            "prompt_cache_retention": str(plan.get("retention") or "in-memory"),
        }

    def _model_requested_metadata(self, prompt_metadata: dict, cache_kwargs: dict) -> dict:
        plan = prompt_metadata.get("prompt_cache", {}) if isinstance(prompt_metadata, dict) else {}
        if not isinstance(plan, dict):
            plan = {}
        return {
            "prompt_cache_key": str(plan.get("prompt_cache_key") or ""),
            "prompt_cache_eligible": bool(plan.get("eligible")),
            "provider_mode": str(plan.get("provider_mode") or ""),
            "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
            "prompt_cache_sent": bool(cache_kwargs),
            "prompt_cache_retention": str(plan.get("retention") or ""),
        }

    def _record_prompt_cache_result(self, prompt_metadata: dict) -> None:
        model = self._model_report()
        prompt_cache = prompt_metadata.setdefault("prompt_cache", {})
        if not isinstance(prompt_cache, dict):
            prompt_cache = {}
            prompt_metadata["prompt_cache"] = prompt_cache
        for key in (
            "prompt_cache_supported",
            "prompt_cache_key",
            "prompt_cache_retention",
            "cached_tokens",
            "cache_hit",
        ):
            if key in model:
                prompt_cache[key] = model[key]
                prompt_metadata[key] = model[key]
        self.last_prompt_metadata = prompt_metadata

    @staticmethod
    def _affected_paths(args: dict) -> list[str]:
        path = args.get("path", "")
        return [path] if isinstance(path, str) and path.strip() else []

    def _recover_tool_rejection(
        self,
        name: str,
        args: dict,
        event: RecoveryEvent,
        metadata: dict | None = None,
    ) -> str:
        payload = self._record_recovery(event, self.current_task_state)
        self._last_tool_result_metadata = {
            "tool_status": "rejected",
            "affected_paths": self._affected_paths(args),
            "recovery_kind": event.kind,
            "recovery_action": event.action,
            "recoverable": event.recoverable,
            **(metadata or {}),
        }
        if self.current_task_state is not None:
            self.emit_trace(
                self.current_task_state,
                "tool_rejected",
                {"name": name, "args": dict(args), "recovery": payload, **self._last_tool_result_metadata},
            )
        return event.observation()

    def _stale_write_target(self, name: str, args: dict) -> str:
        if name not in {"write_file", "patch_file"}:
            return ""
        path = str(args.get("path", "")).strip()
        if not path:
            return ""
        canonical = self.memory.canonical_path(path)
        item = self.memory.to_dict().get("file_access", {}).get(canonical, {})
        if isinstance(item, dict) and item.get("status") == "stale":
            return canonical
        return ""

    def run_tool(self, name: str, args: dict) -> str:
        self._last_tool_result_metadata = {"tool_status": "unknown", "affected_paths": []}
        try:
            spec = self.tools.spec(name)
        except Exception as exc:
            event = self.recovery_policy.tool_schema_error(name, args, exc)
            return self._recover_tool_rejection(name, args, event, {"error": str(exc)})

        try:
            pre_hook = self._emit_hooks(
                EVENT_PRE_TOOL_USE,
                {"tool_name": name, "args": dict(args), "risk_level": spec.risk_level},
            )
            pre_hook_metadata = self._hook_error_metadata(pre_hook)
            decision = self.permission_pipeline.check(
                PermissionRequest(
                    tool_name=name,
                    args=args,
                    risk_level=spec.risk_level,
                    workspace_root=self.root,
                    read_only=self.read_only,
                )
            )
            if decision.action == ACTION_DENY:
                event = self.recovery_policy.permission_denied(name, args, decision.reason)
                return self._recover_tool_rejection(
                    name,
                    args,
                    event,
                    {"stop_reason": STOP_REASON_APPROVAL_DENIED, **decision.to_metadata(), **pre_hook_metadata},
                )
            if decision.action == ACTION_ASK and not self.approve(name, args):
                event = self.recovery_policy.permission_denied(name, args, decision.reason or "approval_rejected")
                return self._recover_tool_rejection(
                    name,
                    args,
                    event,
                    {"stop_reason": STOP_REASON_APPROVAL_DENIED, **decision.to_metadata(), **pre_hook_metadata},
                )
            if decision.action not in {ACTION_ALLOW, ACTION_ASK}:
                raise ValueError(f"unknown permission action: {decision.action}")

            stale_path = self._stale_write_target(name, args)
            if stale_path:
                event = self.recovery_policy.stale_file_state(name, stale_path)
                return self._recover_tool_rejection(name, args, event, {**decision.to_metadata(), **pre_hook_metadata})

            try:
                self.tools.validate_tool_call(name, args, self)
            except Exception as exc:
                event = self.recovery_policy.tool_schema_error(name, args, exc)
                return self._recover_tool_rejection(
                    name,
                    args,
                    event,
                    {"error": str(exc), **decision.to_metadata(), **pre_hook_metadata},
                )
            observation = self.tools.dispatch(name, args, self, validate=False)
            metadata = {"tool_status": observation.status, **observation.metadata, **decision.to_metadata()}
            metadata.update(pre_hook_metadata)
            if observation.error:
                metadata["error"] = observation.error
            if observation.status == toolkit.OBSERVATION_ERROR:
                metadata.setdefault("stop_reason", STOP_REASON_TOOL_ERROR)
            post_hook = self._emit_hooks(
                EVENT_POST_TOOL_USE,
                {
                    "tool_name": name,
                    "args": dict(args),
                    "result": observation.content,
                    "observation_status": observation.status,
                    "observation_metadata": dict(observation.metadata),
                },
            )
            metadata.update(post_hook.metadata_updates())
            self._last_tool_result_metadata = metadata
            self.update_memory_after_tool(name, args, observation.content, observation.status)
            return observation.content
        except Exception as exc:
            self._last_tool_result_metadata = {"tool_status": "error", "affected_paths": [], "stop_reason": STOP_REASON_TOOL_ERROR}
            return f"tool error: {exc}"

    def update_memory_after_tool(self, name: str, args: dict, result: str, status: str) -> None:
        path = str(args.get("path", "")).strip()
        self.memory.remember_tool(name, status=status, path=path)
        if name == "read_file" and path:
            self.memory.record_file_access(
                path,
                start=int(args.get("start", 1)),
                end=int(args.get("end", 200)),
                trace_ref=self.current_task_state.run_id if self.current_task_state else "",
                symbols=memorylib.extract_symbols(result),
            )
        elif name in {"write_file", "patch_file"} and path:
            self.memory.remember_file(path)
            self.memory.mark_file_stale(path)
        elif status == "error":
            self.memory.append_note(
                f"{name} error; inspect the failure before retry",
                tags=("process", "error", name),
                topic="debug-notes",
                source_type="tool_error",
                source_ref=self.current_task_state.run_id if self.current_task_state else "",
            )
        self.session["memory"] = self.memory.to_dict()

    def approve(self, name: str, args: dict) -> bool:
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=False)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def build_report(self, task_state: TaskState) -> dict:
        return {
            "task_state": task_state.to_dict(),
            "todo_state": self.todo_state.to_dict(),
            "session_id": self.session.get("id", ""),
            "model": self._model_report(),
            "prompt_metadata": self.last_prompt_metadata,
            "hooks": self._hook_report(),
            "delegate_artifacts": list(self.session.get("delegate_artifacts", [])),
            "recovery": {
                "events": list(self.recovery_events),
                "event_count": len(self.recovery_events),
                "workspace_drift_detected": bool(self.session.get("recovery", {}).get("workspace_drift_detected")),
            },
            "memory": {
                "file_access_count": len(self.memory.to_dict().get("file_access", {})),
                "episodic_note_count": len(self.memory.to_dict().get("episodic_notes", [])),
            },
        }

    def _model_report(self) -> dict:
        metadata = getattr(self.model_client, "last_completion_metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _hook_report(self) -> dict:
        errors = []
        for result in self.hook_results:
            errors.extend(result.get("errors", []))
        return {
            "emissions": list(self.hook_results),
            "error_count": len(errors),
            "errors": errors,
        }

    @staticmethod
    def _normalize_model_response(raw) -> ModelResponse:
        if isinstance(raw, ModelResponse):
            return raw
        return ModelResponse(text=str(raw or ""))

    @staticmethod
    def _response_record_text(response: ModelResponse) -> str:
        text = str(response.text or "").strip()
        if text:
            return text
        if response.tool_calls:
            calls = [
                {"name": call.name, "args": dict(call.args), "id": call.id}
                for call in response.tool_calls
            ]
            return "<provider_tool_calls>" + json.dumps(calls, ensure_ascii=False, sort_keys=True) + "</provider_tool_calls>"
        return ""

    @staticmethod
    def parse_response(response: ModelResponse, *, strict_action_protocol: bool = True) -> tuple[str, object]:
        if response.tool_calls:
            return "tool", toolkit.normalize_tool_calls(
                [{"name": call.name, "args": dict(call.args)} for call in response.tool_calls]
            )
        return MiniBot.parse(response.text, strict_action_protocol=strict_action_protocol)

    @staticmethod
    def parse(raw: str, *, strict_action_protocol: bool = True) -> tuple[str, object]:
        text = str(raw or "").strip()
        tool_match = re.search(r"<tool>(.*?)</tool>", text, re.S)
        if tool_match:
            try:
                payload = json.loads(tool_match.group(1).strip())
                return "tool", toolkit.normalize_tool_calls(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                return "retry", f"invalid tool JSON: {exc}"
        final_match = re.search(r"<final>(.*?)</final>", text, re.S)
        if final_match:
            return "final", final_match.group(1).strip()
        if strict_action_protocol:
            return (
                "retry",
                "missing action tag: use exactly one <tool>...</tool> or <final>...</final> response",
            )
        return "final", text
