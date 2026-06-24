from __future__ import annotations

import json
from dataclasses import dataclass


SECTION_ORDER = (
    "identity",
    "workspace",
    "tools",
    "task_state",
    "working_memory",
    "relevant_memory",
    "memory_index",
    "history",
    "current_request",
)
TOOL_METADATA_PROMPT_FIELDS = (
    "tool_status",
    "affected_paths",
    "content_chars",
    "truncated",
    "artifact_ref",
    "stop_reason",
)
TRUNCATED_TOOL_OBSERVATION_NOTE = (
    "[Tool observation was truncated for session history. Re-run read_file/search "
    "with a narrower range if exact missing content is needed. artifact_ref is for audit, not resume context.]"
)
SECTION_TRUNCATION_SUFFIX = "\n...[section truncated]"


@dataclass(frozen=True)
class PromptSection:
    name: str
    text: str
    source: str
    raw_chars: int
    budget_chars: int | None = None
    truncated: bool = False
    truncation_reason: str = ""
    preserved: bool = False

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "chars": len(self.text),
            "raw_chars": self.raw_chars,
            "budget_chars": self.budget_chars,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "preserved": self.preserved,
        }


class ContextManager:
    def __init__(self, agent, total_budget: int | None = None, section_budgets: dict[str, int] | None = None):
        self.agent = agent
        self.total_budget = total_budget
        self.section_budgets = dict(section_budgets or {})

    def build(self, user_message: str) -> tuple[str, dict]:
        return self.build_prompt(user_message)

    def build_prompt(self, user_message: str) -> tuple[str, dict]:
        request_text = str(user_message)
        relevant_text, relevant_meta = self.agent.memory.render_relevant_memory(user_message, limit=3)
        section_values = {
            "identity": self._render_identity(),
            "workspace": self._render_workspace(),
            "tools": self._render_tools(),
            "task_state": self._render_task_state(),
            "working_memory": self._render_working_memory(),
            "relevant_memory": relevant_text,
            "memory_index": self._render_memory_index(),
            "history": self._render_history(),
            "current_request": self._render_current_request(request_text),
        }
        sources = {
            "identity": "context_manager.identity",
            "workspace": "workspace.text",
            "tools": "tool_registry.prompt_specs",
            "task_state": "runtime.current_task_state",
            "working_memory": "session.memory.working",
            "relevant_memory": "memory.render_relevant_memory",
            "memory_index": "memory.render_memory_index",
            "history": "session.history",
            "current_request": "user_message",
        }
        sections = [self._section(name, section_values[name], sources[name]) for name in SECTION_ORDER]
        prompt = "\n\n".join(section.text for section in sections)
        metadata = {
            "section_order": list(SECTION_ORDER),
            "sections": {section.name: section.metadata() for section in sections},
            "prompt_chars": len(prompt),
            "total_budget_chars": self.total_budget,
            "over_total_budget": self.total_budget is not None and len(prompt) > int(self.total_budget),
            "relevant_memory": relevant_meta,
            "current_request": {
                "text": request_text,
                "chars": len(request_text),
                "truncated": False,
                "preserved": True,
            },
        }
        return prompt, metadata

    def _section(self, name: str, text: object, source: str) -> PromptSection:
        raw_text = str(text)
        budget = self._section_budget(name)
        if name == "current_request":
            return PromptSection(
                name=name,
                text=raw_text,
                source=source,
                raw_chars=len(raw_text),
                budget_chars=budget,
                truncated=False,
                truncation_reason="",
                preserved=True,
            )
        rendered, truncated = self._apply_section_budget(raw_text, budget)
        return PromptSection(
            name=name,
            text=rendered,
            source=source,
            raw_chars=len(raw_text),
            budget_chars=budget,
            truncated=truncated,
            truncation_reason="section_budget_exceeded" if truncated else "",
        )

    def _section_budget(self, name: str) -> int | None:
        if name not in self.section_budgets:
            return None
        budget = int(self.section_budgets[name])
        return max(0, budget)

    @staticmethod
    def _apply_section_budget(text: str, budget: int | None) -> tuple[str, bool]:
        if budget is None or len(text) <= budget:
            return text, False
        if budget <= len(SECTION_TRUNCATION_SUFFIX):
            return text[:budget], True
        return text[: budget - len(SECTION_TRUNCATION_SUFFIX)] + SECTION_TRUNCATION_SUFFIX, True

    @staticmethod
    def _render_identity() -> str:
        return "\n".join(
            [
                "Identity:",
                "- name: MiniBot",
                "- role: local coding agent",
                "- source_fact_policy: use tools for repository facts; reread source files for exact code facts",
                "- relevant_memory_policy: relevant memory is temporary prompt context, not session state",
                "- response_contract: use <tool>{...}</tool> or <tool>[...]</tool> for tool calls; use <final>...</final> for final answers",
            ]
        )

    def _render_workspace(self) -> str:
        return self.agent.workspace.text()

    def _render_tools(self) -> str:
        lines = ["Tools:"]
        specs = self.agent.tools.prompt_specs()
        if not specs:
            lines.append("- none")
            return "\n".join(lines)
        for name, spec in specs.items():
            lines.append(
                "- "
                + f"{name}: {spec.get('description', '')} "
                + f"schema={self._json_for_prompt(spec.get('schema', {}))} "
                + f"risk_level={spec.get('risk_level', '')} "
                + f"risky={self._json_for_prompt(spec.get('risky', False))}"
            )
        return "\n".join(lines)

    def _render_task_state(self) -> str:
        task_state = getattr(self.agent, "current_task_state", None)
        if task_state is None:
            return "Task state:\n- none"
        snapshot = task_state.to_dict()
        keys = ("run_id", "task_id", "status", "tool_steps", "attempts", "last_tool", "stop_reason")
        lines = ["Task state:"]
        for key in keys:
            value = snapshot.get(key, "")
            lines.append(f"- {key}: {value if value not in ('', None) else '-'}")
        return "\n".join(lines)

    def _render_working_memory(self) -> str:
        state = self.agent.memory.to_dict()
        working = state["working"]
        tools = ", ".join(
            f"{item.get('name', '-')}/{item.get('status', '-')}" for item in working.get("recent_tools", [])
        ) or "-"
        return "\n".join(
            [
                "Working memory:",
                f"- initial_request_summary: {working.get('initial_request_summary') or '-'}",
                f"- task_summary: {working.get('task_summary') or '-'}",
                f"- recent_files: {', '.join(working.get('recent_files', [])) or '-'}",
                f"- recent_tools: {tools}",
                f"- file_access_count: {len(state.get('file_access', {}))}",
                f"- episodic_notes: {len(state.get('episodic_notes', []))}",
                "- source_fact_policy: reread source files for exact code facts; file_access is metadata only",
            ]
        )

    def _render_memory_index(self) -> str:
        return self.agent.memory.render_memory_index()

    def _render_history(self) -> str:
        history = list(self.agent.session.get("history", []))
        if not history:
            return "Transcript:\n- empty"
        lines = ["Transcript:"]
        for item in history[-12:]:
            role = item.get("role", "")
            if role == "tool":
                lines.extend(self._tool_history_lines(item))
            else:
                lines.append(f"[{role}] {item.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _render_current_request(user_message: str) -> str:
        return f"Current user request:\n{user_message}"

    def _tool_history_lines(self, item: dict) -> list[str]:
        metadata = self._tool_metadata_for_prompt(item.get("metadata", {}))
        truncated = bool(metadata.get("truncated"))
        lines = [
            f"[tool:{item.get('name', '')}] args={self._json_for_prompt(item.get('args', {}) or {})}",
        ]
        if metadata:
            lines.append(f"metadata={self._json_for_prompt(metadata)}")
        lines.append("Observation preview:" if truncated else "Observation:")
        lines.append(str(item.get("content", "")))
        if truncated:
            lines.append(TRUNCATED_TOOL_OBSERVATION_NOTE)
        return lines

    @staticmethod
    def _tool_metadata_for_prompt(metadata: object) -> dict:
        if not isinstance(metadata, dict):
            return {}
        return {
            key: metadata[key]
            for key in TOOL_METADATA_PROMPT_FIELDS
            if key in metadata
        }

    @staticmethod
    def _json_for_prompt(value: object) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
