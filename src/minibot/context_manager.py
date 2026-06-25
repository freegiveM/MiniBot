from __future__ import annotations

import json
from dataclasses import dataclass, replace


DEFAULT_TOTAL_BUDGET_CHARS = 8000
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
COMPACTABLE_SECTION_ORDER = (
    "history",
    "relevant_memory",
    "memory_index",
    "workspace",
    "working_memory",
)
TOOL_METADATA_PROMPT_FIELDS = (
    "tool_status",
    "affected_paths",
    "content_chars",
    "truncated",
    "artifact_ref",
    "stop_reason",
    "delegate_id",
    "delegate_artifact",
    "delegate_child_run_id",
    "delegate_schema_valid",
)
TOOL_OBSERVATION_BUDGET_CHARS = 480
TRUNCATED_TOOL_OBSERVATION_NOTE = (
    "[Tool observation was truncated for session history. Re-run read_file/search "
    "with a narrower range if exact missing content is needed. artifact_ref is for audit, not resume context.]"
)
SECTION_TRUNCATION_SUFFIX = "\n...[section truncated]"
SECTION_COMPACTION_SUFFIX = "\n...[section compacted]"
HISTORY_LIMIT = 12


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
    def __init__(
        self,
        agent,
        total_budget: int | None = DEFAULT_TOTAL_BUDGET_CHARS,
        section_budgets: dict[str, int] | None = None,
        summarizer=None,
    ):
        self.agent = agent
        self.total_budget = total_budget
        self.section_budgets = dict(section_budgets or {})
        self.summarizer = summarizer

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
        raw_prompt = self._join_sections(sections)
        raw_prompt_chars = len(raw_prompt)
        sections, compact_summary = self._compact_sections(sections, raw_prompt_chars)
        prompt = self._join_sections(sections)
        metadata = {
            "section_order": list(SECTION_ORDER),
            "sections": {section.name: section.metadata() for section in sections},
            "raw_prompt_chars": raw_prompt_chars,
            "prompt_chars": len(prompt),
            "total_budget_chars": self._total_budget(),
            "over_total_budget": self._over_total_budget(len(prompt)),
            "compact_trigger": compact_summary["trigger"] if compact_summary else "",
            "budget_reduction_count": len(compact_summary["events"]) if compact_summary else 0,
            "current_request_preserved": True,
            "relevant_memory": relevant_meta,
            "current_request": {
                "text": request_text,
                "chars": len(request_text),
                "truncated": False,
                "preserved": True,
            },
        }
        if compact_summary:
            compact_summary["final_prompt_chars"] = len(prompt)
            compact_summary["reduced_chars"] = max(0, raw_prompt_chars - len(prompt))
            compact_summary["over_total_budget"] = metadata["over_total_budget"]
            metadata["compact_summary"] = compact_summary
        return prompt, metadata

    @staticmethod
    def _join_sections(sections: list[PromptSection]) -> str:
        return "\n\n".join(section.text for section in sections)

    def _over_total_budget(self, prompt_chars: int) -> bool:
        budget = self._total_budget()
        return budget is not None and prompt_chars > budget

    def _total_budget(self) -> int | None:
        if self.total_budget is None:
            return None
        return max(0, int(self.total_budget))

    def _compact_sections(
        self,
        sections: list[PromptSection],
        raw_prompt_chars: int,
    ) -> tuple[list[PromptSection], dict | None]:
        budget = self._total_budget()
        if budget is None or raw_prompt_chars <= budget:
            return sections, None

        current_chars = raw_prompt_chars
        compacted = list(sections)
        events = []

        for name in COMPACTABLE_SECTION_ORDER:
            if current_chars <= budget:
                break
            index = self._section_index(compacted, name)
            if index is None:
                continue
            section = compacted[index]
            before_chars = len(section.text)
            if before_chars <= 0:
                continue

            needed_reduction = current_chars - budget
            target_chars = max(0, before_chars - needed_reduction)
            if name == "history":
                next_text, details = self._compact_history(target_chars)
                reason = "history_trimming"
            else:
                reason = "section_budget_compaction"
                next_text = self._compact_section_text(section.text, target_chars, reason)
                details = {}

            after_chars = len(next_text)
            if after_chars >= before_chars:
                continue

            compacted[index] = replace(
                section,
                text=next_text,
                truncated=True,
                truncation_reason=reason,
            )
            current_chars -= before_chars - after_chars
            events.append(
                {
                    "section": name,
                    "reason": reason,
                    "before_chars": before_chars,
                    "after_chars": after_chars,
                    "reduced_chars": before_chars - after_chars,
                    **details,
                }
            )

        return compacted, {
            "trigger": "prompt_budget_exceeded",
            "raw_prompt_chars": raw_prompt_chars,
            "final_prompt_chars": current_chars,
            "reduced_chars": max(0, raw_prompt_chars - current_chars),
            "events": events,
            "current_request_preserved": True,
            "summarizer": {
                "used": False,
                "reason": "deterministic_compaction",
            },
        }

    @staticmethod
    def _section_index(sections: list[PromptSection], name: str) -> int | None:
        for index, section in enumerate(sections):
            if section.name == name:
                return index
        return None

    def _compact_section_text(self, text: str, target_chars: int, reason: str) -> str:
        if len(text) <= target_chars:
            return text

        lines = str(text).splitlines()
        heading = lines[0] if lines else ""
        note = f"[section compacted: {reason}; original_chars={len(text)}]"
        if target_chars <= len(heading):
            return heading
        prefix = "\n".join(item for item in (heading, note) if item)
        if len(prefix) >= target_chars:
            return prefix

        body = "\n".join(lines[1:]).strip()
        suffix = SECTION_COMPACTION_SUFFIX
        body_budget = target_chars - len(prefix) - len(suffix) - 1
        if body_budget <= 0:
            return prefix[:target_chars]
        compacted = prefix + "\n" + body[:body_budget] + suffix
        return compacted[:target_chars]

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
        todo_state = getattr(self.agent, "todo_state", None)
        if todo_state is not None and getattr(todo_state, "items", []):
            lines.extend(todo_state.render_for_prompt().splitlines())
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
        history = self._history_items()
        if not history:
            return "Transcript:\n- empty"
        lines = ["Transcript:"]
        for item in history:
            role = item.get("role", "")
            if role == "tool":
                lines.extend(self._tool_history_lines(item))
            else:
                lines.append(f"[{role}] {item.get('content', '')}")
        return "\n".join(lines)

    def _history_items(self) -> list[dict]:
        return list(self.agent.session.get("history", []))[-HISTORY_LIMIT:]

    def _compact_history(self, target_chars: int) -> tuple[str, dict]:
        history = self._history_items()
        if not history:
            return "Transcript:\n- empty", {
                "trimmed_history_items": 0,
                "collapsed_duplicate_reads": 0,
                "summarized_tool_observations": 0,
            }
        if target_chars <= len("Transcript:"):
            return "Transcript:\n- [context compacted] history omitted to fit prompt budget", {
                "trimmed_history_items": len(history),
                "collapsed_duplicate_reads": 0,
                "summarized_tool_observations": 0,
            }

        latest_read_indexes = self._latest_read_file_indexes(history)
        blocks = []
        collapsed_duplicate_reads = 0
        summarized_tool_observations = 0
        for index, item in enumerate(history):
            role = item.get("role", "")
            if role == "tool":
                duplicate_path = self._duplicate_read_path(item, index, latest_read_indexes)
                if duplicate_path:
                    blocks.append("\n".join(self._collapsed_read_history_lines(item, duplicate_path)))
                    collapsed_duplicate_reads += 1
                    continue
                lines, summarized = self._tool_history_lines(item, compact=True)
                blocks.append("\n".join(lines))
                summarized_tool_observations += int(summarized)
            else:
                blocks.append(f"[{role}] {item.get('content', '')}")

        trimmed_history_items = 0
        text = self._history_text_from_blocks(blocks, trimmed_history_items)
        while target_chars > 0 and len(text) > target_chars and len(blocks) > 1:
            blocks.pop(0)
            trimmed_history_items += 1
            text = self._history_text_from_blocks(blocks, trimmed_history_items)

        if target_chars > 0 and len(text) > target_chars:
            text, _ = self._apply_section_budget(text, target_chars)

        return text, {
            "trimmed_history_items": trimmed_history_items,
            "collapsed_duplicate_reads": collapsed_duplicate_reads,
            "summarized_tool_observations": summarized_tool_observations,
        }

    @staticmethod
    def _history_text_from_blocks(blocks: list[str], trimmed_history_items: int) -> str:
        lines = ["Transcript:"]
        if trimmed_history_items:
            lines.append(f"- [context compacted] trimmed {trimmed_history_items} older history item(s)")
        lines.extend(blocks or ["- empty"])
        return "\n".join(lines)

    def _latest_read_file_indexes(self, history: list[dict]) -> dict[str, int]:
        indexes = {}
        for index, item in enumerate(history):
            if item.get("role") != "tool" or item.get("name") != "read_file":
                continue
            path = self._history_tool_path(item)
            if path:
                indexes[path] = index
        return indexes

    def _duplicate_read_path(self, item: dict, index: int, latest_read_indexes: dict[str, int]) -> str:
        if item.get("role") != "tool" or item.get("name") != "read_file":
            return ""
        path = self._history_tool_path(item)
        if path and latest_read_indexes.get(path) != index:
            return path
        return ""

    @staticmethod
    def _history_tool_path(item: dict) -> str:
        args = item.get("args", {}) or {}
        if not isinstance(args, dict):
            return ""
        return str(args.get("path", "")).strip()

    @staticmethod
    def _render_current_request(user_message: str) -> str:
        return f"Current user request:\n{user_message}"

    def _tool_history_lines(self, item: dict, compact: bool = False) -> list[str] | tuple[list[str], bool]:
        metadata = self._tool_metadata_for_prompt(item.get("metadata", {}))
        truncated = bool(metadata.get("truncated"))
        content = str(item.get("content", ""))
        lines = [
            f"[tool:{item.get('name', '')}] args={self._json_for_prompt(item.get('args', {}) or {})}",
        ]
        if metadata:
            lines.append(f"metadata={self._json_for_prompt(metadata)}")
        if compact:
            should_summarize = truncated or len(content) > TOOL_OBSERVATION_BUDGET_CHARS
            if should_summarize:
                lines.append("Observation summary:")
                lines.extend(self._tool_observation_summary_lines(content, metadata))
                lines.append(TRUNCATED_TOOL_OBSERVATION_NOTE)
                return lines, True
            lines.append("Observation:")
            lines.append(content)
            return lines, False
        lines.append("Observation preview:" if truncated else "Observation:")
        lines.append(content)
        if truncated:
            lines.append(TRUNCATED_TOOL_OBSERVATION_NOTE)
        return lines

    def _collapsed_read_history_lines(self, item: dict, path: str) -> list[str]:
        metadata = self._tool_metadata_for_prompt(item.get("metadata", {}))
        lines = [
            f"[tool:{item.get('name', '')}] args={self._json_for_prompt(item.get('args', {}) or {})}",
        ]
        if metadata:
            lines.append(f"metadata={self._json_for_prompt(metadata)}")
        lines.append("Observation collapsed:")
        lines.append(f"- duplicate read_file collapsed for {path}; newer observation is kept later in transcript")
        artifact_ref = metadata.get("artifact_ref")
        if artifact_ref:
            lines.append(f"- artifact_ref: {artifact_ref}")
        return lines

    def _tool_observation_summary_lines(self, content: str, metadata: dict) -> list[str]:
        preview = self._clip_for_prompt(content, TOOL_OBSERVATION_BUDGET_CHARS)
        lines = [
            f"- preview_chars: {len(preview)}",
            f"- original_chars: {metadata.get('content_chars', len(content))}",
            f"- preview: {preview}",
        ]
        artifact_ref = metadata.get("artifact_ref")
        if artifact_ref:
            lines.append(f"- artifact_ref: {artifact_ref}")
        return lines

    @staticmethod
    def _clip_for_prompt(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        suffix = f"\n...[compacted {len(text) - limit} chars]"
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)] + suffix

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
