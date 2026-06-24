from __future__ import annotations

import json


SECTION_ORDER = (
    "prefix",
    "workspace",
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


class ContextManager:
    def __init__(self, agent):
        self.agent = agent

    def build(self, user_message: str) -> tuple[str, dict]:
        relevant_text, relevant_meta = self.agent.memory.render_relevant_memory(user_message, limit=3)
        sections = {
            "prefix": str(self.agent.prefix),
            "workspace": self.agent.workspace.text(),
            "working_memory": self.agent.memory.render_memory_text(),
            "relevant_memory": relevant_text,
            "memory_index": self.agent.memory.render_memory_index(),
            "history": self._history_text(),
            "current_request": f"Current user request:\n{user_message}",
        }
        prompt = "\n\n".join(sections[name] for name in SECTION_ORDER).strip()
        metadata = {
            "section_order": list(SECTION_ORDER),
            "sections": {name: {"chars": len(sections[name])} for name in SECTION_ORDER},
            "prompt_chars": len(prompt),
            "relevant_memory": relevant_meta,
            "current_request": {
                "text": str(user_message),
                "chars": len(str(user_message)),
            },
        }
        return prompt, metadata

    def _history_text(self) -> str:
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
