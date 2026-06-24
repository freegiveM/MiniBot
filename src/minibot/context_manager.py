from __future__ import annotations


SECTION_ORDER = (
    "prefix",
    "workspace",
    "working_memory",
    "hot_memory",
    "relevant_memory",
    "memory_index",
    "history",
    "current_request",
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
            "hot_memory": "Hot memory:\n- none",
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
            "current_phase": self.agent.memory.working.get("current_phase", ""),
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
                lines.append(f"[tool:{item.get('name', '')}] {item.get('args', {})}")
                lines.append(str(item.get("content", "")))
            else:
                lines.append(f"[{role}] {item.get('content', '')}")
        return "\n".join(lines)

