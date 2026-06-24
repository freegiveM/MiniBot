from __future__ import annotations

import uuid
from dataclasses import dataclass

from .workspace import clip, now


TODO_PENDING = "pending"
TODO_IN_PROGRESS = "in_progress"
TODO_COMPLETED = "completed"
TODO_BLOCKED = "blocked"

VALID_TODO_STATUSES = frozenset(
    {
        TODO_PENDING,
        TODO_IN_PROGRESS,
        TODO_COMPLETED,
        TODO_BLOCKED,
    }
)

TODO_CONTENT_LIMIT = 500


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _required_text(payload: dict, key: str) -> str:
    if key not in payload:
        raise ValueError(f"missing todo item field: {key}")
    value = _text(payload[key]).strip()
    if not value:
        raise ValueError(f"todo item field must not be empty: {key}")
    return value


def _validate_status(status: str) -> None:
    if status not in VALID_TODO_STATUSES:
        raise ValueError(f"unknown todo status: {status}")


@dataclass
class TodoItem:
    id: str
    content: str
    status: str = TODO_PENDING
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.id = _text(self.id).strip()
        if not self.id:
            raise ValueError("todo item id must not be empty")
        self.content = clip(_text(self.content).strip(), TODO_CONTENT_LIMIT)
        if not self.content:
            raise ValueError("todo item content must not be empty")
        self.status = _text(self.status, TODO_PENDING).strip() or TODO_PENDING
        _validate_status(self.status)
        timestamp = now()
        self.created_at = _text(self.created_at).strip() or timestamp
        self.updated_at = _text(self.updated_at).strip() or self.created_at

    @classmethod
    def from_dict(cls, payload: dict) -> "TodoItem":
        if not isinstance(payload, dict):
            raise ValueError("todo item must be an object")
        return cls(
            id=_required_text(payload, "id"),
            content=_required_text(payload, "content"),
            status=_text(payload.get("status"), TODO_PENDING),
            created_at=_text(payload.get("created_at")),
            updated_at=_text(payload.get("updated_at")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TodoState:
    def __init__(self, items: list[TodoItem | dict] | None = None):
        self.items: list[TodoItem] = []
        if items:
            self.set_items(items)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TodoState":
        if payload in (None, ""):
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("TodoState.from_dict() requires a dictionary payload")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError("todo_state.items must be a list")
        return cls(items)

    def to_dict(self) -> dict:
        return {"items": [item.to_dict() for item in self.items]}

    def set_items(self, items: list[TodoItem | dict]) -> None:
        if not isinstance(items, list):
            raise ValueError("todo items must be a list")
        previous = {item.id: item for item in self.items}
        timestamp = now()
        normalized: list[TodoItem] = []
        seen: set[str] = set()
        for raw in items:
            data = raw.to_dict() if isinstance(raw, TodoItem) else raw
            if not isinstance(data, dict):
                raise ValueError("todo item must be an object")
            item_id = _required_text(data, "id")
            if item_id in seen:
                raise ValueError(f"duplicate todo item id: {item_id}")
            seen.add(item_id)

            existing = previous.get(item_id)
            content = _required_text(data, "content")
            status = _text(data.get("status"), TODO_PENDING).strip() or TODO_PENDING
            _validate_status(status)
            created_at = _text(data.get("created_at")).strip()
            if not created_at and existing is not None:
                created_at = existing.created_at
            if not created_at:
                created_at = timestamp

            updated_at = _text(data.get("updated_at")).strip()
            changed = existing is None or existing.content != content or existing.status != status
            if not updated_at and existing is not None and not changed:
                updated_at = existing.updated_at
            if not updated_at:
                updated_at = timestamp

            normalized.append(
                TodoItem(
                    id=item_id,
                    content=content,
                    status=status,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        self._validate_items(normalized)
        self.items = normalized

    def append(self, content: str, status: str = TODO_PENDING, id: str = "") -> TodoItem:
        item_id = str(id or "").strip() or self._new_id()
        item = TodoItem(id=item_id, content=content, status=status)
        self.set_items([*self.items, item])
        return item

    def update(self, id: str, content: str | None = None, status: str | None = None) -> TodoItem:
        item_id = str(id or "").strip()
        if not item_id:
            raise ValueError("todo item id must not be empty")
        next_items = []
        updated_item: TodoItem | None = None
        timestamp = now()
        for item in self.items:
            if item.id != item_id:
                next_items.append(item)
                continue
            next_content = item.content if content is None else str(content).strip()
            next_status = item.status if status is None else str(status).strip()
            changed = next_content != item.content or next_status != item.status
            updated_item = TodoItem(
                id=item.id,
                content=next_content,
                status=next_status,
                created_at=item.created_at,
                updated_at=timestamp if changed else item.updated_at,
            )
            next_items.append(updated_item)
        if updated_item is None:
            raise ValueError(f"todo item not found: {item_id}")
        self._validate_items(next_items)
        self.items = next_items
        return updated_item

    def render_for_prompt(self, max_items: int = 8) -> str:
        lines = ["Todo plan:"]
        if not self.items:
            lines.append("- none")
            return "\n".join(lines)
        limit = max(0, int(max_items))
        visible = self.items[:limit] if limit else []
        for item in visible:
            lines.append(f"- [{item.status}] {item.id}: {item.content}")
        hidden_count = len(self.items) - len(visible)
        if hidden_count > 0:
            lines.append(f"- ... {hidden_count} more todo item(s)")
        return "\n".join(lines)

    def summary(self) -> dict:
        counts = {status: 0 for status in sorted(VALID_TODO_STATUSES)}
        for item in self.items:
            counts[item.status] += 1
        return {
            "total": len(self.items),
            "counts": counts,
            "in_progress_id": next((item.id for item in self.items if item.status == TODO_IN_PROGRESS), ""),
        }

    @staticmethod
    def _validate_items(items: list[TodoItem]) -> None:
        seen: set[str] = set()
        in_progress_count = 0
        for item in items:
            if item.id in seen:
                raise ValueError(f"duplicate todo item id: {item.id}")
            seen.add(item.id)
            _validate_status(item.status)
            if item.status == TODO_IN_PROGRESS:
                in_progress_count += 1
        if in_progress_count > 1:
            raise ValueError("only one todo item may be in_progress")

    def _new_id(self) -> str:
        while True:
            item_id = "todo_" + uuid.uuid4().hex[:8]
            if all(item.id != item_id for item in self.items):
                return item_id
