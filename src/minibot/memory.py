from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .phase import Phase, normalize_phase
from .workspace import clip, now


WORKING_FILE_LIMIT = 8
RECENT_TOOL_LIMIT = 8
EPISODIC_NOTE_LIMIT = 20
DOMAIN_TERMS = ("memory", "retrieval", "delegate", "checkpoint", "phase", "test", "debug", "记忆", "召回", "工具", "恢复", "测评")


def default_memory_state() -> dict:
    return {
        "working": {
            "initial_request_summary": "",
            "task_summary": "",
            "current_phase": Phase.INTAKE.value,
            "phase_reason": "",
            "constraints": [],
            "recent_files": [],
            "recent_tools": [],
            "open_questions": [],
        },
        "file_access": {},
        "episodic_notes": [],
        "durable_cache": {
            "index_hash": "",
            "loaded_at": "",
            "hot_memory_ids": [],
        },
    }


def resolve_workspace_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> Path | None:
    path = Path(str(raw_path))
    if workspace_root is None:
        return path
    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path: str | Path, workspace_root: str | Path | None = None) -> str:
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    return resolved.relative_to(Path(workspace_root).resolve()).as_posix()


def file_freshness(raw_path: str | Path, workspace_root: str | Path | None = None) -> str | None:
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _ensure_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _tokens(text: object) -> set[str]:
    raw = str(text)
    found = {token.lower() for token in re.findall(r"[A-Za-z0-9_./\\-]+", raw)}
    found |= {term for term in DOMAIN_TERMS if term in raw}
    return found


def extract_symbols(text: object, limit: int = 12) -> list[str]:
    symbols = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", str(text)):
        if token.lower() in {"return", "import", "from", "class", "def", "for", "while", "with"}:
            continue
        symbols.append(token)
    return _dedupe(symbols)[:limit]


def normalize_memory_state(state: dict | None, workspace_root: str | Path | None = None) -> dict:
    if not isinstance(state, dict):
        state = default_memory_state()

    default = default_memory_state()
    working = state.get("working") if isinstance(state.get("working"), dict) else {}
    merged_working = {**default["working"], **working}
    merged_working["task_summary"] = clip(str(merged_working.get("task_summary", "")).strip(), 300)
    merged_working["initial_request_summary"] = clip(str(merged_working.get("initial_request_summary", "")).strip(), 300)
    merged_working["current_phase"] = normalize_phase(merged_working.get("current_phase")).value
    merged_working["constraints"] = [str(item).strip() for item in _ensure_list(merged_working.get("constraints")) if str(item).strip()]
    merged_working["open_questions"] = [str(item).strip() for item in _ensure_list(merged_working.get("open_questions")) if str(item).strip()]
    merged_working["recent_files"] = _dedupe(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(merged_working.get("recent_files"))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    recent_tools = []
    for item in _ensure_list(merged_working.get("recent_tools")):
        if isinstance(item, dict):
            recent_tools.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "status": str(item.get("status", "")).strip(),
                    "path": str(item.get("path", "")).strip(),
                }
            )
        elif str(item).strip():
            recent_tools.append({"name": str(item).strip(), "status": "", "path": ""})
    merged_working["recent_tools"] = recent_tools[-RECENT_TOOL_LIMIT:]
    state["working"] = merged_working

    file_access = state.get("file_access") if isinstance(state.get("file_access"), dict) else {}
    normalized_access = {}
    for path, item in file_access.items():
        if not isinstance(item, dict):
            continue
        canonical = canonicalize_path(path, workspace_root)
        ranges = []
        for range_item in _ensure_list(item.get("read_ranges")):
            if isinstance(range_item, (list, tuple)) and len(range_item) == 2:
                ranges.append([int(range_item[0]), int(range_item[1])])
        normalized_access[canonical] = {
            "last_read_at": str(item.get("last_read_at", "")).strip(),
            "trace_ref": str(item.get("trace_ref", "")).strip(),
            "freshness_hash": str(item.get("freshness_hash", "") or "").strip(),
            "read_ranges": ranges[-8:],
            "symbols_seen": _dedupe([str(symbol).strip() for symbol in _ensure_list(item.get("symbols_seen")) if str(symbol).strip()])[:24],
            "status": str(item.get("status", "valid")).strip() or "valid",
        }
    state["file_access"] = normalized_access

    notes = []
    for index, note in enumerate(_ensure_list(state.get("episodic_notes"))):
        if isinstance(note, dict):
            text = clip(str(note.get("text", "")).strip(), 500)
            if not text:
                continue
            notes.append(
                {
                    "id": str(note.get("id", f"note_{index:04d}")).strip(),
                    "text": text,
                    "tags": _dedupe([str(tag).strip() for tag in _ensure_list(note.get("tags")) if str(tag).strip()]),
                    "topic": str(note.get("topic", "")).strip(),
                    "source_type": str(note.get("source_type", "")).strip(),
                    "source_ref": str(note.get("source_ref", "")).strip(),
                    "tier": str(note.get("tier", "warm")).strip() or "warm",
                    "created_at": str(note.get("created_at", "")).strip() or now(),
                    "last_used_at": str(note.get("last_used_at", "")).strip(),
                    "hit_count": int(note.get("hit_count", 0)),
                    "confidence": str(note.get("confidence", "medium")).strip() or "medium",
                }
            )
        elif str(note).strip():
            notes.append(
                {
                    "id": f"note_{index:04d}",
                    "text": clip(str(note).strip(), 500),
                    "tags": [],
                    "topic": "",
                    "source_type": "",
                    "source_ref": "",
                    "tier": "warm",
                    "created_at": now(),
                    "last_used_at": "",
                    "hit_count": 0,
                    "confidence": "medium",
                }
            )
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    durable_cache = state.get("durable_cache") if isinstance(state.get("durable_cache"), dict) else {}
    state["durable_cache"] = {**default["durable_cache"], **durable_cache}
    return state


class LayeredMemory:
    def __init__(self, state: dict | None = None, workspace_root: str | Path | None = None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)

    def to_dict(self) -> dict:
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    @property
    def working(self) -> dict:
        return self.to_dict()["working"]

    def canonical_path(self, path: str | Path) -> str:
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary: str) -> None:
        self.working["task_summary"] = clip(str(summary).strip(), 300)
        if not self.working.get("initial_request_summary"):
            self.working["initial_request_summary"] = self.working["task_summary"]

    def set_phase(self, phase: str | Phase, reason: str = "") -> None:
        self.working["current_phase"] = normalize_phase(phase).value
        self.working["phase_reason"] = clip(reason, 180)

    def remember_file(self, path: str | Path) -> None:
        canonical = self.canonical_path(path)
        files = [item for item in self.working["recent_files"] if item != canonical]
        files.append(canonical)
        self.working["recent_files"] = files[-WORKING_FILE_LIMIT:]

    def remember_tool(self, name: str, status: str = "", path: str = "") -> None:
        item = {"name": str(name), "status": str(status), "path": str(path)}
        tools = [entry for entry in self.working["recent_tools"] if entry != item]
        tools.append(item)
        self.working["recent_tools"] = tools[-RECENT_TOOL_LIMIT:]

    def record_file_access(
        self,
        path: str | Path,
        start: int = 1,
        end: int = 200,
        trace_ref: str = "",
        symbols: list[str] | None = None,
    ) -> None:
        canonical = self.canonical_path(path)
        access = self.to_dict()["file_access"].setdefault(
            canonical,
            {
                "last_read_at": "",
                "trace_ref": "",
                "freshness_hash": "",
                "read_ranges": [],
                "symbols_seen": [],
                "status": "valid",
            },
        )
        access["last_read_at"] = now()
        access["trace_ref"] = trace_ref
        access["freshness_hash"] = file_freshness(canonical, self.workspace_root) or ""
        read_range = [int(start), int(end)]
        if read_range not in access["read_ranges"]:
            access["read_ranges"].append(read_range)
        access["read_ranges"] = access["read_ranges"][-8:]
        access["symbols_seen"] = _dedupe([*access.get("symbols_seen", []), *(symbols or [])])[:24]
        access["status"] = "valid"
        self.remember_file(canonical)

    def mark_file_stale(self, path: str | Path) -> None:
        canonical = self.canonical_path(path)
        item = self.to_dict()["file_access"].get(canonical)
        if item:
            item["status"] = "stale"

    def append_note(self, text: str, tags: list[str] | tuple[str, ...] = (), topic: str = "", source_type: str = "", source_ref: str = "") -> None:
        text = clip(str(text).strip(), 500)
        if not text:
            return
        note_id = "note_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        notes = [note for note in self.to_dict()["episodic_notes"] if note.get("id") != note_id]
        notes.append(
            {
                "id": note_id,
                "text": text,
                "tags": _dedupe([str(tag).strip() for tag in tags if str(tag).strip()]),
                "topic": str(topic).strip(),
                "source_type": str(source_type).strip(),
                "source_ref": str(source_ref).strip(),
                "tier": "warm",
                "created_at": now(),
                "last_used_at": "",
                "hit_count": 0,
                "confidence": "medium",
            }
        )
        self.state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]

    def retrieval_context(self, user_message: str) -> dict:
        state = self.to_dict()
        working = state["working"]
        text = " ".join(
            [
                str(user_message),
                working.get("task_summary", ""),
                working.get("current_phase", ""),
                " ".join(working.get("recent_files", [])),
                " ".join(tool.get("name", "") for tool in working.get("recent_tools", [])),
                " ".join(state.get("file_access", {}).keys()),
                " ".join(symbol for item in state.get("file_access", {}).values() for symbol in item.get("symbols_seen", [])),
            ]
        )
        keywords = sorted(_tokens(text))[:20]
        return {
            "current_user_request": str(user_message),
            "task_summary": working.get("task_summary", ""),
            "current_phase": working.get("current_phase", Phase.INTAKE.value),
            "recent_files": list(working.get("recent_files", [])),
            "recent_tools": list(working.get("recent_tools", [])),
            "keywords": keywords,
        }

    def retrieval_candidates(self, user_message: str, limit: int = 3) -> list[dict]:
        context = self.retrieval_context(user_message)
        query_tokens = set(context["keywords"])
        recent_files = set(context["recent_files"])
        ranked = []
        for note in self.to_dict()["episodic_notes"]:
            note_tokens = _tokens(note.get("text", "")) | _tokens(" ".join(note.get("tags", []))) | _tokens(note.get("topic", ""))
            path_match = int(any(path and path in note.get("text", "") for path in recent_files))
            keyword_overlap = len(query_tokens & note_tokens)
            tag_match = len(query_tokens & {tag.lower() for tag in note.get("tags", [])})
            tier_bonus = {"hot": 2, "warm": 1, "cold": 0}.get(note.get("tier", "warm"), 0)
            score = path_match * 5 + keyword_overlap * 3 + tag_match * 1.5 + tier_bonus
            if score <= 0:
                continue
            ranked.append((score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, note in ranked[:limit]:
            enriched = dict(note)
            enriched["score"] = score
            results.append(enriched)
        return results

    def render_memory_text(self) -> str:
        state = self.to_dict()
        working = state["working"]
        tools = ", ".join(
            f"{item.get('name', '-')}/{item.get('status', '-')}" for item in working.get("recent_tools", [])
        ) or "-"
        return "\n".join(
            [
                "Task state and working memory:",
                f"- initial_request_summary: {working.get('initial_request_summary') or '-'}",
                f"- task_summary: {working.get('task_summary') or '-'}",
                f"- current_phase: {working.get('current_phase') or '-'}",
                f"- phase_reason: {working.get('phase_reason') or '-'}",
                f"- recent_files: {', '.join(working.get('recent_files', [])) or '-'}",
                f"- recent_tools: {tools}",
                f"- file_access_count: {len(state.get('file_access', {}))}",
                f"- episodic_notes: {len(state.get('episodic_notes', []))}",
                "- source_fact_policy: reread source files for exact code facts; file_access is metadata only",
            ]
        )

    def render_relevant_memory(self, user_message: str, limit: int = 3) -> tuple[str, dict]:
        selected = self.retrieval_candidates(user_message, limit=limit)
        lines = ["Relevant memory:"]
        if not selected:
            lines.append("- none")
        else:
            for note in selected:
                lines.append(f"- [{note.get('id')}] {note.get('text')}")
        return "\n".join(lines), {
            "selected_count": len(selected),
            "selected_ids": [note.get("id", "") for note in selected],
            "selected_scores": [note.get("score", 0) for note in selected],
        }

    def render_memory_index(self, max_chars: int = 1200) -> str:
        if self.workspace_root is None:
            return "Memory index:\n- none"
        index_path = Path(self.workspace_root) / ".minibot" / "memory" / "MEMORY.md"
        if not index_path.exists():
            return "Memory index:\n- none"
        return "Memory index:\n" + clip(index_path.read_text(encoding="utf-8", errors="replace"), max_chars)

