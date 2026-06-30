from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import clip, now


WORKING_FILE_LIMIT = 8
RECENT_TOOL_LIMIT = 8
EPISODIC_NOTE_LIMIT = 20
RELEVANT_MEMORY_LIMIT = 3
PROJECT_MEMORY_MAX_CHARS = 1200
TOPIC_MEMORY_MAX_CHARS = 1200
PENDING_MEMORY_LIMIT = 200
PENDING_PROMOTION_LIMIT = 5
PENDING_PROMOTION_THRESHOLD = 90
SOURCE_EXPLICIT_USER_INSTRUCTION = "explicit_user_instruction"
SOURCE_TOOL_VERIFIED_FACT = "tool_verified_fact"
SOURCE_TOOL_ERROR_LESSON = "tool_error_lesson"
SOURCE_ASSISTANT_INFERENCE = "assistant_inference"
SOURCE_MEMORY_EXTRACTION = "memory_extraction"
MEMORY_SOURCE_TYPE_WEIGHTS = {
    SOURCE_EXPLICIT_USER_INSTRUCTION: 100,
    SOURCE_TOOL_VERIFIED_FACT: 75,
    SOURCE_TOOL_ERROR_LESSON: 55,
    SOURCE_ASSISTANT_INFERENCE: 20,
    SOURCE_MEMORY_EXTRACTION: 20,
}
CONFIDENCE_SCORES = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.0,
}
MEMORY_TOPICS = (
    "project-context",
    "user-preferences",
    "key-decisions",
    "dependency-facts",
    "task-experience",
    "debug-notes",
)
DOMAIN_TERMS = ("memory", "retrieval", "delegate", "test", "debug", "记忆", "召回", "工具", "恢复", "测评")
INTENT_KEYWORDS = (
    "remember",
    "preference",
    "prefer",
    "always",
    "never",
    "convention",
    "decision",
    "记住",
    "偏好",
    "以后都",
    "以后不要",
    "约定",
    "决定",
)
USER_PREFERENCE_TERMS = ("preference", "prefer", "always", "never", "偏好", "以后都", "以后不要")
SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{12,}",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)


def default_memory_state() -> dict:
    return {
        "working": {
            "initial_request_summary": "",
            "task_summary": "",
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


def _normal_hash(text: object) -> str:
    normalized = " ".join(str(text).strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _safe_topic(topic: object, default: str = "task-experience") -> str:
    value = str(topic or default).strip()
    if value in MEMORY_TOPICS:
        return value
    return default


def _safe_tags(tags: object, limit: int = 8) -> list[str]:
    return _dedupe([str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()])[:limit]


def secret_shaped(text: object) -> bool:
    raw = str(text)
    return any(re.search(pattern, raw, re.I) for pattern in SECRET_PATTERNS)


def normalize_confidence(value: object) -> tuple[str, float]:
    raw = str(value or "").strip().lower()
    if raw in CONFIDENCE_SCORES:
        return raw, CONFIDENCE_SCORES[raw]
    return "unknown", 0.0


def normalize_source_type(value: object) -> str:
    raw = str(value or "").strip()
    if raw == SOURCE_MEMORY_EXTRACTION:
        return SOURCE_ASSISTANT_INFERENCE
    if raw in MEMORY_SOURCE_TYPE_WEIGHTS:
        return raw
    return SOURCE_ASSISTANT_INFERENCE


def source_type_weight(source_type: object) -> int:
    normalized = normalize_source_type(source_type)
    return int(MEMORY_SOURCE_TYPE_WEIGHTS.get(normalized, 0))


def score_pending_candidate(candidate: dict) -> int:
    if not isinstance(candidate, dict):
        return 0
    text = str(candidate.get("text", "")).strip()
    if not text or secret_shaped(text):
        return 0
    source_weight = source_type_weight(candidate.get("source_type", ""))
    _, confidence_score = normalize_confidence(candidate.get("confidence", "unknown"))
    return int(source_weight + confidence_score * 40)


def _display_memory_path(path: Path, workspace_root: str | Path | None = None) -> str:
    if workspace_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path(workspace_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class MemoryCandidate:
    text: str
    topic: str = "task-experience"
    tags: list[str] = field(default_factory=list)
    source_type: str = ""
    source_ref: str = ""
    confidence: str = "medium"
    extraction_method: str = "deterministic"
    needs_review: bool = True
    rejected_reason: str = ""
    created_at: str = ""
    id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = clip(str(self.text).strip(), 500)
        self.topic = _safe_topic(self.topic)
        self.tags = _safe_tags(self.tags)
        self.source_type = normalize_source_type(self.source_type)
        self.source_ref = str(self.source_ref).strip()
        raw_confidence = self.confidence
        self.confidence, _ = normalize_confidence(raw_confidence)
        self.extraction_method = str(self.extraction_method or "deterministic").strip()
        self.rejected_reason = str(self.rejected_reason).strip()
        self.created_at = str(self.created_at).strip() or now()
        self.id = str(self.id).strip() or "mem_" + _normal_hash(f"{self.topic}:{self.text}")
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        raw_confidence_text = str(raw_confidence or "").strip().lower()
        if raw_confidence_text and raw_confidence_text not in CONFIDENCE_SCORES:
            warnings = _safe_tags(self.metadata.get("schema_warnings", []), limit=12)
            if "invalid_confidence" not in warnings:
                warnings.append("invalid_confidence")
            self.metadata["schema_warnings"] = warnings

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "topic": self.topic,
            "tags": list(self.tags),
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "needs_review": self.needs_review,
            "rejected_reason": self.rejected_reason,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "MemoryCandidate":
        if not isinstance(value, dict):
            raise ValueError("memory candidate must be an object")
        text = str(value.get("text", "")).strip()
        if not text:
            raise ValueError("memory candidate text must not be empty")
        return cls(
            id=str(value.get("id", "")).strip(),
            text=text,
            topic=str(value.get("topic", "task-experience")).strip(),
            tags=_safe_tags(value.get("tags", [])),
            source_type=str(value.get("source_type", "")).strip(),
            source_ref=str(value.get("source_ref", "")).strip(),
            confidence=str(value.get("confidence", "unknown")).strip(),
            extraction_method=str(value.get("extraction_method", "deterministic")).strip(),
            needs_review=bool(value.get("needs_review", True)),
            rejected_reason=str(value.get("rejected_reason", "")).strip(),
            created_at=str(value.get("created_at", "")).strip(),
            metadata=value.get("metadata", {}) if isinstance(value.get("metadata", {}), dict) else {},
        )


@dataclass
class MemoryIntentResult:
    should_extract: bool
    topic: str = "task-experience"
    tags: list[str] = field(default_factory=list)
    confidence: str = "medium"
    reason: str = ""
    extraction_method: str = "deterministic"

    def __post_init__(self) -> None:
        self.topic = _safe_topic(self.topic)
        self.tags = _safe_tags(self.tags)
        self.confidence = str(self.confidence or "medium").strip()
        self.reason = str(self.reason).strip()
        self.extraction_method = str(self.extraction_method or "deterministic").strip()

    def to_dict(self) -> dict:
        return {
            "should_extract": self.should_extract,
            "topic": self.topic,
            "tags": list(self.tags),
            "confidence": self.confidence,
            "reason": self.reason,
            "extraction_method": self.extraction_method,
        }


class MemoryStore:
    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
        self.root = self.workspace_root / ".minibot" / "memory" if self.workspace_root is not None else None

    @property
    def project_memory_path(self) -> Path | None:
        return self.root / "MEMORY.md" if self.root is not None else None

    @property
    def topics_dir(self) -> Path | None:
        return self.root / "topics" if self.root is not None else None

    @property
    def pending_path(self) -> Path | None:
        return self.root / "pending.jsonl" if self.root is not None else None

    def ensure_dirs(self) -> None:
        if self.root is None or self.topics_dir is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)

    def load_project_memory(self, max_chars: int = PROJECT_MEMORY_MAX_CHARS) -> str:
        path = self.project_memory_path
        if path is None or not path.exists() or not path.is_file():
            return ""
        return clip(path.read_text(encoding="utf-8", errors="replace"), max(1, int(max_chars)))

    def write_project_memory(self, text: str) -> Path:
        self.ensure_dirs()
        path = self.project_memory_path
        if path is None:
            raise ValueError("memory store has no workspace root")
        path.write_text(str(text), encoding="utf-8")
        return path

    def topic_path(self, topic: str, *, must_exist: bool = True) -> Path:
        if self.root is None or self.topics_dir is None:
            raise ValueError("memory store has no workspace root")
        topic = str(topic or "").strip()
        if topic in {"index", "memory", "MEMORY.md"}:
            path = self.root / "MEMORY.md"
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", topic):
                raise ValueError("invalid memory topic")
            path = self.topics_dir / f"{topic}.md"
        resolved = path.resolve()
        allowed = [self.root.resolve(), self.topics_dir.resolve()]
        if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
            raise ValueError("memory path escapes memory root")
        if must_exist and (not resolved.exists() or not resolved.is_file()):
            raise ValueError("memory topic not found")
        return resolved

    def read_topic(self, topic: str = "index", max_chars: int = 2000) -> str:
        path = self.topic_path(topic, must_exist=True)
        return clip(path.read_text(encoding="utf-8", errors="replace"), max(1, min(int(max_chars), 4000)))

    def topic_documents(self, max_chars: int = TOPIC_MEMORY_MAX_CHARS) -> list[dict]:
        topics_dir = self.topics_dir
        if topics_dir is None or not topics_dir.exists():
            return []
        docs = []
        for path in sorted(topics_dir.glob("*.md"), key=lambda item: item.name.lower()):
            topic = path.stem
            text = clip(path.read_text(encoding="utf-8", errors="replace"), max_chars)
            if not text.strip():
                continue
            docs.append(
                {
                    "id": f"topic:{topic}",
                    "text": text,
                    "topic": topic,
                    "tags": [topic],
                    "source_type": "memory_topic",
                    "source_ref": str(path.relative_to(self.workspace_root)) if self.workspace_root else str(path),
                    "confidence": "high",
                }
            )
        return docs

    def load_pending(self, limit: int | None = None) -> list[dict]:
        path = self.pending_path
        if path is None or not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(MemoryCandidate.from_dict(json.loads(line)).to_dict())
            except (json.JSONDecodeError, ValueError):
                continue
        if limit is None:
            return rows[-PENDING_MEMORY_LIMIT:]
        limit_count = max(0, int(limit))
        if limit_count == 0:
            return []
        return rows[-limit_count:]

    def append_pending(self, candidate: MemoryCandidate | dict) -> dict:
        self.ensure_dirs()
        item = candidate if isinstance(candidate, MemoryCandidate) else MemoryCandidate.from_dict(candidate)
        if secret_shaped(item.text):
            return {
                "appended": False,
                "duplicate": False,
                "rejected": True,
                "rejection_reason": "secret_shaped",
                "id": item.id,
            }
        existing_ids = {row.get("id", "") for row in self.load_pending()}
        if item.id in existing_ids:
            return {"appended": False, "duplicate": True, "rejected": False, "id": item.id}
        path = self.pending_path
        if path is None:
            raise ValueError("memory store has no workspace root")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        return {"appended": True, "duplicate": False, "rejected": False, "id": item.id}

    def select_promotable_pending(
        self,
        limit: int = PENDING_PROMOTION_LIMIT,
        threshold: int = PENDING_PROMOTION_THRESHOLD,
    ) -> list[dict]:
        limit_count = max(0, int(limit))
        if limit_count == 0:
            return []
        candidates = []
        for row in self.load_pending():
            item = MemoryCandidate.from_dict(row).to_dict()
            if secret_shaped(item.get("text", "")):
                continue
            normalized_confidence, confidence_score = normalize_confidence(item.get("confidence", "unknown"))
            item["confidence"] = normalized_confidence
            item["confidence_score"] = confidence_score
            item["source_type"] = normalize_source_type(item.get("source_type", ""))
            item["promotion_score"] = score_pending_candidate(item)
            if item["promotion_score"] < int(threshold):
                continue
            candidates.append(item)
        candidates.sort(
            key=lambda item: (
                int(item.get("promotion_score", 0)),
                source_type_weight(item.get("source_type", "")),
                str(item.get("created_at", "")),
                str(item.get("id", "")),
            ),
            reverse=True,
        )
        return candidates[:limit_count]

    def promote_pending_candidates(
        self,
        limit: int = PENDING_PROMOTION_LIMIT,
        threshold: int = PENDING_PROMOTION_THRESHOLD,
    ) -> list[dict]:
        promoted = []
        for item in self.select_promotable_pending(limit=limit, threshold=threshold):
            path = self.promote_candidate(item)
            promoted.append(
                {
                    "id": item["id"],
                    "topic": item["topic"],
                    "promotion_score": item["promotion_score"],
                    "path": _display_memory_path(path, self.workspace_root),
                }
            )
        return promoted

    def promote_candidate(self, candidate: MemoryCandidate | dict | str) -> Path:
        rows = self.load_pending()
        if isinstance(candidate, str):
            match = next((row for row in rows if row.get("id") == candidate), None)
            if match is None:
                raise ValueError("memory candidate not found")
            item = MemoryCandidate.from_dict(match)
        else:
            item = candidate if isinstance(candidate, MemoryCandidate) else MemoryCandidate.from_dict(candidate)
        self.ensure_dirs()
        path = self.topic_path(item.topic, must_exist=False)
        existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else f"# {item.topic}\n"
        line = f"\n- {item.text}"
        if line.strip() not in existing:
            path.write_text(existing.rstrip() + line + "\n", encoding="utf-8")
        return path


def build_extraction_payload(
    *,
    user_message: str = "",
    final_answer: str = "",
    history: list[dict] | None = None,
    tool_events: list[dict] | None = None,
    task_state: dict | None = None,
    source_ref: str = "",
) -> dict:
    return {
        "current_user_request": clip(str(user_message), 1000),
        "final_answer": clip(str(final_answer), 1000),
        "history": list(history or [])[-8:],
        "tool_events": list(tool_events or [])[-12:],
        "task_state": dict(task_state or {}),
        "source_ref": str(source_ref).strip(),
    }


def _payload_text(payload: dict) -> str:
    parts = [
        payload.get("current_user_request", ""),
        payload.get("final_answer", ""),
        json.dumps(payload.get("task_state", {}), sort_keys=True, ensure_ascii=False),
    ]
    for item in _ensure_list(payload.get("history", [])):
        if isinstance(item, dict):
            parts.append(str(item.get("content", "")))
    return "\n".join(str(part) for part in parts if str(part).strip())


class DeterministicMemoryIntentDetector:
    def detect(self, payload: dict) -> MemoryIntentResult:
        text = _payload_text(payload)
        lowered = text.lower()
        matched = [term for term in INTENT_KEYWORDS if term.lower() in lowered or term in text]
        if not matched:
            return MemoryIntentResult(
                should_extract=False,
                reason="no_explicit_memory_intent",
                extraction_method="deterministic",
            )
        preference = any(term.lower() in lowered or term in text for term in USER_PREFERENCE_TERMS)
        decision = "decision" in lowered or "决定" in text or "约定" in text
        topic = "user-preferences" if preference else "key-decisions" if decision else "project-context"
        tags = ["user-preference"] if topic == "user-preferences" else ["project-rule"] if topic == "project-context" else ["decision"]
        return MemoryIntentResult(
            should_extract=True,
            topic=topic,
            tags=tags,
            confidence="medium",
            reason="explicit_memory_intent:" + ",".join(matched[:3]),
            extraction_method="deterministic",
        )


class DeterministicMemorySummarizer:
    def summarize(self, payload: dict, intent: MemoryIntentResult, max_chars: int = 500) -> MemoryCandidate | None:
        text = str(payload.get("current_user_request") or payload.get("final_answer") or _payload_text(payload)).strip()
        if not text:
            return None
        return MemoryCandidate(
            text=clip(text, max(1, int(max_chars))),
            topic=intent.topic,
            tags=intent.tags,
            source_type="memory_extraction",
            source_ref=str(payload.get("source_ref", "")).strip(),
            confidence=intent.confidence,
            extraction_method=intent.extraction_method,
            needs_review=True,
            metadata={"intent_reason": intent.reason},
        )


class MiniLLMMemoryIntentDetector:
    def __init__(self, model_client, max_new_tokens: int = 256):
        self.model_client = model_client
        self.max_new_tokens = int(max_new_tokens)

    def detect(self, payload: dict) -> MemoryIntentResult:
        prompt = "\n".join(
            [
                "Decide whether this coding-agent turn contains durable memory worth saving.",
                "Return JSON with should_extract, topic, tags, confidence, reason.",
                "Allowed topics: " + ", ".join(MEMORY_TOPICS),
                "Payload:",
                clip(json.dumps(payload, sort_keys=True, ensure_ascii=False), 4000),
            ]
        )
        try:
            raw = self.model_client.complete(
                prompt,
                self.max_new_tokens,
                purpose="memory_intent",
                response_format="json",
            )
            data = json.loads(str(raw).strip())
            return MemoryIntentResult(
                should_extract=bool(data.get("should_extract", False)),
                topic=str(data.get("topic", "task-experience")),
                tags=_safe_tags(data.get("tags", [])),
                confidence=str(data.get("confidence", "medium")),
                reason=str(data.get("reason", "")),
                extraction_method="mini_llm",
            )
        except Exception as exc:
            return MemoryIntentResult(
                should_extract=False,
                reason=f"mini_llm_intent_failed:{exc}",
                extraction_method="mini_llm",
            )


class MiniLLMMemorySummarizer:
    def __init__(self, model_client, max_new_tokens: int = 256):
        self.model_client = model_client
        self.max_new_tokens = int(max_new_tokens)

    def summarize(self, payload: dict, intent: MemoryIntentResult, max_chars: int = 500) -> MemoryCandidate | None:
        prompt = "\n".join(
            [
                "Summarize this turn into one bounded durable memory candidate.",
                "Return JSON with text, topic, tags, confidence.",
                f"Max text chars: {max_chars}",
                "Intent:",
                json.dumps(intent.to_dict(), sort_keys=True, ensure_ascii=False),
                "Payload:",
                clip(json.dumps(payload, sort_keys=True, ensure_ascii=False), 4000),
            ]
        )
        try:
            raw = self.model_client.complete(
                prompt,
                self.max_new_tokens,
                purpose="memory_summary",
                response_format="json",
            )
            data = json.loads(str(raw).strip())
            text = clip(str(data.get("text", "")).strip(), max(1, int(max_chars)))
            if not text:
                return None
            return MemoryCandidate(
                text=text,
                topic=str(data.get("topic", intent.topic)),
                tags=_safe_tags(data.get("tags", intent.tags)),
                source_type="memory_extraction",
                source_ref=str(payload.get("source_ref", "")).strip(),
                confidence=str(data.get("confidence", intent.confidence)),
                extraction_method="mini_llm",
                needs_review=True,
                metadata={"intent_reason": intent.reason},
            )
        except Exception:
            return None


class MemoryExtractionEngine:
    def __init__(self, intent_detector=None, summarizer=None):
        self.intent_detector = intent_detector or DeterministicMemoryIntentDetector()
        self.summarizer = summarizer or DeterministicMemorySummarizer()

    def extract(self, payload: dict, max_chars: int = 500) -> tuple[list[MemoryCandidate], dict]:
        intent = self.intent_detector.detect(payload)
        metadata = {
            "intent": intent.to_dict(),
            "candidate_count": 0,
            "skipped_reason": "",
        }
        if not intent.should_extract:
            metadata["skipped_reason"] = intent.reason or "intent_not_selected"
            return [], metadata
        candidate = self.summarizer.summarize(payload, intent, max_chars=max_chars)
        if candidate is None:
            metadata["skipped_reason"] = "summary_failed"
            return [], metadata
        metadata["candidate_count"] = 1
        return [candidate], metadata


def normalize_memory_state(state: dict | None, workspace_root: str | Path | None = None) -> dict:
    if not isinstance(state, dict):
        state = default_memory_state()

    default = default_memory_state()
    working = state.get("working") if isinstance(state.get("working"), dict) else {}
    merged_working = {**default["working"], **working}
    merged_working["task_summary"] = clip(str(merged_working.get("task_summary", "")).strip(), 300)
    merged_working["initial_request_summary"] = clip(str(merged_working.get("initial_request_summary", "")).strip(), 300)
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
        self.store = MemoryStore(workspace_root)
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
            "recent_files": list(working.get("recent_files", [])),
            "recent_tools": list(working.get("recent_tools", [])),
            "keywords": keywords,
        }

    def retrieval_candidates(self, user_message: str, limit: int = RELEVANT_MEMORY_LIMIT) -> list[dict]:
        context = self.retrieval_context(user_message)
        query_tokens = set(context["keywords"])
        recent_files = set(context["recent_files"])
        ranked = []
        candidates = []
        for note in self.to_dict()["episodic_notes"]:
            candidates.append(
                {
                    **note,
                    "source_type": note.get("source_type") or "session_note",
                    "source_ref": note.get("source_ref", ""),
                }
            )
        candidates.extend(self.store.topic_documents())
        for note in candidates:
            note_tokens = _tokens(note.get("text", "")) | _tokens(" ".join(note.get("tags", []))) | _tokens(note.get("topic", ""))
            path_match = int(any(path and path in note.get("text", "") for path in recent_files))
            keyword_overlap = len(query_tokens & note_tokens)
            tag_match = len(query_tokens & {tag.lower() for tag in note.get("tags", [])})
            source_bonus = 1 if note.get("source_type") == "memory_topic" else 0
            score = path_match * 5 + keyword_overlap * 3 + tag_match * 1.5 + source_bonus
            if score <= 0:
                continue
            ranked.append((score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        limit = max(0, min(int(limit), RELEVANT_MEMORY_LIMIT))
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
                f"- recent_files: {', '.join(working.get('recent_files', [])) or '-'}",
                f"- recent_tools: {tools}",
                f"- file_access_count: {len(state.get('file_access', {}))}",
                f"- episodic_notes: {len(state.get('episodic_notes', []))}",
                "- source_fact_policy: reread source files for exact code facts; file_access is metadata only",
            ]
        )

    def render_relevant_memory(self, user_message: str, limit: int = RELEVANT_MEMORY_LIMIT) -> tuple[str, dict]:
        limit = max(0, min(int(limit), RELEVANT_MEMORY_LIMIT))
        selected = self.retrieval_candidates(user_message, limit=limit)
        lines = ["Relevant memory:"]
        if not selected:
            lines.append("- none")
        else:
            for note in selected:
                source = note.get("source_type") or "-"
                topic = note.get("topic") or "-"
                lines.append(f"- [{note.get('id')}] topic={topic} source={source} {note.get('text')}")
        return "\n".join(lines), {
            "selector": "deterministic_fallback",
            "fallback_used": True,
            "limit": limit,
            "selected_count": len(selected),
            "selected_ids": [note.get("id", "") for note in selected],
            "selected_scores": [note.get("score", 0) for note in selected],
            "selected_topics": [note.get("topic", "") for note in selected],
            "stable_project_memory_included": False,
        }

    def render_memory_index(self, max_chars: int = PROJECT_MEMORY_MAX_CHARS) -> str:
        project_memory = self.store.load_project_memory(max_chars=max_chars)
        if not project_memory:
            return "Memory index:\n- none"
        return "Memory index:\n" + project_memory

    def extract_memory_candidates(self, payload: dict, engine: MemoryExtractionEngine | None = None, max_chars: int = 500) -> tuple[list[dict], dict]:
        extractor = engine or MemoryExtractionEngine()
        candidates, metadata = extractor.extract(payload, max_chars=max_chars)
        return [candidate.to_dict() for candidate in candidates], metadata

    def append_pending_candidate(self, candidate: MemoryCandidate | dict) -> dict:
        return self.store.append_pending(candidate)
