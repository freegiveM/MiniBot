from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


PROMPT_CACHE_SCHEMA_VERSION = 1
CACHEABLE_PREFIX_SECTIONS = ("identity", "workspace", "tools", "memory_index")
DYNAMIC_SECTIONS = ("task_state", "working_memory", "relevant_memory", "history", "current_request")

PROMPT_CACHE_AUTO = "auto"
PROMPT_CACHE_OFF = "off"
PROMPT_CACHE_OPENAI_EXPLICIT = "openai_explicit"
PROMPT_CACHE_MODES = (PROMPT_CACHE_AUTO, PROMPT_CACHE_OFF, PROMPT_CACHE_OPENAI_EXPLICIT)

PROMPT_CACHE_RETENTION_IN_MEMORY = "in-memory"
PROMPT_CACHE_RETENTION_24H = "24h"
PROMPT_CACHE_RETENTIONS = (PROMPT_CACHE_RETENTION_IN_MEMORY, PROMPT_CACHE_RETENTION_24H)


@dataclass(frozen=True)
class PromptCachePlan:
    eligible: bool
    provider_mode: str
    stable_prefix_hash: str
    prompt_cache_key: str
    retention: str
    cacheable_sections: tuple[str, ...]
    dynamic_sections: tuple[str, ...]
    invalidation_reasons: tuple[str, ...]
    prefix_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "provider_mode": self.provider_mode,
            "stable_prefix_hash": self.stable_prefix_hash,
            "prompt_cache_key": self.prompt_cache_key,
            "retention": self.retention,
            "cacheable_sections": list(self.cacheable_sections),
            "dynamic_sections": list(self.dynamic_sections),
            "invalidation_reasons": list(self.invalidation_reasons),
            "prefix_chars": self.prefix_chars,
        }

    def kwargs_for_provider(self) -> dict[str, str]:
        if not self.eligible or not self.prompt_cache_key:
            return {}
        return {
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.retention,
        }


def build_prompt_cache_plan(
    *,
    sections: Iterable[Any],
    workspace_fingerprint: str,
    tool_signature: str,
    provider: str,
    api_format: str,
    model_name: str,
    prompt_cache: str = PROMPT_CACHE_AUTO,
    retention: str = PROMPT_CACHE_RETENTION_IN_MEMORY,
) -> PromptCachePlan:
    section_by_name = {str(getattr(section, "name", "")): section for section in sections}
    invalidation_reasons: list[str] = []
    for name in CACHEABLE_PREFIX_SECTIONS:
        if name not in section_by_name:
            invalidation_reasons.append(f"missing_cacheable_section:{name}")

    normalized_cache = normalize_prompt_cache_mode(prompt_cache)
    normalized_retention = normalize_prompt_cache_retention(retention)
    provider_mode = prompt_cache_provider_mode(provider=provider, api_format=api_format, prompt_cache=normalized_cache)
    eligible = provider_mode == PROMPT_CACHE_OPENAI_EXPLICIT and not invalidation_reasons
    if normalized_cache == PROMPT_CACHE_OFF:
        invalidation_reasons.append("prompt_cache_off")
    elif provider_mode != PROMPT_CACHE_OPENAI_EXPLICIT:
        invalidation_reasons.append(f"provider_not_openai_cache_capable:{provider_mode}")

    prefix_chars = sum(len(str(getattr(section_by_name[name], "text", ""))) for name in CACHEABLE_PREFIX_SECTIONS if name in section_by_name)
    payload = {
        "schema_version": PROMPT_CACHE_SCHEMA_VERSION,
        "provider": str(provider or ""),
        "api_format": str(api_format or ""),
        "model_name": str(model_name or ""),
        "sections": {
            "identity_hash": stable_hash(_section_text(section_by_name.get("identity"))),
            "workspace_fingerprint": str(workspace_fingerprint or ""),
            "tools_signature_hash": stable_hash(str(tool_signature or "")),
            "memory_index_hash": stable_hash(_section_text(section_by_name.get("memory_index"))),
        },
    }
    stable_prefix_hash = stable_hash(payload)
    prompt_cache_key = f"minibot:v{PROMPT_CACHE_SCHEMA_VERSION}:{stable_prefix_hash[:32]}"
    return PromptCachePlan(
        eligible=eligible,
        provider_mode=provider_mode,
        stable_prefix_hash=stable_prefix_hash,
        prompt_cache_key=prompt_cache_key,
        retention=normalized_retention,
        cacheable_sections=CACHEABLE_PREFIX_SECTIONS,
        dynamic_sections=DYNAMIC_SECTIONS,
        invalidation_reasons=tuple(invalidation_reasons),
        prefix_chars=prefix_chars,
    )


def stable_hash(value: Any) -> str:
    payload = canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_prompt_cache_mode(value: str | None) -> str:
    mode = str(value or PROMPT_CACHE_AUTO).strip().lower()
    if mode not in PROMPT_CACHE_MODES:
        raise ValueError(f"unsupported prompt_cache mode: {value}")
    return mode


def normalize_prompt_cache_retention(value: str | None) -> str:
    retention = str(value or PROMPT_CACHE_RETENTION_IN_MEMORY).strip().lower()
    if retention not in PROMPT_CACHE_RETENTIONS:
        raise ValueError(f"unsupported prompt_cache_retention: {value}")
    return retention


def prompt_cache_provider_mode(*, provider: str, api_format: str, prompt_cache: str) -> str:
    provider_name = str(provider or "").strip().lower()
    wire_format = str(api_format or "").strip().lower()
    mode = normalize_prompt_cache_mode(prompt_cache)
    if mode == PROMPT_CACHE_OFF:
        return PROMPT_CACHE_OFF
    if wire_format != "openai":
        return f"{wire_format or 'unknown'}_unsupported"
    if mode == PROMPT_CACHE_OPENAI_EXPLICIT:
        return PROMPT_CACHE_OPENAI_EXPLICIT
    if provider_name == "openai":
        return PROMPT_CACHE_OPENAI_EXPLICIT
    return f"{provider_name or 'unknown'}_implicit_only"


def provider_supports_prompt_cache(*, provider: str, api_format: str, prompt_cache: str) -> bool:
    return (
        prompt_cache_provider_mode(provider=provider, api_format=api_format, prompt_cache=prompt_cache)
        == PROMPT_CACHE_OPENAI_EXPLICIT
    )


def _section_text(section: Any) -> str:
    if section is None:
        return ""
    return str(getattr(section, "text", ""))
