from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import memory as memorylib
from .model_providers import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    DEFAULT_ANTHROPIC_MESSAGES_URL,
    DEFAULT_OPENAI_CHAT_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_FAKE,
    PROVIDER_HTTP,
    PROVIDER_OPENAI,
    ProviderConfig,
    build_model_client_from_config,
    load_dotenv,
    resolve_provider_config,
)
from .models import FakeModelClient
from .workspace import clip


MEMORY_MODEL_PROVIDER_ENV = "MINIBOT_MEMORY_MODEL_PROVIDER"
MEMORY_API_FORMAT_ENV = "MINIBOT_MEMORY_API_FORMAT"
MEMORY_MODEL_NAME_ENV = "MINIBOT_MEMORY_MODEL_NAME"
MEMORY_BASE_URL_ENV = "MINIBOT_MEMORY_BASE_URL"
MEMORY_API_KEY_ENV = "MINIBOT_MEMORY_API_KEY"
MEMORY_API_KEY_ENV_NAME_ENV = "MINIBOT_MEMORY_API_KEY_ENV"
MEMORY_MODEL_CANDIDATES_ENV = "MINIBOT_MEMORY_MODEL_CANDIDATES"
MEMORY_TIMEOUT_ENV = "MINIBOT_MEMORY_TIMEOUT_SECONDS"

DEFAULT_MEMORY_MODEL_CANDIDATES = ("sonnet", "gpt-mini", "deepseek-v4-pro")
SUMMARY_MAX_CHARS = 500


@dataclass(frozen=True)
class MemoryModelProfile:
    name: str
    provider: str
    api_format: str
    model_name: str
    base_url: str
    api_key_envs: tuple[str, ...]


@dataclass
class MemoryModelSelection:
    model_client: object | None = None
    selected_memory_model: str = ""
    selection_source: str = "deterministic_fallback"
    fallback_chain: list[str] = field(default_factory=list)
    provider_metadata: dict = field(default_factory=dict)

    @property
    def has_model(self) -> bool:
        return self.model_client is not None


PROFILE_ALIASES = {
    "sonnet": MemoryModelProfile(
        name="sonnet",
        provider=PROVIDER_ANTHROPIC,
        api_format=API_FORMAT_ANTHROPIC,
        model_name="claude-3-5-sonnet-latest",
        base_url=DEFAULT_ANTHROPIC_MESSAGES_URL,
        api_key_envs=("MINIBOT_MEMORY_SONNET_API_KEY", "ANTHROPIC_API_KEY"),
    ),
    "gpt-mini": MemoryModelProfile(
        name="gpt-mini",
        provider=PROVIDER_OPENAI,
        api_format=API_FORMAT_OPENAI,
        model_name="gpt-4o-mini",
        base_url=DEFAULT_OPENAI_CHAT_URL,
        api_key_envs=("MINIBOT_MEMORY_GPT_MINI_API_KEY", "OPENAI_API_KEY"),
    ),
    "gptmini": MemoryModelProfile(
        name="gptmini",
        provider=PROVIDER_OPENAI,
        api_format=API_FORMAT_OPENAI,
        model_name="gpt-4o-mini",
        base_url=DEFAULT_OPENAI_CHAT_URL,
        api_key_envs=("MINIBOT_MEMORY_GPTMINI_API_KEY", "OPENAI_API_KEY"),
    ),
    "deepseek-v4-pro": MemoryModelProfile(
        name="deepseek-v4-pro",
        provider=PROVIDER_DEEPSEEK,
        api_format=API_FORMAT_OPENAI,
        model_name="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1/chat/completions",
        api_key_envs=("MINIBOT_MEMORY_DEEPSEEK_V4_PRO_API_KEY", "DEEPSEEK_API_KEY"),
    ),
}


class MemoryModelResolver:
    def __init__(self, cwd: str | Path = ".", env_file: str | Path = ".env", environ: dict[str, str] | None = None):
        self.cwd = Path(cwd).resolve()
        env_file_path = Path(env_file)
        self.env_file = env_file_path if env_file_path.is_absolute() else self.cwd / env_file_path
        dotenv = load_dotenv(self.env_file)
        process_env = dict(os.environ if environ is None else environ)
        self.env = {**dotenv, **process_env}

    def resolve(self, current_model_client=None, explicit_model_client=None) -> MemoryModelSelection:
        chain: list[str] = []
        if explicit_model_client is not None:
            return MemoryModelSelection(
                model_client=explicit_model_client,
                selected_memory_model=_model_name(explicit_model_client),
                selection_source="agent_override",
                fallback_chain=chain,
                provider_metadata=_safe_model_metadata(explicit_model_client),
            )

        explicit = self._explicit_memory_config(chain)
        if explicit is not None:
            return explicit

        profile = self._candidate_profile_config(chain)
        if profile is not None:
            return profile

        main = self._main_model_selection(current_model_client, chain)
        if main is not None:
            return main

        chain.append("deterministic_fallback_selected")
        return MemoryModelSelection(fallback_chain=chain)

    def _explicit_memory_config(self, chain: list[str]) -> MemoryModelSelection | None:
        keys = (
            MEMORY_MODEL_PROVIDER_ENV,
            MEMORY_API_FORMAT_ENV,
            MEMORY_MODEL_NAME_ENV,
            MEMORY_BASE_URL_ENV,
            MEMORY_API_KEY_ENV,
            MEMORY_API_KEY_ENV_NAME_ENV,
        )
        if not any(self.env.get(key) for key in keys):
            chain.append("explicit_memory_config_missing")
            return None
        mapped = {
            "MINIBOT_MODEL_PROVIDER": self.env.get(MEMORY_MODEL_PROVIDER_ENV, PROVIDER_HTTP),
            "MINIBOT_API_FORMAT": self.env.get(MEMORY_API_FORMAT_ENV, API_FORMAT_OPENAI),
            "MINIBOT_MODEL_NAME": self.env.get(MEMORY_MODEL_NAME_ENV, ""),
            "MINIBOT_BASE_URL": self.env.get(MEMORY_BASE_URL_ENV, ""),
            "MINIBOT_API_KEY": self.env.get(MEMORY_API_KEY_ENV, ""),
            "MINIBOT_API_KEY_ENV": self.env.get(MEMORY_API_KEY_ENV_NAME_ENV, MEMORY_API_KEY_ENV),
            "MINIBOT_TIMEOUT_SECONDS": self.env.get(MEMORY_TIMEOUT_ENV, ""),
        }
        api_key_env = mapped["MINIBOT_API_KEY_ENV"]
        if api_key_env and not mapped["MINIBOT_API_KEY"]:
            mapped["MINIBOT_API_KEY"] = self.env.get(api_key_env, "")
        try:
            config = resolve_provider_config(cwd=self.cwd, env_file="__missing__.env", environ=mapped)
        except Exception as exc:
            chain.append(f"explicit_memory_config_unavailable:{exc}")
            return None
        client = build_model_client_from_config(config)
        return MemoryModelSelection(
            model_client=client,
            selected_memory_model=config.model_name,
            selection_source="explicit_memory_config",
            fallback_chain=chain,
            provider_metadata=config.safe_metadata(),
        )

    def _candidate_profile_config(self, chain: list[str]) -> MemoryModelSelection | None:
        for name in self._candidate_names():
            profile = PROFILE_ALIASES.get(name)
            if profile is None:
                chain.append(f"candidate_unknown:{name}")
                continue
            config = self._profile_config(profile)
            if config is None:
                chain.append(f"candidate_unavailable:{name}")
                continue
            client = build_model_client_from_config(config)
            return MemoryModelSelection(
                model_client=client,
                selected_memory_model=config.model_name,
                selection_source=f"candidate_profile:{name}",
                fallback_chain=chain,
                provider_metadata=config.safe_metadata(),
            )
        return None

    def _candidate_names(self) -> list[str]:
        raw = self.env.get(MEMORY_MODEL_CANDIDATES_ENV, ",".join(DEFAULT_MEMORY_MODEL_CANDIDATES))
        names = []
        for item in str(raw).split(","):
            name = item.strip().lower()
            if name:
                names.append(name)
        return names

    def _profile_config(self, profile: MemoryModelProfile) -> ProviderConfig | None:
        prefix = re.sub(r"[^A-Z0-9]+", "_", profile.name.upper()).strip("_")
        model_name = self.env.get(f"MINIBOT_MEMORY_{prefix}_MODEL_NAME", profile.model_name)
        base_url = self.env.get(f"MINIBOT_MEMORY_{prefix}_BASE_URL", profile.base_url)
        api_key = self.env.get(f"MINIBOT_MEMORY_{prefix}_API_KEY", "") or self.env.get(MEMORY_API_KEY_ENV, "")
        for key in profile.api_key_envs:
            api_key = api_key or self.env.get(key, "")
        if not model_name or not base_url or not api_key:
            return None
        return ProviderConfig(
            provider=profile.provider,
            api_format=profile.api_format,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_env=profile.api_key_envs[0],
            timeout_seconds=_timeout(self.env.get(MEMORY_TIMEOUT_ENV, 60.0)),
            env_file=str(self.env_file),
        )

    @staticmethod
    def _main_model_selection(current_model_client, chain: list[str]) -> MemoryModelSelection | None:
        if current_model_client is None:
            chain.append("main_model_unavailable:missing")
            return None
        if isinstance(current_model_client, FakeModelClient):
            chain.append("main_model_unavailable:fake_provider")
            return None
        metadata = _safe_model_metadata(current_model_client)
        provider = str(metadata.get("provider") or "").strip()
        if provider == PROVIDER_FAKE:
            chain.append("main_model_unavailable:fake_provider")
            return None
        selected_model = _model_name(current_model_client)
        if not selected_model:
            chain.append("main_model_unavailable:no_model_name")
            return None
        return MemoryModelSelection(
            model_client=current_model_client,
            selected_memory_model=selected_model,
            selection_source="main_model",
            fallback_chain=chain,
            provider_metadata=metadata,
        )


class MemoryLLMExtractionEngine:
    def __init__(self, selection: MemoryModelSelection):
        self.selection = selection

    def extract(self, payload: dict, max_chars: int = SUMMARY_MAX_CHARS) -> tuple[list[memorylib.MemoryCandidate], dict]:
        metadata = _base_metadata(self.selection)
        metadata["extraction_attempt_count"] = 1
        if self.selection.has_model:
            candidates = self._extract_with_mini_llm(payload, max_chars, metadata)
            if candidates is not None:
                return candidates, metadata
        return self._deterministic_fallback(payload, max_chars, metadata, fallback_reason="no_memory_model_available")

    def _extract_with_mini_llm(self, payload: dict, max_chars: int, metadata: dict):
        metadata["mini_llm_used"] = True
        try:
            intent = _mini_llm_intent(self.selection.model_client, payload)
            metadata["intent"] = intent.to_dict()
            metadata["source_refs"] = list(intent.metadata.get("source_refs", []))
            if not intent.should_extract:
                metadata["skipped_reason"] = intent.reason or "intent_not_selected"
                metadata["summary_method"] = "mini_llm"
                return []
            try:
                candidate = _mini_llm_summary(self.selection.model_client, payload, intent, max_chars)
                metadata["candidate_count"] = 1
                metadata["extraction_success_count"] = 1
                metadata["summary_method"] = "mini_llm"
                return [candidate]
            except Exception as exc:
                metadata["schema_error_count"] += 1
                metadata["fallback_chain"].append(f"mini_llm_summary_failed:{exc}")
                candidate = _direct_clip_candidate(payload, intent, max_chars, reason=str(exc))
                if candidate is not None:
                    metadata["candidate_count"] = 1
                    metadata["extraction_success_count"] = 1
                    metadata["summary_method"] = "direct_clip"
                    metadata["fallback_reason"] = "mini_llm_summary_failed"
                    return [candidate]
                return self._deterministic_fallback(payload, max_chars, metadata, fallback_reason="mini_llm_summary_failed")
        except Exception as exc:
            metadata["schema_error_count"] += 1
            metadata["fallback_chain"].append(f"mini_llm_intent_failed:{exc}")
            metadata["fallback_reason"] = "mini_llm_intent_failed"
            return None

    def _deterministic_fallback(self, payload: dict, max_chars: int, metadata: dict, fallback_reason: str):
        metadata["deterministic_fallback_count"] += 1
        metadata["fallback_reason"] = metadata.get("fallback_reason") or fallback_reason
        engine = memorylib.MemoryExtractionEngine()
        candidates, fallback_metadata = engine.extract(payload, max_chars=max_chars)
        metadata["fallback_metadata"] = fallback_metadata
        metadata["intent"] = fallback_metadata.get("intent", metadata.get("intent", {}))
        metadata["candidate_count"] = len(candidates)
        metadata["extraction_success_count"] = len(candidates)
        if candidates:
            for candidate in candidates:
                candidate.extraction_method = "deterministic_fallback"
                candidate.metadata.update(
                    {
                        "fallback_reason": metadata["fallback_reason"],
                        "selection_source": metadata["memory_model_selection_source"],
                    }
                )
            metadata["summary_method"] = "deterministic_fallback"
        else:
            metadata["skipped_reason"] = fallback_metadata.get("skipped_reason", metadata["fallback_reason"])
            metadata["summary_method"] = "deterministic_fallback"
        return candidates, metadata


def build_memory_extraction_engine(agent) -> MemoryLLMExtractionEngine:
    resolver = MemoryModelResolver(cwd=getattr(agent, "root", "."), env_file=getattr(agent, "env_file", ".env"))
    selection = resolver.resolve(
        current_model_client=getattr(agent, "model_client", None),
        explicit_model_client=getattr(agent, "memory_extraction_model_client", None),
    )
    return MemoryLLMExtractionEngine(selection)


def _mini_llm_intent(model_client, payload: dict) -> memorylib.MemoryIntentResult:
    prompt = "\n".join(
        [
            "Decide whether this coding-agent turn contains durable memory worth saving.",
            "Return strict JSON with should_extract, topic, tags, confidence, reason, source_refs, rejection_reason.",
            "Allowed topics: " + ", ".join(memorylib.MEMORY_TOPICS),
            "Payload:",
            clip(json.dumps(payload, sort_keys=True, ensure_ascii=False), 4000),
        ]
    )
    raw = model_client.complete(prompt, 256, purpose="memory_intent", response_format="json")
    data = _strict_json_object(raw)
    if not isinstance(data.get("should_extract"), bool):
        raise ValueError("intent.should_extract must be boolean")
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("intent.tags must be a list")
    source_refs = _string_list(data.get("source_refs", []))
    reason = str(data.get("reason") or data.get("rejection_reason") or "").strip()
    intent = memorylib.MemoryIntentResult(
        should_extract=data["should_extract"],
        topic=str(data.get("topic", "task-experience")),
        tags=tags,
        confidence=str(data.get("confidence", "medium")),
        reason=reason,
        extraction_method="mini_llm",
    )
    intent.metadata = {"source_refs": source_refs, "rejection_reason": str(data.get("rejection_reason", "")).strip()}
    return intent


def _mini_llm_summary(model_client, payload: dict, intent: memorylib.MemoryIntentResult, max_chars: int) -> memorylib.MemoryCandidate:
    prompt = "\n".join(
        [
            "Summarize this turn into one bounded durable memory candidate.",
            "Return strict JSON with text, topic, tags, confidence, source_refs, rejection_reason.",
            f"Max text chars: {max_chars}",
            "Intent:",
            json.dumps(intent.to_dict(), sort_keys=True, ensure_ascii=False),
            "Payload:",
            clip(json.dumps(payload, sort_keys=True, ensure_ascii=False), 4000),
        ]
    )
    raw = model_client.complete(prompt, 256, purpose="memory_summary", response_format="json")
    data = _strict_json_object(raw)
    text = str(data.get("text") or data.get("summary") or "").strip()
    if not text:
        raise ValueError("summary.text must not be empty")
    tags = data.get("tags", intent.tags)
    if not isinstance(tags, list):
        raise ValueError("summary.tags must be a list")
    source_refs = _string_list(data.get("source_refs", getattr(intent, "metadata", {}).get("source_refs", [])))
    return memorylib.MemoryCandidate(
        text=clip(text, max(1, int(max_chars))),
        topic=str(data.get("topic", intent.topic)),
        tags=tags,
        source_type="memory_extraction",
        source_ref=str(payload.get("source_ref", "")).strip(),
        confidence=str(data.get("confidence", intent.confidence)),
        extraction_method="mini_llm",
        needs_review=True,
        metadata={
            "intent_reason": intent.reason,
            "source_refs": source_refs,
            "schema_valid": True,
            "summary_method": "mini_llm",
        },
    )


def _direct_clip_candidate(
    payload: dict,
    intent: memorylib.MemoryIntentResult,
    max_chars: int,
    reason: str,
) -> memorylib.MemoryCandidate | None:
    text = str(payload.get("current_user_request") or payload.get("final_answer") or "").strip()
    if not text:
        return None
    return memorylib.MemoryCandidate(
        text=clip(text, max(1, int(max_chars))),
        topic=intent.topic,
        tags=intent.tags,
        source_type="memory_extraction",
        source_ref=str(payload.get("source_ref", "")).strip(),
        confidence=intent.confidence,
        extraction_method="deterministic_fallback",
        needs_review=True,
        metadata={
            "intent_reason": intent.reason,
            "fallback_reason": reason,
            "schema_valid": False,
            "summary_method": "direct_clip",
            "source_refs": list(getattr(intent, "metadata", {}).get("source_refs", [])),
        },
    )


def _strict_json_object(raw: object) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("miniLLM response must be a JSON object")
    return data


def _base_metadata(selection: MemoryModelSelection) -> dict:
    return {
        "intent": {},
        "candidate_count": 0,
        "skipped_reason": "",
        "extraction_attempt_count": 0,
        "extraction_success_count": 0,
        "schema_error_count": 0,
        "deterministic_fallback_count": 0,
        "mini_llm_used": False,
        "selected_memory_model": selection.selected_memory_model,
        "memory_model_selection_source": selection.selection_source,
        "fallback_chain": list(selection.fallback_chain),
        "summary_method": "",
        "fallback_reason": "",
        "provider": dict(selection.provider_metadata),
    }


def _safe_model_metadata(model_client) -> dict:
    config = getattr(model_client, "config", None)
    if config is not None and hasattr(config, "safe_metadata"):
        return config.safe_metadata()
    metadata = getattr(model_client, "last_completion_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _model_name(model_client) -> str:
    config = getattr(model_client, "config", None)
    if config is not None:
        return str(getattr(config, "model_name", "")).strip()
    return str(getattr(model_client, "model", "")).strip()


def _string_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("source_refs must be a list")
    return [str(item).strip() for item in value if str(item).strip()][:8]


def _timeout(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 60.0
