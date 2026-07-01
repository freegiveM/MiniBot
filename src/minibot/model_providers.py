from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .models import FakeModelClient
from .prompt_cache import (
    PROMPT_CACHE_AUTO,
    PROMPT_CACHE_RETENTION_IN_MEMORY,
    normalize_prompt_cache_mode,
    normalize_prompt_cache_retention,
    provider_supports_prompt_cache,
)


PROVIDER_FAKE = "fake"
PROVIDER_HTTP = "http"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK = "deepseek"

API_FORMAT_OPENAI = "openai"
API_FORMAT_ANTHROPIC = "anthropic"

DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_ENV_FILE = ".env"
DEFAULT_API_KEY_ENV = "MINIBOT_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 60.0

MODEL_PROVIDER_ENV = "MINIBOT_MODEL_PROVIDER"
API_FORMAT_ENV = "MINIBOT_API_FORMAT"
MODEL_NAME_ENV = "MINIBOT_MODEL_NAME"
BASE_URL_ENV = "MINIBOT_BASE_URL"
API_KEY_ENV = "MINIBOT_API_KEY"
API_KEY_ENV_NAME_ENV = "MINIBOT_API_KEY_ENV"
TIMEOUT_ENV = "MINIBOT_TIMEOUT_SECONDS"
PROMPT_CACHE_ENV = "MINIBOT_PROMPT_CACHE"
PROMPT_CACHE_RETENTION_ENV = "MINIBOT_PROMPT_CACHE_RETENTION"
NATIVE_TOOLS_ENV = "MINIBOT_NATIVE_TOOLS"

NATIVE_TOOLS_AUTO = "auto"
NATIVE_TOOLS_ON = "on"
NATIVE_TOOLS_OFF = "off"


class ModelProviderError(RuntimeError):
    pass


class ProviderConfigurationError(ValueError):
    pass


class ProviderResponseError(ModelProviderError):
    pass


class ModelClient(Protocol):
    supports_prompt_cache: bool
    last_completion_metadata: dict

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str | "ModelResponse":
        ...


@dataclass(frozen=True)
class HTTPRequest:
    url: str
    headers: dict[str, str]
    body: str
    timeout: float


@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    body: str
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class ModelToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[ModelToolCall, ...] = ()
    provider_stop_reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = PROVIDER_FAKE
    api_format: str = API_FORMAT_OPENAI
    model_name: str = "fake"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    env_file: str = DEFAULT_ENV_FILE
    prompt_cache: str = PROMPT_CACHE_AUTO
    prompt_cache_retention: str = PROMPT_CACHE_RETENTION_IN_MEMORY
    native_tools: str = NATIVE_TOOLS_AUTO

    def safe_metadata(self) -> dict:
        return {
            "provider": self.provider,
            "api_format": self.api_format,
            "model": self.model_name,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "prompt_cache": self.prompt_cache,
            "prompt_cache_retention": self.prompt_cache_retention,
            "native_tools": self.native_tools,
        }


Transport = Callable[[HTTPRequest], HTTPResponse]


class HTTPModelClient:
    def __init__(self, config: ProviderConfig, transport: Transport | None = None):
        if config.provider == PROVIDER_FAKE:
            raise ProviderConfigurationError("HTTPModelClient requires a non-fake provider")
        self.config = config
        self.transport = transport or _urllib_transport
        self.last_completion_metadata: dict = {}
        self.supports_prompt_cache = provider_supports_prompt_cache(
            provider=config.provider,
            api_format=config.api_format,
            prompt_cache=config.prompt_cache,
        )

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str | ModelResponse:
        started = time.perf_counter()
        temperature = kwargs.get("temperature")
        prompt_cache_key = str(kwargs.get("prompt_cache_key") or "")
        prompt_cache_retention = str(kwargs.get("prompt_cache_retention") or self.config.prompt_cache_retention)
        native_tool_schemas = self._native_tool_schemas(kwargs.get("tools"))
        request = self._build_request(
            prompt,
            max_new_tokens,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            native_tool_schemas=native_tool_schemas,
        )
        response_status_code = 0
        response_shape: dict = {}
        try:
            response = self.transport(request)
            response_status_code = response.status_code
            latency_ms = int((time.perf_counter() - started) * 1000)
            if response.status_code >= 400:
                raise ModelProviderError(f"provider returned HTTP {response.status_code}: {response.body[:500]}")
            payload = _json_object(response.body)
            response_shape = _response_shape_summary(payload)
            parsed = self._parse_payload(payload)
            text = parsed.text
            response_metadata = {
                **dict(parsed.metadata),
                "native_tools_enabled": bool(native_tool_schemas),
                "native_tool_schema_count": len(native_tool_schemas),
                "provider_tool_call_count": len(parsed.tool_calls),
                "provider_stop_reason": parsed.provider_stop_reason,
            }
            parsed = ModelResponse(
                text=parsed.text,
                tool_calls=parsed.tool_calls,
                provider_stop_reason=parsed.provider_stop_reason,
                metadata=response_metadata,
            )
            self.last_completion_metadata = {
                **self.config.safe_metadata(),
                "request_url": _safe_url_for_metadata(request.url),
                "response_shape": response_shape,
                "input_chars": len(prompt),
                "output_chars": len(text),
                "latency_ms": latency_ms,
                "retry_count": 0,
                "native_tools_enabled": bool(native_tool_schemas),
                "native_tool_schema_count": len(native_tool_schemas),
                "provider_tool_call_count": len(parsed.tool_calls),
                "provider_stop_reason": parsed.provider_stop_reason,
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": prompt_cache_key if self.supports_prompt_cache else "",
                "prompt_cache_retention": prompt_cache_retention if self.supports_prompt_cache else "",
                "status_code": response.status_code,
                "error_category": "",
                **response_metadata,
            }
            if native_tool_schemas or parsed.tool_calls:
                return parsed
            return text
        except Exception as exc:
            self.last_completion_metadata = {
                **self.config.safe_metadata(),
                "request_url": _safe_url_for_metadata(request.url),
                "response_shape": response_shape,
                "input_chars": len(prompt),
                "output_chars": 0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "retry_count": 0,
                "native_tools_enabled": bool(native_tool_schemas),
                "native_tool_schema_count": len(native_tool_schemas),
                "provider_tool_call_count": 0,
                "provider_stop_reason": "",
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": prompt_cache_key if self.supports_prompt_cache else "",
                "prompt_cache_retention": prompt_cache_retention if self.supports_prompt_cache else "",
                "error_category": _error_category(exc),
                "error": str(exc),
            }
            if response_status_code:
                self.last_completion_metadata["status_code"] = response_status_code
            raise

    def _build_request(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature=None,
        prompt_cache_key: str = "",
        prompt_cache_retention: str = PROMPT_CACHE_RETENTION_IN_MEMORY,
        native_tool_schemas: list[dict] | None = None,
    ) -> HTTPRequest:
        if self.config.api_format == API_FORMAT_OPENAI:
            return self._build_openai_request(
                prompt,
                max_new_tokens,
                temperature=temperature,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                native_tool_schemas=native_tool_schemas,
            )
        if self.config.api_format == API_FORMAT_ANTHROPIC:
            return self._build_anthropic_request(
                prompt,
                max_new_tokens,
                temperature=temperature,
                native_tool_schemas=native_tool_schemas,
            )
        raise ProviderConfigurationError(f"unsupported api_format: {self.config.api_format}")

    def _build_openai_request(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature=None,
        prompt_cache_key: str = "",
        prompt_cache_retention: str = PROMPT_CACHE_RETENTION_IN_MEMORY,
        native_tool_schemas: list[dict] | None = None,
    ) -> HTTPRequest:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
        }
        if native_tool_schemas:
            payload["tools"] = native_tool_schemas
            payload["tool_choice"] = "auto"
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
            payload["prompt_cache_retention"] = normalize_prompt_cache_retention(prompt_cache_retention)
        if _is_deepseek_endpoint(self.config):
            payload["thinking"] = {"type": "disabled"}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        return HTTPRequest(
            url=_request_endpoint_url(self.config),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            body=json.dumps(payload, ensure_ascii=False),
            timeout=self.config.timeout_seconds,
        )

    def _build_anthropic_request(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature=None,
        native_tool_schemas: list[dict] | None = None,
    ) -> HTTPRequest:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
        }
        if native_tool_schemas:
            payload["tools"] = native_tool_schemas
        if _is_deepseek_endpoint(self.config):
            payload["thinking"] = {"type": "disabled"}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        return HTTPRequest(
            url=_request_endpoint_url(self.config),
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            body=json.dumps(payload, ensure_ascii=False),
            timeout=self.config.timeout_seconds,
        )

    def _parse_payload(self, payload: dict) -> ModelResponse:
        if self.config.api_format == API_FORMAT_OPENAI:
            return _parse_openai_chat_response(payload)
        if self.config.api_format == API_FORMAT_ANTHROPIC:
            return _parse_anthropic_messages_response(payload)
        raise ProviderConfigurationError(f"unsupported api_format: {self.config.api_format}")

    def _native_tool_schemas(self, tools) -> list[dict]:
        if tools is None or self.config.native_tools == NATIVE_TOOLS_OFF:
            return []
        schemas = _provider_tool_schemas(tools, self.config.api_format)
        if self.config.native_tools == NATIVE_TOOLS_ON and not schemas:
            raise ProviderConfigurationError("native tools are enabled but no tool schemas were available")
        return schemas


def load_dotenv(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _clean_env_value(value)
    return values


def resolve_provider_config(
    *,
    cwd: str | Path = ".",
    env_file: str | Path = DEFAULT_ENV_FILE,
    environ: dict[str, str] | None = None,
    model_provider: str | None = None,
    api_format: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: float | str | None = None,
    prompt_cache: str | None = None,
    prompt_cache_retention: str | None = None,
    native_tools: str | None = None,
) -> ProviderConfig:
    cwd_path = Path(cwd).resolve()
    env_file_path = Path(env_file)
    if not env_file_path.is_absolute():
        env_file_path = cwd_path / env_file_path
    dotenv = load_dotenv(env_file_path)
    process_env = dict(os.environ if environ is None else environ)
    merged = {**dotenv, **process_env}

    provider = _coalesce(model_provider, merged.get(MODEL_PROVIDER_ENV), PROVIDER_FAKE).lower()
    provider = _normalize_provider(provider)
    resolved_api_format = _coalesce(api_format, merged.get(API_FORMAT_ENV), _default_api_format(provider)).lower()
    resolved_api_format = _normalize_api_format(resolved_api_format)
    resolved_model = _coalesce(model_name, merged.get(MODEL_NAME_ENV), "fake" if provider == PROVIDER_FAKE else "")
    resolved_base_url = _coalesce(base_url, merged.get(BASE_URL_ENV), _default_base_url(provider, resolved_api_format))
    resolved_api_key_env = _coalesce(api_key_env, merged.get(API_KEY_ENV_NAME_ENV), DEFAULT_API_KEY_ENV)
    resolved_api_key = _coalesce(merged.get(API_KEY_ENV), merged.get(resolved_api_key_env), "")
    resolved_timeout = _float_value(_coalesce(timeout_seconds, merged.get(TIMEOUT_ENV), DEFAULT_TIMEOUT_SECONDS))
    resolved_prompt_cache = _prompt_cache_mode(_coalesce(prompt_cache, merged.get(PROMPT_CACHE_ENV), PROMPT_CACHE_AUTO))
    resolved_prompt_cache_retention = _prompt_cache_retention(
        _coalesce(prompt_cache_retention, merged.get(PROMPT_CACHE_RETENTION_ENV), PROMPT_CACHE_RETENTION_IN_MEMORY)
    )
    resolved_native_tools = _native_tools_mode(_coalesce(native_tools, merged.get(NATIVE_TOOLS_ENV), NATIVE_TOOLS_AUTO))

    config = ProviderConfig(
        provider=provider,
        api_format=resolved_api_format,
        model_name=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        api_key_env=resolved_api_key_env,
        timeout_seconds=resolved_timeout,
        env_file=str(env_file_path),
        prompt_cache=resolved_prompt_cache,
        prompt_cache_retention=resolved_prompt_cache_retention,
        native_tools=resolved_native_tools,
    )
    _validate_provider_config(config)
    return config


def build_model_client_from_config(
    config: ProviderConfig,
    *,
    fake_response: str = "<final>MiniBot scaffold is running.</final>",
    transport: Transport | None = None,
) -> ModelClient:
    if config.provider == PROVIDER_FAKE:
        return FakeModelClient([fake_response], model=config.model_name or "fake-cli")
    return HTTPModelClient(config, transport=transport)


def _parse_openai_chat_response(payload: dict) -> ModelResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("OpenAI-compatible response missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderResponseError("OpenAI-compatible choice must be an object")
    message = choice.get("message", {})
    tool_calls: list[ModelToolCall] = []
    if isinstance(message, dict):
        content = _content_to_text(message.get("content"))
        tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
    else:
        content = ""
    if not content and "text" in choice:
        content = _content_to_text(choice.get("text"))
    if not content and not tool_calls:
        raise ProviderResponseError("OpenAI-compatible response missing message content")
    usage = payload.get("usage", {})
    provider_stop_reason = str(choice.get("finish_reason") or "")
    return ModelResponse(
        text=content,
        tool_calls=tuple(tool_calls),
        provider_stop_reason=provider_stop_reason,
        metadata=_usage_metadata(
        input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        output_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        total_tokens=usage.get("total_tokens") if isinstance(usage, dict) else None,
        cached_tokens=_cached_tokens_from_usage(usage) if isinstance(usage, dict) else None,
        ),
    )


def _parse_anthropic_messages_response(payload: dict) -> ModelResponse:
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise ProviderResponseError("Anthropic-compatible response missing content blocks")
    text_parts: list[str] = []
    tool_calls: list[ModelToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_calls.append(_parse_anthropic_tool_use(block))
            continue
        if block_type == "thinking":
            continue
        text = block.get("text") or block.get("content")
        if text:
            text_parts.append(str(text))
    text = "\n".join(text_parts).strip()
    if not text and not tool_calls:
        shape = _response_shape_summary(payload)
        raise ProviderResponseError(f"Anthropic-compatible response missing text content; shape={shape}")
    usage = payload.get("usage", {})
    provider_stop_reason = str(payload.get("stop_reason") or "")
    return ModelResponse(
        text=text,
        tool_calls=tuple(tool_calls),
        provider_stop_reason=provider_stop_reason,
        metadata=_usage_metadata(
        input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
        output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
        cached_tokens=_cached_tokens_from_usage(usage) if isinstance(usage, dict) else None,
        ),
    )


def _usage_metadata(input_tokens=None, output_tokens=None, total_tokens=None, cached_tokens=None) -> dict:
    result = {}
    if input_tokens is not None:
        result["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        result["total_tokens"] = int(input_tokens) + int(output_tokens)
    if cached_tokens is not None:
        result["cached_tokens"] = int(cached_tokens)
        result["cache_hit"] = int(cached_tokens) > 0
    return result


def _cached_tokens_from_usage(usage: dict) -> int | None:
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            return int(details.get("cached_tokens") or 0)
    if usage.get("cached_tokens") is not None:
        return int(usage.get("cached_tokens") or 0)
    return None


def _urllib_transport(request: HTTPRequest) -> HTTPResponse:
    http_request = urllib.request.Request(
        request.url,
        data=request.body.encode("utf-8"),
        headers=request.headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return HTTPResponse(status_code=int(response.status), body=body, headers=dict(response.headers))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return HTTPResponse(status_code=int(exc.code), body=body, headers=dict(exc.headers))


def _json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"provider response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderResponseError("provider response must be a JSON object")
    return payload


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


def _provider_tool_schemas(tools, api_format: str) -> list[dict]:
    if tools is None:
        return []
    if hasattr(tools, "items"):
        raw_items = list(tools.items())
    elif isinstance(tools, dict):
        raw_items = list(tools.items())
    else:
        return []

    schemas = []
    for name, spec in sorted(raw_items, key=lambda item: str(item[0])):
        tool_name = str(getattr(spec, "name", "") or name).strip()
        if not tool_name:
            continue
        description = str(getattr(spec, "description", "") or "")
        parameters = _minibot_args_schema_to_json_schema(getattr(spec, "schema", {}) or {})
        if api_format == API_FORMAT_OPENAI:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        elif api_format == API_FORMAT_ANTHROPIC:
            schemas.append(
                {
                    "name": tool_name,
                    "description": description,
                    "input_schema": parameters,
                }
            )
    return schemas


def _minibot_args_schema_to_json_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        raise ProviderConfigurationError("tool schema must be an object")
    properties = {}
    required = []
    for key, descriptor in sorted(schema.items()):
        if not isinstance(key, str) or not key:
            raise ProviderConfigurationError("tool schema keys must be non-empty strings")
        text = str(descriptor)
        kind = text.split("=", 1)[0].strip()
        is_required = "=" not in text
        json_type = _json_schema_type(kind)
        prop = {
            "type": json_type,
            "description": f"Argument `{key}` for this MiniBot tool.",
        }
        if json_type == "array":
            prop["items"] = {}
        properties[key] = prop
        if is_required:
            required.append(key)
    result = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _json_schema_type(kind: str) -> str:
    mapping = {
        "str": "string",
        "int": "integer",
        "bool": "boolean",
        "list": "array",
    }
    try:
        return mapping[kind]
    except KeyError as exc:
        raise ProviderConfigurationError(f"unsupported tool arg descriptor: {kind}") from exc


def _parse_openai_tool_calls(raw_tool_calls) -> list[ModelToolCall]:
    if raw_tool_calls in (None, ""):
        return []
    if not isinstance(raw_tool_calls, list):
        raise ProviderResponseError("OpenAI-compatible tool_calls must be a list")
    calls = []
    for index, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            raise ProviderResponseError(f"OpenAI-compatible tool_call at index {index} must be an object")
        function = item.get("function")
        if not isinstance(function, dict):
            raise ProviderResponseError(f"OpenAI-compatible tool_call at index {index} missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProviderResponseError(f"OpenAI-compatible tool_call at index {index} missing function name")
        args = _tool_args_object(function.get("arguments"), provider="OpenAI-compatible", index=index)
        calls.append(ModelToolCall(name=name.strip(), args=args, id=str(item.get("id") or "")))
    return calls


def _parse_anthropic_tool_use(block: dict) -> ModelToolCall:
    name = block.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProviderResponseError("Anthropic-compatible tool_use missing name")
    args = _tool_args_object(block.get("input", block.get("args", {})), provider="Anthropic-compatible", index=0)
    return ModelToolCall(name=name.strip(), args=args, id=str(block.get("id") or ""))


def _tool_args_object(raw_args, *, provider: str, index: int) -> dict:
    if raw_args in (None, ""):
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(f"{provider} tool arguments at index {index} were not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ProviderResponseError(f"{provider} tool arguments at index {index} must be an object")
        return parsed
    raise ProviderResponseError(f"{provider} tool arguments at index {index} must be an object")


def _response_shape_summary(payload: dict) -> dict:
    summary: dict = {"top_level_keys": sorted(str(key) for key in payload.keys())[:20]}
    if "content" in payload:
        content = payload.get("content")
        summary["content_type"] = type(content).__name__
        if isinstance(content, list):
            summary["content_block_count"] = len(content)
            summary["content_block_types"] = [
                str(block.get("type", ""))[:80] if isinstance(block, dict) else type(block).__name__
                for block in content[:8]
            ]
            summary["content_tool_use_count"] = sum(
                1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use"
            )
            summary["content_block_keys"] = [
                sorted(str(key) for key in block.keys())[:12] if isinstance(block, dict) else []
                for block in content[:8]
            ]
        elif isinstance(content, dict):
            summary["content_keys"] = sorted(str(key) for key in content.keys())[:12]
        elif isinstance(content, str):
            summary["content_chars"] = len(content)
    choices = payload.get("choices")
    if isinstance(choices, list):
        summary["choices_count"] = len(choices)
        if choices and isinstance(choices[0], dict):
            first_choice = choices[0]
            summary["first_choice_keys"] = sorted(str(key) for key in first_choice.keys())[:12]
            message = first_choice.get("message")
            if isinstance(message, dict):
                summary["message_keys"] = sorted(str(key) for key in message.keys())[:12]
                summary["message_content_type"] = type(message.get("content")).__name__
                tool_calls = message.get("tool_calls")
                summary["message_tool_call_count"] = len(tool_calls) if isinstance(tool_calls, list) else 0
    error = payload.get("error")
    if isinstance(error, dict):
        summary["error_keys"] = sorted(str(key) for key in error.keys())[:12]
        if "type" in error:
            summary["error_type"] = str(error.get("type", ""))[:80]
        if "code" in error:
            summary["error_code"] = str(error.get("code", ""))[:80]
    elif isinstance(error, str):
        summary["error_type"] = "string"
    return summary


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _coalesce(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    if provider not in {PROVIDER_FAKE, PROVIDER_HTTP, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_DEEPSEEK}:
        raise ProviderConfigurationError(f"unsupported model provider: {value}")
    return provider


def _normalize_api_format(value: str) -> str:
    api_format = value.strip().lower()
    if api_format not in {API_FORMAT_OPENAI, API_FORMAT_ANTHROPIC}:
        raise ProviderConfigurationError(f"unsupported api_format: {value}")
    return api_format


def _default_api_format(provider: str) -> str:
    if provider == PROVIDER_ANTHROPIC:
        return API_FORMAT_ANTHROPIC
    return API_FORMAT_OPENAI


def _default_base_url(provider: str, api_format: str) -> str:
    if provider == PROVIDER_FAKE:
        return ""
    if provider == PROVIDER_DEEPSEEK:
        if api_format == API_FORMAT_ANTHROPIC:
            return DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL
        return DEFAULT_DEEPSEEK_OPENAI_BASE_URL
    if provider == PROVIDER_ANTHROPIC or api_format == API_FORMAT_ANTHROPIC:
        return DEFAULT_ANTHROPIC_MESSAGES_URL
    return DEFAULT_OPENAI_CHAT_URL


def _request_endpoint_url(config: ProviderConfig) -> str:
    if config.api_format == API_FORMAT_OPENAI:
        return _openai_endpoint_url(config.provider, config.base_url)
    if config.api_format == API_FORMAT_ANTHROPIC:
        return _anthropic_endpoint_url(config.provider, config.base_url)
    raise ProviderConfigurationError(f"unsupported api_format: {config.api_format}")


def _openai_endpoint_url(provider: str, base_url: str) -> str:
    path = _url_path(base_url)
    if path.endswith("/chat/completions"):
        return base_url
    if path.endswith("/v1"):
        return _append_url_path(base_url, "chat/completions")
    if path == "":
        if provider == PROVIDER_DEEPSEEK or _url_host_contains(base_url, "deepseek"):
            return _append_url_path(base_url, "chat/completions")
        return _append_url_path(base_url, "v1/chat/completions")
    return base_url


def _anthropic_endpoint_url(provider: str, base_url: str) -> str:
    path = _url_path(base_url)
    if path.endswith("/v1/messages") or path.endswith("/messages"):
        return base_url
    if path.endswith("/v1"):
        return _append_url_path(base_url, "messages")
    if path == "" or path.endswith("/anthropic") or provider == PROVIDER_DEEPSEEK:
        return _append_url_path(base_url, "v1/messages")
    return base_url


def _url_path(url: str) -> str:
    return urllib.parse.urlsplit(url.strip()).path.rstrip("/").lower()


def _url_host_contains(url: str, text: str) -> bool:
    return text.lower() in urllib.parse.urlsplit(url.strip()).netloc.lower()


def _is_deepseek_endpoint(config: ProviderConfig) -> bool:
    return config.provider == PROVIDER_DEEPSEEK or _url_host_contains(config.base_url, "deepseek")


def _append_url_path(url: str, suffix: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    path = parts.path.rstrip("/")
    suffix_path = "/" + suffix.strip("/")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path + suffix_path, parts.query, parts.fragment))


def _safe_url_for_metadata(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _validate_provider_config(config: ProviderConfig) -> None:
    if config.provider == PROVIDER_FAKE:
        return
    if not config.model_name:
        raise ProviderConfigurationError("MINIBOT_MODEL_NAME or --model-name is required for HTTP providers")
    if not config.base_url:
        raise ProviderConfigurationError("MINIBOT_BASE_URL or --base-url is required for HTTP providers")
    base_url_api_format = _base_url_api_format_hint(config.base_url)
    if base_url_api_format and base_url_api_format != config.api_format:
        raise ProviderConfigurationError(
            f"MINIBOT_API_FORMAT={config.api_format} does not match MINIBOT_BASE_URL; "
            f"use api_format={base_url_api_format} or a matching base URL"
        )
    if not config.api_key:
        raise ProviderConfigurationError(f"API key is required; set MINIBOT_API_KEY or {config.api_key_env}")
    if config.timeout_seconds <= 0:
        raise ProviderConfigurationError("timeout_seconds must be positive")


def _base_url_api_format_hint(base_url: str) -> str:
    path = _url_path(base_url)
    if path.endswith("/chat/completions"):
        return API_FORMAT_OPENAI
    if path.endswith("/v1/messages") or path.endswith("/messages") or "/anthropic" in path:
        return API_FORMAT_ANTHROPIC
    return ""


def _float_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(f"timeout_seconds must be a number: {value}") from exc


def _prompt_cache_mode(value: str) -> str:
    try:
        return normalize_prompt_cache_mode(value)
    except ValueError as exc:
        raise ProviderConfigurationError(str(exc)) from exc


def _prompt_cache_retention(value: str) -> str:
    try:
        return normalize_prompt_cache_retention(value)
    except ValueError as exc:
        raise ProviderConfigurationError(str(exc)) from exc


def _native_tools_mode(value: str) -> str:
    mode = str(value or NATIVE_TOOLS_AUTO).strip().lower()
    if mode not in {NATIVE_TOOLS_AUTO, NATIVE_TOOLS_ON, NATIVE_TOOLS_OFF}:
        raise ProviderConfigurationError(f"unsupported native tools mode: {value}")
    return mode


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderResponseError):
        return "provider_response_error"
    if isinstance(exc, ModelProviderError):
        return "provider_http_error"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "provider_network_error"
    return "provider_error"
