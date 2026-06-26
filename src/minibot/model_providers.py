from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .models import FakeModelClient


PROVIDER_FAKE = "fake"
PROVIDER_HTTP = "http"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK = "deepseek"

API_FORMAT_OPENAI = "openai"
API_FORMAT_ANTHROPIC = "anthropic"

DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
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


class ModelProviderError(RuntimeError):
    pass


class ProviderConfigurationError(ValueError):
    pass


class ProviderResponseError(ModelProviderError):
    pass


class ModelClient(Protocol):
    supports_prompt_cache: bool
    last_completion_metadata: dict

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
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
class ProviderConfig:
    provider: str = PROVIDER_FAKE
    api_format: str = API_FORMAT_OPENAI
    model_name: str = "fake"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    env_file: str = DEFAULT_ENV_FILE

    def safe_metadata(self) -> dict:
        return {
            "provider": self.provider,
            "api_format": self.api_format,
            "model": self.model_name,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
        }


Transport = Callable[[HTTPRequest], HTTPResponse]


class HTTPModelClient:
    supports_prompt_cache = False

    def __init__(self, config: ProviderConfig, transport: Transport | None = None):
        if config.provider == PROVIDER_FAKE:
            raise ProviderConfigurationError("HTTPModelClient requires a non-fake provider")
        self.config = config
        self.transport = transport or _urllib_transport
        self.last_completion_metadata: dict = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        del kwargs
        started = time.perf_counter()
        request = self._build_request(prompt, max_new_tokens)
        try:
            response = self.transport(request)
            latency_ms = int((time.perf_counter() - started) * 1000)
            if response.status_code >= 400:
                raise ModelProviderError(f"provider returned HTTP {response.status_code}: {response.body[:500]}")
            payload = _json_object(response.body)
            text, usage = self._parse_payload(payload)
            self.last_completion_metadata = {
                **self.config.safe_metadata(),
                "input_chars": len(prompt),
                "output_chars": len(text),
                "latency_ms": latency_ms,
                "retry_count": 0,
                "prompt_cache_supported": False,
                "status_code": response.status_code,
                "error_category": "",
                **usage,
            }
            return text
        except Exception as exc:
            self.last_completion_metadata = {
                **self.config.safe_metadata(),
                "input_chars": len(prompt),
                "output_chars": 0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "retry_count": 0,
                "prompt_cache_supported": False,
                "error_category": _error_category(exc),
                "error": str(exc),
            }
            raise

    def _build_request(self, prompt: str, max_new_tokens: int) -> HTTPRequest:
        if self.config.api_format == API_FORMAT_OPENAI:
            return self._build_openai_request(prompt, max_new_tokens)
        if self.config.api_format == API_FORMAT_ANTHROPIC:
            return self._build_anthropic_request(prompt, max_new_tokens)
        raise ProviderConfigurationError(f"unsupported api_format: {self.config.api_format}")

    def _build_openai_request(self, prompt: str, max_new_tokens: int) -> HTTPRequest:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
        }
        return HTTPRequest(
            url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            body=json.dumps(payload, ensure_ascii=False),
            timeout=self.config.timeout_seconds,
        )

    def _build_anthropic_request(self, prompt: str, max_new_tokens: int) -> HTTPRequest:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
        }
        return HTTPRequest(
            url=self.config.base_url,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            body=json.dumps(payload, ensure_ascii=False),
            timeout=self.config.timeout_seconds,
        )

    def _parse_payload(self, payload: dict) -> tuple[str, dict]:
        if self.config.api_format == API_FORMAT_OPENAI:
            return _parse_openai_chat_response(payload)
        if self.config.api_format == API_FORMAT_ANTHROPIC:
            return _parse_anthropic_messages_response(payload)
        raise ProviderConfigurationError(f"unsupported api_format: {self.config.api_format}")


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

    config = ProviderConfig(
        provider=provider,
        api_format=resolved_api_format,
        model_name=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        api_key_env=resolved_api_key_env,
        timeout_seconds=resolved_timeout,
        env_file=str(env_file_path),
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


def _parse_openai_chat_response(payload: dict) -> tuple[str, dict]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("OpenAI-compatible response missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderResponseError("OpenAI-compatible choice must be an object")
    message = choice.get("message", {})
    if isinstance(message, dict):
        content = _content_to_text(message.get("content"))
    else:
        content = ""
    if not content and "text" in choice:
        content = _content_to_text(choice.get("text"))
    if not content:
        raise ProviderResponseError("OpenAI-compatible response missing message content")
    usage = payload.get("usage", {})
    return content, _usage_metadata(
        input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        output_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        total_tokens=usage.get("total_tokens") if isinstance(usage, dict) else None,
    )


def _parse_anthropic_messages_response(payload: dict) -> tuple[str, dict]:
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise ProviderResponseError("Anthropic-compatible response missing content blocks")
    text = "\n".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ).strip()
    if not text:
        raise ProviderResponseError("Anthropic-compatible response missing text content")
    usage = payload.get("usage", {})
    return text, _usage_metadata(
        input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
        output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
    )


def _usage_metadata(input_tokens=None, output_tokens=None, total_tokens=None) -> dict:
    result = {}
    if input_tokens is not None:
        result["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        result["total_tokens"] = int(input_tokens) + int(output_tokens)
    return result


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
    if provider == PROVIDER_ANTHROPIC or api_format == API_FORMAT_ANTHROPIC:
        return DEFAULT_ANTHROPIC_MESSAGES_URL
    return DEFAULT_OPENAI_CHAT_URL


def _validate_provider_config(config: ProviderConfig) -> None:
    if config.provider == PROVIDER_FAKE:
        return
    if not config.model_name:
        raise ProviderConfigurationError("MINIBOT_MODEL_NAME or --model-name is required for HTTP providers")
    if not config.base_url:
        raise ProviderConfigurationError("MINIBOT_BASE_URL or --base-url is required for HTTP providers")
    if not config.api_key:
        raise ProviderConfigurationError(f"API key is required; set MINIBOT_API_KEY or {config.api_key_env}")
    if config.timeout_seconds <= 0:
        raise ProviderConfigurationError("timeout_seconds must be positive")


def _float_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(f"timeout_seconds must be a number: {value}") from exc


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderResponseError):
        return "provider_response_error"
    if isinstance(exc, ModelProviderError):
        return "provider_http_error"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "provider_network_error"
    return "provider_error"
