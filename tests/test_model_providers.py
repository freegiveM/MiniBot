from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.model_providers import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    HTTPModelClient,
    HTTPResponse,
    ModelResponse,
    ProviderConfig,
    ProviderConfigurationError,
    ProviderResponseError,
    PROVIDER_DEEPSEEK,
    build_model_client_from_config,
    load_dotenv,
    resolve_provider_config,
)
from minibot.models import FakeModelClient
from minibot.tools import BASE_TOOL_SPECS


class ModelProviderTests(unittest.TestCase):
    def test_load_dotenv_supports_comments_quotes_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "MINIBOT_MODEL_PROVIDER=http",
                        "export MINIBOT_MODEL_NAME='mini-model'",
                        'MINIBOT_BASE_URL="https://example.test/chat"',
                        "MINIBOT_API_KEY=secret # local only",
                    ]
                ),
                encoding="utf-8",
            )

            values = load_dotenv(path)

            self.assertEqual(values["MINIBOT_MODEL_PROVIDER"], "http")
            self.assertEqual(values["MINIBOT_MODEL_NAME"], "mini-model")
            self.assertEqual(values["MINIBOT_BASE_URL"], "https://example.test/chat")
            self.assertEqual(values["MINIBOT_API_KEY"], "secret")

    def test_resolve_provider_config_uses_dotenv_env_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "MINIBOT_MODEL_PROVIDER=http",
                        "MINIBOT_API_FORMAT=openai",
                        "MINIBOT_MODEL_NAME=dotenv-model",
                        "MINIBOT_BASE_URL=https://dotenv.test/chat",
                        "MINIBOT_API_KEY_ENV=CUSTOM_KEY",
                        "CUSTOM_KEY=dotenv-key",
                        "MINIBOT_PROMPT_CACHE=openai_explicit",
                        "MINIBOT_PROMPT_CACHE_RETENTION=24h",
                        "MINIBOT_NATIVE_TOOLS=off",
                    ]
                ),
                encoding="utf-8",
            )

            config = resolve_provider_config(
                cwd=root,
                environ={"MINIBOT_MODEL_NAME": "env-model"},
                api_format=API_FORMAT_ANTHROPIC,
                base_url="https://cli.test/messages",
            )

            self.assertEqual(config.provider, "http")
            self.assertEqual(config.api_format, API_FORMAT_ANTHROPIC)
            self.assertEqual(config.model_name, "env-model")
            self.assertEqual(config.base_url, "https://cli.test/messages")
            self.assertEqual(config.api_key, "dotenv-key")
            self.assertEqual(config.api_key_env, "CUSTOM_KEY")
            self.assertEqual(config.prompt_cache, "openai_explicit")
            self.assertEqual(config.prompt_cache_retention, "24h")
            self.assertEqual(config.native_tools, "off")

    def test_resolve_provider_config_defaults_deepseek_base_url_by_api_format(self):
        openai_config = resolve_provider_config(
            cwd=".",
            env_file="missing.env",
            environ={"MINIBOT_API_KEY": "secret"},
            model_provider=PROVIDER_DEEPSEEK,
            model_name="deepseek-v4-pro",
        )
        anthropic_config = resolve_provider_config(
            cwd=".",
            env_file="missing.env",
            environ={"MINIBOT_API_KEY": "secret"},
            model_provider=PROVIDER_DEEPSEEK,
            api_format=API_FORMAT_ANTHROPIC,
            model_name="deepseek-v4-pro",
        )

        self.assertEqual(openai_config.api_format, API_FORMAT_OPENAI)
        self.assertEqual(openai_config.base_url, "https://api.deepseek.com")
        self.assertEqual(anthropic_config.api_format, API_FORMAT_ANTHROPIC)
        self.assertEqual(anthropic_config.base_url, "https://api.deepseek.com/anthropic")

    def test_resolve_provider_config_rejects_missing_api_key_for_http(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ProviderConfigurationError):
                resolve_provider_config(
                    cwd=temp,
                    env_file="missing.env",
                    environ={},
                    model_provider="http",
                    api_format=API_FORMAT_OPENAI,
                    model_name="mini",
                    base_url="https://example.test/chat",
                )

    def test_resolve_provider_config_rejects_api_format_base_url_mismatch(self):
        with self.assertRaisesRegex(ProviderConfigurationError, "does not match MINIBOT_BASE_URL"):
            resolve_provider_config(
                cwd=".",
                env_file="missing.env",
                environ={"MINIBOT_API_KEY": "secret"},
                model_provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com/anthropic",
            )

    def test_build_model_client_from_fake_config_keeps_fake_path(self):
        client = build_model_client_from_config(
            ProviderConfig(provider="fake", model_name="fake-cli"),
            fake_response="<final>ok</final>",
        )

        self.assertIsInstance(client, FakeModelClient)
        self.assertEqual(client.complete("hello", 10), "<final>ok</final>")

    def test_http_model_client_parses_openai_compatible_response(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "choices": [{"message": {"content": "<final>openai ok</final>"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="mini",
                base_url="https://example.test/chat",
                api_key="secret",
            ),
            transport=transport,
        )

        self.assertEqual(client.complete("hello", 32, temperature=0.2), "<final>openai ok</final>")
        request_payload = json.loads(captured[0].body)
        self.assertEqual(captured[0].url, "https://example.test/chat")
        self.assertEqual(captured[0].headers["Authorization"], "Bearer secret")
        self.assertEqual(request_payload["messages"][0]["content"], "hello")
        self.assertEqual(request_payload["max_tokens"], 32)
        self.assertEqual(request_payload["temperature"], 0.2)
        self.assertNotIn("thinking", request_payload)
        self.assertEqual(client.last_completion_metadata["api_format"], API_FORMAT_OPENAI)
        self.assertEqual(client.last_completion_metadata["request_url"], "https://example.test/chat")
        self.assertEqual(client.last_completion_metadata["total_tokens"], 7)
        self.assertTrue(client.last_completion_metadata["api_key_present"])
        self.assertNotIn("secret", json.dumps(client.last_completion_metadata))

    def test_http_model_client_sends_and_parses_openai_native_tool_calls(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": "{\"path\":\"README.md\",\"start\":1,\"end\":3}",
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="mini",
                base_url="https://example.test/chat",
                api_key="secret",
            ),
            transport=transport,
        )

        response = client.complete("hello", 32, tools=BASE_TOOL_SPECS)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].args, {"path": "README.md", "start": 1, "end": 3})
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.provider_stop_reason, "tool_calls")
        request_payload = json.loads(captured[0].body)
        self.assertEqual(request_payload["tool_choice"], "auto")
        read_file_schema = [
            item for item in request_payload["tools"] if item["function"]["name"] == "read_file"
        ][0]["function"]["parameters"]
        self.assertEqual(read_file_schema["properties"]["path"]["type"], "string")
        self.assertIn("path", read_file_schema["required"])
        self.assertEqual(read_file_schema["properties"]["start"]["type"], "integer")
        self.assertNotIn("start", read_file_schema["required"])
        self.assertEqual(client.last_completion_metadata["native_tool_schema_count"], len(BASE_TOOL_SPECS))
        self.assertEqual(client.last_completion_metadata["provider_tool_call_count"], 1)
        self.assertEqual(client.last_completion_metadata["response_shape"]["message_tool_call_count"], 1)

    def test_http_model_client_normalizes_openai_base_urls(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        root_client = HTTPModelClient(
            ProviderConfig(
                provider=PROVIDER_DEEPSEEK,
                api_format=API_FORMAT_OPENAI,
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com",
                api_key="secret",
            ),
            transport=transport,
        )
        v1_client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="mini",
                base_url="https://proxy.test/v1",
                api_key="secret",
            ),
            transport=transport,
        )

        self.assertEqual(root_client.complete("hello", 16), "ok")
        self.assertEqual(v1_client.complete("hello", 16), "ok")
        self.assertEqual(captured[0].url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured[1].url, "https://proxy.test/v1/chat/completions")
        self.assertEqual(json.loads(captured[0].body)["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", json.loads(captured[1].body))

    def test_http_model_client_parses_anthropic_compatible_response(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "content": [
                            {"type": "text", "text": "<final>anthropic"},
                            {"type": "text", "text": "ok</final>"},
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 6},
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="anthropic",
                api_format=API_FORMAT_ANTHROPIC,
                model_name="claude-mini",
                base_url="https://example.test/messages",
                api_key="secret",
            ),
            transport=transport,
        )

        self.assertEqual(client.complete("hello", 16, temperature=0.1), "<final>anthropic\nok</final>")
        request_payload = json.loads(captured[0].body)
        self.assertEqual(captured[0].url, "https://example.test/messages")
        self.assertEqual(captured[0].headers["x-api-key"], "secret")
        self.assertIn("anthropic-version", captured[0].headers)
        self.assertEqual(request_payload["messages"][0]["content"], "hello")
        self.assertEqual(request_payload["temperature"], 0.1)
        self.assertNotIn("thinking", request_payload)
        self.assertEqual(client.last_completion_metadata["api_format"], API_FORMAT_ANTHROPIC)
        self.assertEqual(client.last_completion_metadata["request_url"], "https://example.test/messages")
        self.assertEqual(client.last_completion_metadata["total_tokens"], 11)

    def test_http_model_client_returns_native_metadata_for_text_final(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "content": [{"type": "text", "text": "plain final from provider"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 6},
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="anthropic",
                api_format=API_FORMAT_ANTHROPIC,
                model_name="claude-mini",
                base_url="https://example.test/messages",
                api_key="secret",
            ),
            transport=transport,
        )

        response = client.complete("hello", 16, tools=BASE_TOOL_SPECS)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.text, "plain final from provider")
        self.assertTrue(response.metadata["native_tools_enabled"])
        self.assertEqual(response.metadata["native_tool_schema_count"], len(BASE_TOOL_SPECS))
        self.assertEqual(response.metadata["provider_tool_call_count"], 0)
        self.assertEqual(response.provider_stop_reason, "end_turn")
        self.assertIn("tools", json.loads(captured[0].body))

    def test_http_model_client_sends_and_parses_anthropic_native_tool_use(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "patch_file",
                                "input": {
                                    "path": "README.md",
                                    "old_text": "Status: draft",
                                    "new_text": "Status: reviewed",
                                },
                            }
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 5, "output_tokens": 6},
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="anthropic",
                api_format=API_FORMAT_ANTHROPIC,
                model_name="claude-mini",
                base_url="https://example.test/messages",
                api_key="secret",
            ),
            transport=transport,
        )

        response = client.complete("hello", 16, tools=BASE_TOOL_SPECS)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.tool_calls[0].name, "patch_file")
        self.assertEqual(response.tool_calls[0].args["path"], "README.md")
        self.assertEqual(response.tool_calls[0].id, "toolu_1")
        self.assertEqual(response.provider_stop_reason, "tool_use")
        request_payload = json.loads(captured[0].body)
        patch_schema = [item for item in request_payload["tools"] if item["name"] == "patch_file"][0]["input_schema"]
        self.assertEqual(patch_schema["properties"]["new_text"]["type"], "string")
        self.assertIn("new_text", patch_schema["required"])
        self.assertEqual(client.last_completion_metadata["native_tool_schema_count"], len(BASE_TOOL_SPECS))
        self.assertEqual(client.last_completion_metadata["provider_tool_call_count"], 1)
        self.assertEqual(client.last_completion_metadata["response_shape"]["content_tool_use_count"], 1)

    def test_http_model_client_normalizes_deepseek_anthropic_base_url(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps({"content": [{"type": "text", "text": "ok"}], "usage": {}}),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider=PROVIDER_DEEPSEEK,
                api_format=API_FORMAT_ANTHROPIC,
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com/anthropic",
                api_key="secret",
            ),
            transport=transport,
        )

        self.assertEqual(client.complete("hello", 16), "ok")
        self.assertEqual(captured[0].url, "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(json.loads(captured[0].body)["thinking"], {"type": "disabled"})

    def test_http_model_client_records_bad_response_metadata(self):
        def transport(request):
            del request
            return HTTPResponse(200, json.dumps({"choices": []}))

        client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="mini",
                base_url="https://example.test/chat",
                api_key="secret",
            ),
            transport=transport,
        )

        with self.assertRaises(ProviderResponseError):
            client.complete("hello", 16)
        self.assertEqual(client.last_completion_metadata["error_category"], "provider_response_error")
        self.assertEqual(client.last_completion_metadata["request_url"], "https://example.test/chat")
        self.assertEqual(client.last_completion_metadata["status_code"], 200)
        self.assertEqual(client.last_completion_metadata["response_shape"]["top_level_keys"], ["choices"])
        self.assertTrue(client.last_completion_metadata["api_key_present"])

    def test_http_model_client_records_anthropic_response_shape_without_text(self):
        def transport(request):
            del request
            return HTTPResponse(200, json.dumps({"content": [{"type": "thinking", "thinking": "hidden"}]}))

        client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_ANTHROPIC,
                model_name="mini",
                base_url="https://example.test/messages",
                api_key="secret",
            ),
            transport=transport,
        )

        with self.assertRaisesRegex(ProviderResponseError, "content_block_types"):
            client.complete("hello", 16)
        shape = client.last_completion_metadata["response_shape"]
        self.assertEqual(shape["content_block_types"], ["thinking"])
        self.assertEqual(shape["content_block_keys"], [["thinking", "type"]])
        self.assertNotIn("hidden", json.dumps(client.last_completion_metadata))

    def test_openai_provider_sends_prompt_cache_key_and_in_memory_retention(self):
        captured = []

        def transport(request):
            captured.append(request)
            return HTTPResponse(
                200,
                json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                            "prompt_tokens_details": {"cached_tokens": 7},
                        },
                    }
                ),
            )

        client = HTTPModelClient(
            ProviderConfig(
                provider="openai",
                api_format=API_FORMAT_OPENAI,
                model_name="gpt-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                api_key="secret",
                prompt_cache="auto",
                prompt_cache_retention="in-memory",
            ),
            transport=transport,
        )

        self.assertTrue(client.supports_prompt_cache)
        self.assertEqual(
            client.complete("hello", 16, prompt_cache_key="cache-key", prompt_cache_retention="in-memory"),
            "ok",
        )
        request_payload = json.loads(captured[0].body)
        self.assertEqual(request_payload["prompt_cache_key"], "cache-key")
        self.assertEqual(request_payload["prompt_cache_retention"], "in-memory")
        self.assertEqual(client.last_completion_metadata["cached_tokens"], 7)
        self.assertTrue(client.last_completion_metadata["cache_hit"])
        self.assertTrue(client.last_completion_metadata["prompt_cache_supported"])

    def test_fake_anthropic_and_deepseek_do_not_send_openai_cache_fields_by_default(self):
        captured_http = []
        captured_anthropic = []
        captured_deepseek = []

        def http_transport(request):
            captured_http.append(json.loads(request.body))
            if "messages" in json.loads(request.body):
                return HTTPResponse(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        def anthropic_transport(request):
            captured_anthropic.append(json.loads(request.body))
            return HTTPResponse(200, json.dumps({"content": [{"type": "text", "text": "ok"}]}))

        def deepseek_transport(request):
            captured_deepseek.append(json.loads(request.body))
            return HTTPResponse(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}))

        http_client = HTTPModelClient(
            ProviderConfig(
                provider="http",
                api_format=API_FORMAT_OPENAI,
                model_name="mini",
                base_url="https://proxy.test/v1/chat/completions",
                api_key="secret",
            ),
            transport=http_transport,
        )
        anthropic_client = HTTPModelClient(
            ProviderConfig(
                provider="anthropic",
                api_format=API_FORMAT_ANTHROPIC,
                model_name="claude-mini",
                base_url="https://example.test/messages",
                api_key="secret",
                prompt_cache="openai_explicit",
            ),
            transport=anthropic_transport,
        )
        deepseek_client = HTTPModelClient(
            ProviderConfig(
                provider=PROVIDER_DEEPSEEK,
                api_format=API_FORMAT_OPENAI,
                model_name="deepseek-v4-pro",
                base_url="https://api.deepseek.com",
                api_key="secret",
            ),
            transport=deepseek_transport,
        )

        http_client.complete("hello", 16, prompt_cache_key="ignored")
        anthropic_client.complete("hello", 16, prompt_cache_key="ignored")
        deepseek_client.complete("hello", 16, prompt_cache_key="ignored")

        self.assertFalse(http_client.supports_prompt_cache)
        self.assertFalse(anthropic_client.supports_prompt_cache)
        self.assertFalse(deepseek_client.supports_prompt_cache)
        self.assertNotIn("prompt_cache_key", captured_http[0])
        self.assertNotIn("prompt_cache_retention", captured_http[0])
        self.assertNotIn("prompt_cache_key", captured_anthropic[0])
        self.assertNotIn("prompt_cache_retention", captured_anthropic[0])
        self.assertNotIn("prompt_cache_key", captured_deepseek[0])
        self.assertNotIn("prompt_cache_retention", captured_deepseek[0])


if __name__ == "__main__":
    unittest.main()
