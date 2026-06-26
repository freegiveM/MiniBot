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
    ProviderConfig,
    ProviderConfigurationError,
    ProviderResponseError,
    build_model_client_from_config,
    load_dotenv,
    resolve_provider_config,
)
from minibot.models import FakeModelClient


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
        self.assertEqual(captured[0].headers["Authorization"], "Bearer secret")
        self.assertEqual(request_payload["messages"][0]["content"], "hello")
        self.assertEqual(request_payload["max_tokens"], 32)
        self.assertEqual(request_payload["temperature"], 0.2)
        self.assertEqual(client.last_completion_metadata["api_format"], API_FORMAT_OPENAI)
        self.assertEqual(client.last_completion_metadata["total_tokens"], 7)
        self.assertTrue(client.last_completion_metadata["api_key_present"])
        self.assertNotIn("secret", json.dumps(client.last_completion_metadata))

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
        self.assertEqual(captured[0].headers["x-api-key"], "secret")
        self.assertIn("anthropic-version", captured[0].headers)
        self.assertEqual(request_payload["messages"][0]["content"], "hello")
        self.assertEqual(request_payload["temperature"], 0.1)
        self.assertEqual(client.last_completion_metadata["api_format"], API_FORMAT_ANTHROPIC)
        self.assertEqual(client.last_completion_metadata["total_tokens"], 11)

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
        self.assertTrue(client.last_completion_metadata["api_key_present"])


if __name__ == "__main__":
    unittest.main()
