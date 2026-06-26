from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.prompt_cache import (
    CACHEABLE_PREFIX_SECTIONS,
    DYNAMIC_SECTIONS,
    PROMPT_CACHE_OPENAI_EXPLICIT,
    build_prompt_cache_plan,
    normalize_prompt_cache_retention,
    provider_supports_prompt_cache,
)


def _sections(**overrides):
    values = {
        "identity": "Identity:\n- MiniBot",
        "workspace": "Workspace:\n- stable summary",
        "tools": "Tools:\n- read_file",
        "task_state": "Task state:\n- run_1",
        "working_memory": "Working memory:\n- recent_files: a.py",
        "relevant_memory": "Relevant memory:\n- none",
        "memory_index": "Memory index:\n- none",
        "history": "Transcript:\n[user] old request",
        "current_request": "Current user request:\nnew request",
    }
    values.update(overrides)
    return [SimpleNamespace(name=name, text=text) for name, text in values.items()]


def _plan(**kwargs):
    defaults = {
        "sections": _sections(),
        "workspace_fingerprint": "workspace-a",
        "tool_signature": "tools-a",
        "provider": "openai",
        "api_format": "openai",
        "model_name": "gpt-mini",
        "prompt_cache": "auto",
        "retention": "in-memory",
    }
    defaults.update(kwargs)
    return build_prompt_cache_plan(**defaults)


class PromptCacheTests(unittest.TestCase):
    def test_stable_prefix_hash_ignores_current_request_history_and_working_memory(self):
        baseline = _plan()
        changed_dynamic = _plan(
            sections=_sections(
                task_state="Task state:\n- run_2",
                working_memory="Working memory:\n- recent_files: secret.py",
                relevant_memory="Relevant memory:\n- changing",
                history="Transcript:\n[user] totally different",
                current_request="Current user request:\nplease do something else",
            )
        )

        self.assertEqual(baseline.stable_prefix_hash, changed_dynamic.stable_prefix_hash)
        self.assertEqual(baseline.prompt_cache_key, changed_dynamic.prompt_cache_key)
        self.assertEqual(baseline.cacheable_sections, CACHEABLE_PREFIX_SECTIONS)
        self.assertEqual(baseline.dynamic_sections, DYNAMIC_SECTIONS)

    def test_stable_prefix_hash_changes_when_tool_signature_workspace_memory_or_model_changes(self):
        baseline = _plan()
        variants = [
            _plan(workspace_fingerprint="workspace-b"),
            _plan(tool_signature="tools-b"),
            _plan(sections=_sections(memory_index="Memory index:\n- changed")),
            _plan(model_name="gpt-other"),
        ]

        for variant in variants:
            self.assertNotEqual(baseline.stable_prefix_hash, variant.stable_prefix_hash)
            self.assertNotEqual(baseline.prompt_cache_key, variant.prompt_cache_key)

    def test_provider_capability_modes_are_explicit(self):
        self.assertTrue(provider_supports_prompt_cache(provider="openai", api_format="openai", prompt_cache="auto"))
        self.assertTrue(
            provider_supports_prompt_cache(provider="http", api_format="openai", prompt_cache=PROMPT_CACHE_OPENAI_EXPLICIT)
        )
        self.assertFalse(provider_supports_prompt_cache(provider="http", api_format="openai", prompt_cache="auto"))
        self.assertFalse(provider_supports_prompt_cache(provider="anthropic", api_format="anthropic", prompt_cache="auto"))

    def test_openai_plan_is_eligible_but_fake_plan_only_records_key(self):
        openai_plan = _plan()
        fake_plan = _plan(provider="fake", model_name="fake", prompt_cache="auto")

        self.assertTrue(openai_plan.eligible)
        self.assertEqual(openai_plan.kwargs_for_provider()["prompt_cache_retention"], "in-memory")
        self.assertFalse(fake_plan.eligible)
        self.assertTrue(fake_plan.prompt_cache_key.startswith("minibot:v1:"))
        self.assertIn("provider_not_openai_cache_capable", fake_plan.invalidation_reasons[0])

    def test_retention_rejects_old_underscore_spelling(self):
        self.assertEqual(normalize_prompt_cache_retention("in-memory"), "in-memory")
        with self.assertRaises(ValueError):
            normalize_prompt_cache_retention("in_memory")


if __name__ == "__main__":
    unittest.main()
