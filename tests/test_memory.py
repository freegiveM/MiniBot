from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot.memory import (
    LayeredMemory,
    MemoryCandidate,
    MemoryExtractionEngine,
    MemoryStore,
    MiniLLMMemoryIntentDetector,
    MiniLLMMemorySummarizer,
    build_extraction_payload,
)
from minibot.memory_llm import MemoryLLMExtractionEngine, MemoryModelResolver, MemoryModelSelection
from minibot.model_providers import API_FORMAT_OPENAI, ProviderConfig


class ScriptedMiniLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = []
        self.model = "scripted-mini"

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens, **kwargs})
        return self.outputs.pop(0)


class MainModelMiniLLM(ScriptedMiniLLM):
    def __init__(self, outputs: list[str]):
        super().__init__(outputs)
        self.config = ProviderConfig(
            provider="http",
            api_format=API_FORMAT_OPENAI,
            model_name="main-real-model",
            base_url="https://example.test/chat",
            api_key="secret",
        )


class MemoryTests(unittest.TestCase):
    def test_project_memory_is_stably_injected_but_not_dynamic_relevant_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MemoryStore(root)
            store.write_project_memory("ROOT_RULE: always run tests after edits")
            topics_dir = root / ".minibot" / "memory" / "topics"
            topics_dir.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                (topics_dir / f"topic-{index}.md").write_text(
                    f"alpha topic {index} remembers the runtime testing path",
                    encoding="utf-8",
                )
            memory = LayeredMemory(workspace_root=root)

            project_section = memory.render_memory_index()
            relevant_text, relevant_meta = memory.render_relevant_memory("alpha runtime testing")

            self.assertIn("ROOT_RULE", project_section)
            self.assertNotIn("ROOT_RULE", relevant_text)
            self.assertEqual(relevant_meta["limit"], 3)
            self.assertEqual(relevant_meta["selected_count"], 3)
            self.assertEqual(relevant_meta["selector"], "deterministic_fallback")
            self.assertFalse(relevant_meta["stable_project_memory_included"])

    def test_memory_store_reads_topics_and_dedupes_pending_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MemoryStore(root)
            topic_path = root / ".minibot" / "memory" / "topics" / "debug-notes.md"
            topic_path.parent.mkdir(parents=True)
            topic_path.write_text("Debug note body", encoding="utf-8")
            candidate = MemoryCandidate(
                text="Remember this stable debugging lesson",
                topic="debug-notes",
                tags=["debug"],
                source_ref="run_1",
            )

            first = store.append_pending(candidate)
            second = store.append_pending(candidate)

            self.assertEqual(store.read_topic("debug-notes"), "Debug note body")
            self.assertTrue(first["appended"])
            self.assertFalse(second["appended"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(len(store.load_pending()), 1)

    def test_memory_store_rejects_secret_shaped_pending_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MemoryStore(root)
            candidate = MemoryCandidate(
                text="Remember API key is sk-test-secret-value",
                topic="dependency-facts",
            )

            result = store.append_pending(candidate)

            self.assertFalse(result["appended"])
            self.assertTrue(result["rejected"])
            self.assertEqual(result["rejection_reason"], "secret_shaped")
            self.assertEqual(store.load_pending(), [])

    def test_memory_topic_path_cannot_escape_memory_root(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryStore(Path(temp))
            store.ensure_dirs()

            with self.assertRaises(ValueError):
                store.topic_path("../outside")

    def test_deterministic_memory_extraction_requires_explicit_intent(self):
        payload = build_extraction_payload(user_message="请记住：以后都先跑 context_manager 测试", source_ref="run_1")
        engine = MemoryExtractionEngine()

        candidates, metadata = engine.extract(payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].topic, "user-preferences")
        self.assertTrue(candidates[0].needs_review)
        self.assertIn("以后都", candidates[0].text)
        self.assertEqual(metadata["candidate_count"], 1)
        self.assertTrue(metadata["intent"]["should_extract"])

    def test_deterministic_memory_extraction_skips_without_intent(self):
        payload = build_extraction_payload(user_message="read README and continue")
        engine = MemoryExtractionEngine()

        candidates, metadata = engine.extract(payload)

        self.assertEqual(candidates, [])
        self.assertEqual(metadata["candidate_count"], 0)
        self.assertEqual(metadata["skipped_reason"], "no_explicit_memory_intent")

    def test_mini_llm_memory_interfaces_call_purpose_specific_contracts(self):
        model = ScriptedMiniLLM(
            [
                '{"should_extract": true, "topic": "key-decisions", "tags": ["decision"], "confidence": "high", "reason": "explicit decision"}',
                '{"text": "Use deterministic memory tests before wiring runtime hooks.", "topic": "key-decisions", "tags": ["decision"], "confidence": "high"}',
            ]
        )
        payload = build_extraction_payload(user_message="Decision: keep runtime hook wiring for later", source_ref="run_2")
        engine = MemoryExtractionEngine(
            intent_detector=MiniLLMMemoryIntentDetector(model),
            summarizer=MiniLLMMemorySummarizer(model),
        )

        candidates, metadata = engine.extract(payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].topic, "key-decisions")
        self.assertEqual(candidates[0].extraction_method, "mini_llm")
        self.assertEqual(metadata["intent"]["extraction_method"], "mini_llm")
        self.assertEqual([call["purpose"] for call in model.calls], ["memory_intent", "memory_summary"])
        self.assertTrue(all(call["response_format"] == "json" for call in model.calls))

    def test_memory_llm_engine_generates_candidate_with_traceable_metadata(self):
        model = ScriptedMiniLLM(
            [
                '{"should_extract": true, "topic": "key-decisions", "tags": ["decision"], "confidence": "high", "reason": "keep this", "source_refs": ["run_3"]}',
                '{"text": "Keep provider parsing isolated from runtime.", "topic": "key-decisions", "tags": ["provider"], "confidence": "high", "source_refs": ["run_3"]}',
            ]
        )
        payload = build_extraction_payload(user_message="remember: provider parser stays isolated", source_ref="run_3")
        engine = MemoryLLMExtractionEngine(
            MemoryModelSelection(
                model_client=model,
                selected_memory_model="scripted-mini",
                selection_source="explicit_memory_config",
            )
        )

        candidates, metadata = engine.extract(payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].extraction_method, "mini_llm")
        self.assertEqual(candidates[0].metadata["source_refs"], ["run_3"])
        self.assertTrue(candidates[0].metadata["schema_valid"])
        self.assertTrue(metadata["mini_llm_used"])
        self.assertEqual(metadata["selected_memory_model"], "scripted-mini")
        self.assertEqual(metadata["memory_model_selection_source"], "explicit_memory_config")
        self.assertEqual(metadata["extraction_success_count"], 1)
        self.assertEqual(metadata["summary_method"], "mini_llm")

    def test_memory_llm_bad_summary_falls_back_to_direct_clip(self):
        model = ScriptedMiniLLM(
            [
                '{"should_extract": true, "topic": "user-preferences", "tags": ["preference"], "confidence": "medium", "reason": "explicit", "source_refs": []}',
                '{"topic": "user-preferences"}',
            ]
        )
        payload = build_extraction_payload(user_message="remember: always keep summaries bounded", source_ref="run_4")
        engine = MemoryLLMExtractionEngine(
            MemoryModelSelection(model_client=model, selected_memory_model="scripted-mini", selection_source="agent_override")
        )

        candidates, metadata = engine.extract(payload)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].extraction_method, "deterministic_fallback")
        self.assertIn("always keep summaries bounded", candidates[0].text)
        self.assertEqual(candidates[0].metadata["summary_method"], "direct_clip")
        self.assertEqual(metadata["schema_error_count"], 1)
        self.assertEqual(metadata["summary_method"], "direct_clip")
        self.assertEqual(metadata["fallback_reason"], "mini_llm_summary_failed")

    def test_memory_model_resolver_explicit_config_precedes_candidate_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "MINIBOT_MEMORY_MODEL_PROVIDER=http",
                        "MINIBOT_MEMORY_API_FORMAT=openai",
                        "MINIBOT_MEMORY_MODEL_NAME=memory-explicit",
                        "MINIBOT_MEMORY_BASE_URL=https://memory.test/chat",
                        "MINIBOT_MEMORY_API_KEY=memory-key",
                        "MINIBOT_MEMORY_MODEL_CANDIDATES=deepseek-v4-pro",
                        "MINIBOT_MEMORY_DEEPSEEK_V4_PRO_API_KEY=deepseek-key",
                    ]
                ),
                encoding="utf-8",
            )

            selection = MemoryModelResolver(root).resolve()

            self.assertTrue(selection.has_model)
            self.assertEqual(selection.selected_memory_model, "memory-explicit")
            self.assertEqual(selection.selection_source, "explicit_memory_config")
            self.assertEqual(selection.provider_metadata["base_url"], "https://memory.test/chat")
            self.assertTrue(selection.provider_metadata["api_key_present"])

    def test_memory_model_resolver_falls_back_to_current_main_model(self):
        with tempfile.TemporaryDirectory() as temp:
            model = MainModelMiniLLM(
                [
                    '{"should_extract": true, "topic": "key-decisions", "tags": ["decision"], "confidence": "high", "reason": "main model", "source_refs": []}',
                    '{"text": "Use the current main model when memory profiles are unavailable.", "topic": "key-decisions", "tags": ["decision"], "confidence": "high"}',
                ]
            )

            selection = MemoryModelResolver(temp, environ={}).resolve(current_model_client=model)

            self.assertTrue(selection.has_model)
            self.assertEqual(selection.selected_memory_model, "main-real-model")
            self.assertEqual(selection.selection_source, "main_model")
            self.assertIn("candidate_unavailable:sonnet", selection.fallback_chain)

    def test_memory_model_resolver_without_models_uses_deterministic_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            selection = MemoryModelResolver(temp, environ={}).resolve()

            self.assertFalse(selection.has_model)
            self.assertEqual(selection.selection_source, "deterministic_fallback")
            self.assertIn("deterministic_fallback_selected", selection.fallback_chain)


if __name__ == "__main__":
    unittest.main()
