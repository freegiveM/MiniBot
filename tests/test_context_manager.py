from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minibot import FakeModelClient, MiniBot, SessionStore, WorkspaceContext
from minibot.context_manager import SECTION_ORDER, ContextManager
from minibot.task_state import TaskState


class ContextManagerTests(unittest.TestCase):
    def build_agent(self, root: Path, outputs: list[str] | None = None) -> MiniBot:
        workspace = WorkspaceContext.build(root)
        return MiniBot(
            model_client=FakeModelClient(outputs or []),
            workspace=workspace,
            session_store=SessionStore(root / ".minibot" / "sessions"),
            approval_policy="auto",
        )

    def test_prompt_order_and_current_request_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            context = ContextManager(agent, total_budget=800)
            request = "keep this exact request\nwith trailing spaces   "

            prompt, metadata = context.build_prompt(request)

            self.assertEqual(metadata["section_order"], list(SECTION_ORDER))
            self.assertEqual(metadata["section_order"][-1], "current_request")
            self.assertTrue(prompt.endswith("Current user request:\n" + request))
            positions = [prompt.index(self._section_heading(name)) for name in SECTION_ORDER]
            self.assertEqual(positions, sorted(positions))
            self.assertFalse(metadata["sections"]["current_request"]["truncated"])
            self.assertTrue(metadata["sections"]["current_request"]["preserved"])

    def test_section_metadata_explains_source_size_and_truncation_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            _, metadata = agent.context_manager.build("inspect")

            self.assertEqual(set(metadata["sections"]), set(SECTION_ORDER))
            for name in SECTION_ORDER:
                section = metadata["sections"][name]
                self.assertIsInstance(section["source"], str)
                self.assertGreaterEqual(section["chars"], 0)
                self.assertGreaterEqual(section["raw_chars"], section["chars"])
                self.assertFalse(section["truncated"])
                self.assertEqual(section["truncation_reason"], "")
                self.assertIn(section["cache_class"], {"stable_prefix", "dynamic"})
            self.assertEqual(metadata["sections"]["identity"]["cache_class"], "stable_prefix")
            self.assertEqual(metadata["sections"]["history"]["cache_class"], "dynamic")
            self.assertTrue(metadata["prompt_cache_key"].startswith("minibot:v1:"))
            self.assertEqual(metadata["cacheable_sections"], ["identity", "workspace", "tools", "memory_index"])
            self.assertIn("history", metadata["dynamic_sections"])
            self.assertIn("stable_prefix_hash", metadata["prompt_cache"])

    def test_prompt_cache_key_ignores_current_request_history_and_working_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            _, first = agent.context_manager.build("first request")
            agent.session["history"] = [{"role": "assistant", "content": "different history"}]
            agent.memory.remember_tool("read_file", status="succeeded", path="README.md")
            agent.session["memory"] = agent.memory.to_dict()
            _, second = agent.context_manager.build("second request with new text")

            self.assertEqual(first["stable_prefix_hash"], second["stable_prefix_hash"])
            self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])

    def test_identity_section_spells_out_tool_call_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)

            prompt, _ = agent.context_manager.build("inspect")

            self.assertIn('<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":40}}</tool>', prompt)
            self.assertIn('tool_batch_example: <tool>[{"name":"read_file"', prompt)
            self.assertIn('every call needs "name" and "args"', prompt)
            self.assertIn("schemas describe the args object", prompt)
            self.assertIn("args_schema=", prompt)
            self.assertIn("example_args=", prompt)

    def test_section_budget_can_truncate_non_current_request_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [{"role": "assistant", "content": "x" * 200}]
            context = ContextManager(agent, section_budgets={"history": 40, "current_request": 5})
            request = "current request must stay whole"

            prompt, metadata = context.build_prompt(request)

            history = metadata["sections"]["history"]
            current = metadata["sections"]["current_request"]
            self.assertTrue(history["truncated"])
            self.assertEqual(history["truncation_reason"], "section_budget_exceeded")
            self.assertLessEqual(history["chars"], 40)
            self.assertFalse(current["truncated"])
            self.assertTrue(prompt.endswith("Current user request:\n" + request))

    def test_task_state_section_renders_current_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            state = TaskState.create(task_id="task_1", user_request="inspect", run_id="run_1")
            state.record_attempt()
            state.record_tool("read_file")
            agent.current_task_state = state

            prompt, metadata = agent.context_manager.build("continue")

            self.assertIn("Task state:\n- run_id: run_1", prompt)
            self.assertIn("- task_id: task_1", prompt)
            self.assertIn("- status: running", prompt)
            self.assertIn("- tool_steps: 1", prompt)
            self.assertIn("- attempts: 1", prompt)
            self.assertIn("- last_tool: read_file", prompt)
            self.assertEqual(metadata["sections"]["task_state"]["source"], "runtime.current_task_state")

    def test_context_compaction_preserves_current_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [
                {"role": "assistant", "content": f"older context {index} " + ("x" * 500)}
                for index in range(8)
            ]
            request = "latest request must remain exactly"
            context = ContextManager(agent, total_budget=1200)

            prompt, metadata = context.build_prompt(request)

            self.assertTrue(prompt.endswith("Current user request:\n" + request))
            self.assertTrue(metadata["current_request_preserved"])
            self.assertGreater(metadata["budget_reduction_count"], 0)
            self.assertEqual(metadata["compact_trigger"], "prompt_budget_exceeded")
            self.assertFalse(metadata["sections"]["identity"]["truncated"])
            self.assertFalse(metadata["sections"]["tools"]["truncated"])
            self.assertFalse(metadata["sections"]["task_state"]["truncated"])
            self.assertFalse(metadata["sections"]["current_request"]["truncated"])

    def test_history_trimming_prefers_old_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [
                {"role": "assistant", "content": "OLD_MARKER_0 " + ("x" * 300)},
                {"role": "assistant", "content": "OLD_MARKER_1 " + ("x" * 300)},
                {"role": "assistant", "content": "OLD_MARKER_2 " + ("x" * 300)},
                {"role": "assistant", "content": "NEW_MARKER_KEEP"},
            ]
            _, raw_metadata = ContextManager(agent, total_budget=None).build_prompt("continue")
            budget = raw_metadata["raw_prompt_chars"] - raw_metadata["sections"]["history"]["chars"] + 260

            prompt, metadata = ContextManager(agent, total_budget=budget).build_prompt("continue")

            self.assertNotIn("OLD_MARKER_0", prompt)
            self.assertIn("NEW_MARKER_KEEP", prompt)
            history_event = metadata["compact_summary"]["events"][0]
            self.assertEqual(history_event["section"], "history")
            self.assertGreater(history_event["trimmed_history_items"], 0)

    def test_duplicate_read_observations_are_collapsed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [
                {
                    "role": "tool",
                    "name": "read_file",
                    "args": {"path": "app.py", "start": 1, "end": 50},
                    "content": "OLD_DUPLICATE_READ_CONTENT " + ("x" * 900),
                    "metadata": {"artifact_ref": "runs/run_old/trace.jsonl", "content_chars": 925},
                },
                {
                    "role": "tool",
                    "name": "read_file",
                    "args": {"path": "app.py", "start": 1, "end": 80},
                    "content": "LATEST_READ_CONTENT",
                    "metadata": {"artifact_ref": "runs/run_new/trace.jsonl", "content_chars": 19},
                },
            ]
            _, raw_metadata = ContextManager(agent, total_budget=None).build_prompt("continue")
            budget = raw_metadata["raw_prompt_chars"] - raw_metadata["sections"]["history"]["chars"] + 900

            prompt, metadata = ContextManager(agent, total_budget=budget).build_prompt("continue")

            self.assertIn("duplicate read_file collapsed for app.py", prompt)
            self.assertIn("LATEST_READ_CONTENT", prompt)
            self.assertNotIn("OLD_DUPLICATE_READ_CONTENT", prompt)
            self.assertIn("runs/run_old/trace.jsonl", prompt)
            history_event = metadata["compact_summary"]["events"][0]
            self.assertGreater(history_event["collapsed_duplicate_reads"], 0)

    def test_tool_observation_compaction_keeps_artifact_ref(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [
                {
                    "role": "tool",
                    "name": "search",
                    "args": {"query": "needle"},
                    "content": "SEARCH_PREVIEW " + ("x" * 1000) + " TAIL_SHOULD_COMPACT",
                    "metadata": {
                        "artifact_ref": "runs/run_1/trace.jsonl",
                        "content_chars": 2048,
                        "tool_status": "succeeded",
                        "truncated": True,
                    },
                }
            ]
            _, raw_metadata = ContextManager(agent, total_budget=None).build_prompt("continue")
            budget = raw_metadata["raw_prompt_chars"] - raw_metadata["sections"]["history"]["chars"] + 900

            prompt, metadata = ContextManager(agent, total_budget=budget).build_prompt("continue")

            self.assertIn("Observation summary:", prompt)
            self.assertIn("SEARCH_PREVIEW", prompt)
            self.assertNotIn("TAIL_SHOULD_COMPACT", prompt)
            self.assertIn("runs/run_1/trace.jsonl", prompt)
            self.assertIn("- original_chars: 2048", prompt)
            history_event = metadata["compact_summary"]["events"][0]
            self.assertGreater(history_event["summarized_tool_observations"], 0)

    def test_compact_summary_records_trigger_sections_and_reduction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root)
            agent.session["history"] = [{"role": "assistant", "content": "x" * 2000}]

            _, metadata = ContextManager(agent, total_budget=1200).build_prompt("continue")

            summary = metadata["compact_summary"]
            self.assertEqual(summary["trigger"], "prompt_budget_exceeded")
            self.assertGreater(summary["raw_prompt_chars"], summary["final_prompt_chars"])
            self.assertGreater(summary["reduced_chars"], 0)
            self.assertTrue(summary["current_request_preserved"])
            self.assertFalse(summary["summarizer"]["used"])
            self.assertEqual(metadata["budget_reduction_count"], len(summary["events"]))
            self.assertTrue(any(event["section"] == "history" for event in summary["events"]))

    def test_compaction_trims_relevant_memory_before_memory_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            memory_dir = root / ".minibot" / "memory"
            memory_dir.mkdir(parents=True)
            (memory_dir / "MEMORY.md").write_text("INDEX_MARKER " + ("i" * 1000), encoding="utf-8")
            agent = self.build_agent(root)
            for index in range(3):
                agent.memory.append_note(
                    "signal relevant memory note " + str(index) + " " + ("r" * 450),
                    tags=("signal",),
                    topic="task-experience",
                )
            _, raw_metadata = ContextManager(agent, total_budget=None).build_prompt("signal")
            budget = raw_metadata["raw_prompt_chars"] - raw_metadata["sections"]["relevant_memory"]["chars"] + 700

            prompt, metadata = ContextManager(agent, total_budget=budget).build_prompt("signal")

            self.assertEqual(metadata["compact_summary"]["events"][0]["section"], "relevant_memory")
            self.assertTrue(metadata["sections"]["relevant_memory"]["truncated"])
            self.assertFalse(metadata["sections"]["memory_index"]["truncated"])
            self.assertIn("INDEX_MARKER", prompt)

    def test_prompt_metadata_reaches_trace_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            agent = self.build_agent(root, ["<final>Done.</final>"])

            self.assertEqual(agent.ask("hello"), "Done.")

            run_id = agent.session["runs"]["last_run_id"]
            run_dir = root / ".minibot" / "runs" / run_id
            events = [
                json.loads(line)
                for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            prompt_events = [event for event in events if event["event"] == "prompt_built"]
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_events[0]["prompt_metadata"]["section_order"], list(SECTION_ORDER))
            self.assertEqual(report["prompt_metadata"]["section_order"], list(SECTION_ORDER))
            self.assertIn("task_state", report["prompt_metadata"]["sections"])

    @staticmethod
    def _section_heading(name: str) -> str:
        return {
            "identity": "Identity:",
            "workspace": "Workspace:",
            "tools": "Tools:",
            "task_state": "Task state:",
            "working_memory": "Working memory:",
            "relevant_memory": "Relevant memory:",
            "memory_index": "Memory index:",
            "history": "Transcript:",
            "current_request": "Current user request:",
        }[name]


if __name__ == "__main__":
    unittest.main()
