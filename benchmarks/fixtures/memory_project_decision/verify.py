from pathlib import Path

text = Path("notes/decision.txt").read_text(encoding="utf-8")
assert "decision=alpha-retained" in text
assert "decision=pending" not in text
assert "non_target=keep" in text

trace_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(".minibot/runs").glob("*/trace.jsonl"))
assert '"name": "read_memory"' in trace_text or '"name":"read_memory"' in trace_text
assert "alpha-retained" in trace_text

