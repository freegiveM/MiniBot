from pathlib import Path

text = Path("sample.txt").read_text(encoding="utf-8")
assert "alpha-guarded" in text
assert "control=unchanged" in text

trace_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(".minibot/runs").glob("*/trace.jsonl"))
assert "tool_rejected" in trace_text
assert "path_escape" in trace_text

