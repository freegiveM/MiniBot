from pathlib import Path

text = Path("sample.txt").read_text(encoding="utf-8")
assert "alpha-guarded" in text
assert "alpha\n" not in text
assert "control=unchanged" in text

trace_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(".minibot/runs").glob("*/trace.jsonl"))
assert "../outside.txt" not in trace_text
assert "..\\outside.txt" not in trace_text
