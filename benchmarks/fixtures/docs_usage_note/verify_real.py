from pathlib import Path

text = Path("docs/USAGE.md").read_text(encoding="utf-8")
lower = text.lower()
assert "Before: run minibot with a task." in text
assert "Note: TODO" not in text
assert "note:" in lower
assert "benchmark" in lower
assert "artifact" in lower
assert "report" in lower
assert "After: review artifacts after every run." in text
