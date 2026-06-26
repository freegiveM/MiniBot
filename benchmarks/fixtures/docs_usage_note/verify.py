from pathlib import Path

text = Path("docs/USAGE.md").read_text(encoding="utf-8")
assert "Before: run minibot with a task." in text
assert "Note: Keep benchmark artifacts attached to reports." in text
assert "Note: TODO" not in text
assert "After: review artifacts after every run." in text

