from pathlib import Path

text = Path("sample.txt").read_text(encoding="utf-8")
assert "header=keep" in text
assert "priority=high" in text
assert "priority=low" not in text
assert "footer=keep" in text

