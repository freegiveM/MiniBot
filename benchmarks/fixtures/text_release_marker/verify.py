from pathlib import Path

text = Path("notes.txt").read_text(encoding="utf-8")
assert "alpha marker" in text
assert "release: ready" in text
assert "release: pending" not in text
assert "omega marker" in text

