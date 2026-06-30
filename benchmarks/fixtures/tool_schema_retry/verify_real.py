from pathlib import Path

text = Path("config.txt").read_text(encoding="utf-8")
assert "mode=new" in text
assert "mode=old" not in text
assert "untouched=yes" in text
