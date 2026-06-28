from pathlib import Path

text = Path("settings.ini").read_text(encoding="utf-8")
assert "flag=first" in text
assert "mode=active" in text
assert "mode=passive" not in text
