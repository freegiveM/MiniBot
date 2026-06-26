import json
from pathlib import Path

text = Path("config.txt").read_text(encoding="utf-8")
assert "mode=new" in text
assert "mode=old" not in text
assert "untouched=yes" in text

reports = list(Path(".minibot/runs").glob("*/report.json"))
assert reports
report = json.loads(reports[0].read_text(encoding="utf-8"))
kinds = [event.get("kind") for event in report.get("recovery", {}).get("events", [])]
assert "tool_schema_error" in kinds

