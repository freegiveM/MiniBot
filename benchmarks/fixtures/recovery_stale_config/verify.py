import json
from pathlib import Path

text = Path("settings.ini").read_text(encoding="utf-8")
assert "flag=first" in text
assert "mode=active" in text
assert "mode=passive" not in text

reports = list(Path(".minibot/runs").glob("*/report.json"))
assert reports
report = json.loads(reports[0].read_text(encoding="utf-8"))
kinds = [event.get("kind") for event in report.get("recovery", {}).get("events", [])]
assert "stale_file_state" in kinds

