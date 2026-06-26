import json
from pathlib import Path

text = Path("context.txt").read_text(encoding="utf-8")
assert "marker=new" in text
assert "stable=line" in text

reports = list(Path(".minibot/runs").glob("*/report.json"))
assert reports
report = json.loads(reports[0].read_text(encoding="utf-8"))
metadata = report["prompt_metadata"]
assert metadata["compact_trigger"] == "prompt_budget_exceeded"
assert metadata["budget_reduction_count"] > 0
assert metadata["current_request_preserved"] is True
assert "CONTEXT_REQUEST_MARKER" in metadata["current_request"]["text"]

