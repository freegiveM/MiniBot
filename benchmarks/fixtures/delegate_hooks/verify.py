import json
from pathlib import Path

artifact_path = Path(".minibot/delegates/bench_delegate.json")
assert artifact_path.exists()
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
assert artifact["parent_observation"]["schema_valid"] is True
assert "delegate-ready" in artifact["parent_observation"]["summary"]
assert artifact["child"]["read_only"] is True
assert artifact["child"]["allowed_tools"] == ["read_file"]

reports = list(Path(".minibot/runs").glob("*/report.json"))
assert reports
report = json.loads(reports[0].read_text(encoding="utf-8"))
hook_events = [item["event"] for item in report["hooks"]["emissions"]]
assert "PostToolUse" in hook_events
assert "Stop" in hook_events

trace_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(".minibot/runs").glob("*/trace.jsonl"))
assert "delegate_artifact" in trace_text
assert ".minibot/delegates/bench_delegate.json" in trace_text
