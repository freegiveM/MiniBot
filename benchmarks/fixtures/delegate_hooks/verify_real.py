import json
from pathlib import Path

artifacts = list(Path(".minibot/delegates").glob("*.json"))
assert artifacts
artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
assert artifact["parent_observation"]["schema_valid"] is True
assert "delegate-ready" in artifact["parent_observation"]["summary"]
assert artifact["child"]["read_only"] is True
assert "read_file" in artifact["child"]["allowed_tools"]
assert "patch_file" not in artifact["child"]["allowed_tools"]
assert "write_file" not in artifact["child"]["allowed_tools"]
assert "run_shell" not in artifact["child"]["allowed_tools"]
assert "delegate" not in artifact["child"]["allowed_tools"]

reports = list(Path(".minibot/runs").glob("*/report.json"))
assert reports
report = json.loads(reports[0].read_text(encoding="utf-8"))
artifact_refs = report.get("delegate_artifacts", [])
assert artifact_refs
assert any(Path(ref).name == artifacts[0].name for ref in artifact_refs)

trace_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(".minibot/runs").glob("*/trace.jsonl"))
assert "delegate_artifact" in trace_text
assert artifacts[0].name in trace_text
