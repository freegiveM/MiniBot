from pathlib import Path

text = Path("README.md").read_text(encoding="utf-8")
assert "Status: reviewed" in text
assert "Status: draft" not in text
assert "Guardrail: preserve this setup note." in text
assert "Do not edit: API keys stay out of docs." in text

