# MiniBot Benchmark Methodology

This document explains how MiniBot evaluates agent behavior without turning the project into a large benchmark clone. The goal is a small, reproducible, executable benchmark suite that can separate harness reliability from real-model behavior.

## Goals

Stage 20 turns the benchmark work into a method that is easy to explain, rerun, and defend:

- Use deterministic mock tasks to prove fixture, tool, verifier, recovery, memory, and metrics wiring.
- Use explicit real-model smoke runs to test provider integration and real model decision quality.
- Keep pass/fail grounded in executable verifiers and trace/report evidence, not model self-judgment.
- Preserve failure examples and failure taxonomy so regressions are explainable.
- Define which numbers are resume-safe and which are only exploratory.

## References And MiniBot Choices

| Reference | Useful idea | MiniBot choice |
| --- | --- | --- |
| HumanEval / EvalPlus | Functional correctness should be checked by executable tests. | Code tasks use unit tests or import checks, while text tasks use file assertions. |
| SWE-bench | A patch should be judged against repository state after the agent acts. | Small fresh fixtures simulate repository issues without heavyweight environments. |
| AgentBench | Agent evaluation needs interaction traces, not only final answers. | MiniBot records task_state, trace.jsonl, report.json, prompt metadata, and tool events. |
| GAIA | Short tasks can still require tool use and evidence. | MiniBot avoids pure chat tasks; tasks require local files, tools, memory, recovery, or delegation. |
| WebArena / OSWorld | Final environment state and action history both matter. | MiniBot combines verifier result, expected artifacts, stop reason, budget use, and trace evidence. |

MiniBot does not claim scale equivalence with these benchmarks. It borrows evaluation principles and applies them to a local coding-agent harness.

## Task Levels

MiniBot tasks are grouped by mechanism rather than by difficulty alone:

| Level | Task type | Primary signal |
| --- | --- | --- |
| L0 | Single-step text or documentation edit | File assertion and non-target text preservation |
| L1 | Bounded code modification | Unit test, import check, or focused executable verifier |
| L2 | Tool-boundary or recovery task | Trace/report evidence for permission denial, retry, reread, or stop reason |
| L3 | Memory or miniLLM extraction task | Memory hit, pending memory candidate, schema reliability, and fallback evidence |
| L4 | Delegate or multi-step analysis task | Bounded child artifact and parent observation quality |

The MVP benchmark keeps tasks short. That makes failures easier to inspect and prevents long-running environment setup from dominating the evaluation.

## Mock Vs Real Artifacts

MiniBot keeps deterministic and real-model artifacts separate.

| Artifact type | Default path | Purpose |
| --- | --- | --- |
| Mock harness | `artifacts/harness-regression-v2.json` | Regression signal for benchmark harness, fixtures, tools, verifiers, recovery, memory hooks, and metrics |
| Real harness | `artifacts/harness-real-v1.json` | Opt-in smoke signal for real provider behavior, tool-call protocol, latency, and provider failure taxonomy |
| Core report | `artifacts/minibot-benchmark-core-report.md` | Summary of one harness artifact plus ablation artifacts |
| Methodology report | `artifacts/minibot-benchmark-methodology-report.md` | Mock-vs-real comparison, failure examples, claim boundaries, and methodology notes |

Mock results answer: "Does the evaluation harness work?"

Real results answer: "Can a real model drive this harness under explicit limits?"

They should not be merged into one score.

## Scoring Rules

Each benchmark row is judged from structured evidence:

- `passed`: final combined row result.
- `verifier_passed`: executable verifier success.
- `within_budget`: tool steps stayed within task budget.
- `expected_artifact_exists`: expected file or artifact exists.
- `stop_reason`: final runtime stop reason.
- `failure_category`: missing artifact, budget exceeded, verifier failed, provider error, tool error, stop reason, or unknown.

Executable verifiers are preferred:

- Text tasks assert target text changed and guardrail text did not change.
- Code tasks run focused tests or imports.
- Tool-boundary and recovery tasks inspect trace/report evidence.
- Memory tasks inspect memory-related tool use, pending candidates, and hook metadata.

LLM-as-judge is not a primary pass/fail mechanism in this stage.

## miniLLM Memory Evaluation

Stage 17 added miniLLM memory extraction as a subchain. Stage 20 reports it separately from task pass rate.

The methodology report tracks:

- `mini_llm_schema_error_rate`: schema or parsing errors divided by extraction attempts, including cases that fall back to deterministic extraction.
- `memory_extraction_accept_rate`: accepted pending candidates divided by candidates.
- `extraction_attempt_count`: number of memory extraction attempts found in hook metadata.
- `candidate_count`: pending memory candidates produced.
- `accepted_count`: candidates appended to the pending store.
- `deterministic_fallback_count`: fallback use when miniLLM output is unavailable or invalid.

This is a subchain metric. It should not be used as a standalone claim that memory quality is solved.

## Real-Model Guardrails

Real benchmark mode is explicit opt-in:

```powershell
$env:PYTHONPATH="src"
python -m minibot benchmark --real --max-tasks 5 --artifact-path artifacts/harness-real-v1.json
```

Recommended real smoke constraints:

- Use low temperature, usually `0.0`.
- Use `--max-tasks` for a 3 to 5 task smoke slice.
- Use provider timeout, max tokens, and cost guard options.
- Keep API keys in `.env` or process environment only.
- Do not commit `.env` or real provider secrets.

Real runs should record provider metadata, request endpoint, response shape, latency, cost fields when available, and failure category.

## Metrics And Reports

Generate the deterministic harness:

```powershell
$env:PYTHONPATH="src"
python -m minibot benchmark --artifact-path artifacts/harness-regression-v2.json
```

Generate the core report:

```powershell
$env:PYTHONPATH="src"
python -m minibot metrics --harness-artifact-path artifacts/harness-regression-v2.json --report-path artifacts/minibot-benchmark-core-report.md
```

Generate the methodology report:

```powershell
$env:PYTHONPATH="src"
python -m minibot metrics --methodology-report --harness-artifact-path artifacts/harness-regression-v2.json --real-harness-artifact-path artifacts/harness-real-v1.json --report-path artifacts/minibot-benchmark-methodology-report.md
```

The methodology report tolerates a missing real artifact and marks it as `missing` instead of treating it as a failed run.

## Claim Boundaries

Resume-safe claims:

- Number of deterministic fixture tasks and pass rate, when backed by committed harness design and reproducible artifacts.
- Verifier pass rate, within-budget rate, median tool steps, category pass rates, and failure taxonomy counts.
- Tool-boundary rejection counts when backed by trace/report permission evidence.

Good interview discussion claims:

- Why deterministic tasks test harness reliability rather than model quality.
- How real smoke tests surfaced provider endpoint, thinking-block, and tool-call prompt-contract issues.
- How miniLLM memory extraction is tracked through schema reliability, pending candidates, and fallback behavior.

Do not overclaim:

- Do not compare MiniBot scores directly to SWE-bench, GAIA, WebArena, or OSWorld.
- Do not use one real run as a general model-quality claim.
- Do not report missing artifacts as zero performance.
- Do not use LLM self-judgment as primary pass/fail evidence.

## Current Limitations

- The fixture suite is intentionally small.
- Real-model smoke runs depend on provider behavior and account configuration.
- Cost can be estimated only when provider metadata exposes enough information.
- Memory quality is evaluated through bounded signals, not through long-term user studies.
- The benchmark does not cover browser or operating-system automation.
