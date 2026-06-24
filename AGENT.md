# MiniBot Agent Constraints

This document is the project contract for the MiniBot refactor. Treat the local
`pico/` directory as a reference implementation only. It is ignored by Git and
must not be committed as project source.

## Project Goal

MiniBot is a lightweight local coding agent for repository inspection,
modification, debugging, review, session-based resume, memory-assisted context,
and regression evaluation.

The refactor should keep Pico's useful chain:

```text
user request -> prompt build -> model decision -> tool execution ->
history/trace/task_state/report -> session persistence
```

But the memory design must follow the v2 optimization plan:

```text
accuracy first, source facts from source files, memory as guidance
```

## Non-Negotiable Rules

1. Do not commit or copy the `pico/` directory as the production package.
2. Do not store source-code summaries as trusted memory by default.
3. When exact code facts are needed, read the source file again.
4. `Relevant memory` is a per-prompt temporary view. Do not persist it in
   session state.
5. `tag/topic` is not the primary recall signal. Use it only for memory index
   navigation, filtering, and re-ranking.
6. Keep the runtime loop explicit. Model calls, tool execution, history writes,
   final responses, and safety checks must stay on the main path.
7. Use hooks only for side effects such as trace metadata, file access updates,
   candidate memory generation, and maintenance. Hooks must not be the source of
   truth for run lifecycle state.
8. A normal `read_file` result may update file access metadata, but must not
   create durable memory, candidate memory, or episodic notes by default.
9. Local runtime artifacts stay under `.minibot/` or `.pico/` style state
   directories and must remain ignored by Git.
10. Every feature that changes agent behavior needs a focused test.
11. After each refactor stage, create or update one review note under `Note/`
    before asking for acceptance.
12. Add concise code comments for non-obvious design decisions, runtime
    boundaries, safety checks, and test intent. Do not add mechanical comments
    that merely restate the code.

## Stage Review Notes

Each accepted or review-ready stage must have a matching Markdown note:

```text
Note/
  00-boundary-and-baseline.md
  01-run-store.md
  02-task-state.md
  ...
```

The note is for later interview review and must cover:

- background and goal
- module design idea
- where the module sits in the overall agent framework
- runtime flow, if the stage affects execution
- design highlights and tradeoffs
- tests and acceptance method
- future extension points
- differences from mature commercial coding agents such as Claude Code or Codex,
  using conservative wording and official/current docs when concrete product
  claims matter
- interview talking points

When a stage only changes project boundaries or documents, still write a short
note explaining why the boundary matters. Avoid copying external note structure
verbatim; keep MiniBot's own structure and language.

## Target Storage Shape

```text
.minibot/
  sessions/
    <session_id>.json
  runs/
    <run_id>/
      task_state.json
      trace.jsonl
      report.json
  memory/
    MEMORY.md
    pending.jsonl
    topics/
      project-context.md
      user-preferences.md
      key-decisions.md
      dependency-facts.md
      task-experience.md
      debug-notes.md
  delegates/
    <delegate_task_id>.json
```

## Session Contract

Session state is a recovery container, not a knowledge base.
MVP resume loads session state and injects bounded history, working memory,
memory index, and workspace context. Do not add checkpoint state in the MVP;
snapshot-based recovery must be designed later as a context-compaction feature
with its own schema and tests.

State ownership is split deliberately:

- `SessionStore` is the resume source of truth. `MiniBot.from_session(...)` must
  be able to rebuild prompt context from session data without reading previous
  run artifacts.
- `RunStore` is the evaluation and audit source of truth. Evaluators should read
  `runs/<run_id>/task_state.json`, `trace.jsonl`, and `report.json`, not session
  history.
- Session `runs` fields are navigation pointers only, such as last/recent run
  ids. They must not become hidden recovery payloads.

Allowed session fields:

- `id`, `schema_version`, `created_at`, `updated_at`
- `workspace_root`
- `runtime_identity`
- `turn_count`
- `history`
- `memory.working`
- `memory.file_access`
- `memory.episodic_notes`
- `memory.durable_cache`
- `runs`
- `memory_maintenance`
- `pending_delegates`

Disallowed session fields:

- persisted `relevant_memory`
- source `file_summaries` that are used as code facts
- unbounded raw tool output
- pending candidate bodies embedded directly in session

## Memory Contract

Working memory should contain:

- initial request summary
- current task summary
- constraints
- recent files
- recent tools
- open questions

File access metadata should contain:

- canonical path
- last read timestamp
- read ranges
- freshness hash
- symbols seen when cheaply extractable
- trace reference
- status

Episodic notes should contain reusable process facts only:

- tool failures
- debugging conclusions
- delegate findings
- user-confirmed preferences
- key design decisions
- task completion learnings

Candidate memory belongs in `memory/pending.jsonl`, not in session.

## Retrieval Contract

Relevant memory should follow the learn-claude-code style design:

- keep a compact `MEMORY.md` index in prompt
- build a memory catalog from topic name and short description
- ask a side-query model step to select at most five relevant memory topics or
  entries for the current turn
- inject only bounded excerpts into the prompt
- fall back to deterministic keyword/path matching when the side-query fails

Do not make hand-tuned weights the primary retrieval mechanism. Path, symbol,
keyword, tag, recency, and hit-count signals may be used for fallback,
re-ranking, observability, and ablation reports only.

Retrieval metadata should explain:

- selected ids or topic names
- selection reason
- fallback path, if used
- prompt character cost
- false-positive or missed-memory labels when evaluator can infer them

## Task State Contract

`TaskState.status` is owned by the runtime loop, not by hooks. Hooks may observe
the lifecycle and write trace/report side effects, but they must not decide
whether a run is completed, stopped, failed, or blocked.

Use `status` for coarse lifecycle outcome and `stop_reason` for the precise
transition reason. Evaluators, reports, resume logic, and UI summaries should
read both fields:

- `status`: coarse category such as `running`, `completed`, `stopped`, `failed`
- `stop_reason`: stable reason such as `final_answer_returned`,
  `step_limit_reached`, `tool_error`, `approval_denied`, `model_error`

Do not add `checkpoint_id` to `TaskState` in the MVP. If a run needs to refer to
a future recovery snapshot, introduce a dedicated snapshot schema first.

## Prompt Order

The prompt must preserve this order:

```text
identity
workspace
tools
task_state
working_memory
relevant_memory
memory_index
history
current_request
```

Each section is rendered independently by `ContextManager` and has prompt
metadata for source, character count, budget, truncation state, and truncation
reason. The current user request is never trimmed.

## Verification Gates

Before considering a refactor phase complete, run the relevant local checks:

- unit tests for changed modules
- prompt metadata tests when context assembly changes
- memory selection/retrieval tests when prompt injection or persistence changes
- runtime smoke test with a fake model
- CLI smoke test for import and help text
