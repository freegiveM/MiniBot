# MiniBot Agent Constraints

This document is the project contract for the MiniBot refactor. Treat the local
`pico/` directory as a reference implementation only. It is ignored by Git and
must not be committed as project source.

## Project Goal

MiniBot is a lightweight local coding agent for repository inspection,
modification, debugging, review, checkpoint recovery, memory-assisted context,
and regression evaluation.

The refactor should keep Pico's useful chain:

```text
user request -> prompt build -> model decision -> tool execution ->
history/trace/task_state/report -> checkpoint/session persistence
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
   candidate memory generation, phase updates, and maintenance.
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
- `checkpoints`
- `runs`
- `memory_maintenance`
- `pending_delegates`

Disallowed session fields:

- persisted `relevant_memory`
- source `file_summaries` that are used as code facts
- unbounded raw tool output
- pending candidate bodies embedded directly in session

## Phase Contract

Use a fixed enum:

```text
intake, inspect, plan, implement, verify, debug, finalize, blocked
```

The runtime owns phase updates. The model may suggest a phase, but may not write
free-form phase values.

Expected transitions:

- new user request -> `intake` or `inspect`
- list/read/search/delegate -> `inspect`
- write/patch -> `implement`
- test/build/compile shell command -> `verify`
- tool error or partial success -> `debug`
- rejected action, denied approval, path escape, schema mismatch -> `blocked`
- normal final answer -> `finalize`

## Memory Contract

Working memory should contain:

- initial request summary
- current task summary
- current phase and phase reason
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

Build a lightweight retrieval context from:

- current user request
- task summary
- current phase
- recent files
- recent tools
- file access paths and symbols

Primary scoring signals:

- path match
- symbol match
- keyword overlap
- recent context match

Secondary signals:

- tag match
- topic match
- tier
- hit count
- recency
- freshness

## Prompt Order

The prompt must preserve this order:

```text
system/prefix
workspace/tool/checkpoint context
task state and working memory
hot memory
relevant memory topK
memory index
history/transcript
current user request
```

The current user request is never trimmed.

## Verification Gates

Before considering a refactor phase complete, run the relevant local checks:

- unit tests for changed modules
- prompt metadata tests when context assembly changes
- memory retrieval tests when scoring or persistence changes
- runtime smoke test with a fake model
- CLI smoke test for import and help text
