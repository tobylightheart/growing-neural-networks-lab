# Tasks

> **Agents:** read this file at the start of every session, then consult
> `.tasks/LOG.jsonl` for the authoritative queue. Use the external
> `task-cycle` skill to file, start, complete, and debrief work.

## Current focus

Use the first portfolio-cycle pilot to add one small, pure-Python mechanism lab
without expanding this repository into a second literature catalogue.

The repository task-ID prefix is `GNL`.

## Queue

See `.tasks/LOG.jsonl`. An incomplete task has a corresponding Markdown file in
`.tasks/`; completed task files are deleted after their debrief is committed.

## Structure

```text
.tasks/
├── LOG.jsonl
├── debriefs/
└── GNL-N-....md
```

## Quick reference

| What | Where |
|---|---|
| Full task queue | `.tasks/LOG.jsonl` |
| Active task files | `.tasks/*.md` |
| Completed debriefs | `.tasks/debriefs/` |
| Templates and procedure | external `task-cycle` skill |
