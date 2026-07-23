# Debrief: GNL-1 Add a deterministic perceptron trace lab

**Completed:** 2026-07-23
**Commit:** 07dafd8310fe953faae6a5fbb9620b9def4e151c

## What shipped

Added a dependency-free perceptron lab for the AND truth table with a fully
replayable sample/epoch training trace, deterministic tests, synchronized static
`trace.json`, browser explainer, metadata registry entries, root navigation, and
roadmap/README updates. Generalized the repository validator so any successful
JSON-object script can be compared with its declared trace artifact while still
checking that `growth_trace` is a list when present.

## Descoped / deferred

ADALINE and learning-rate/margin comparisons remain roadmap items as explicitly
scoped out of this first perceptron slice.

## Design decisions

None — followed the refined brief's pinned dataset order, zero initialization,
learning rate, strict-positive threshold, and ten-epoch cap.

## Observations

- The frozen packet's task-file and exact JSONL-line hashes matched cleanly and
  were straightforward to verify before changing task state.
- The existing validator's `growth_trace` gate would have silently skipped
  snapshot comparison for fixed-capacity algorithms; returning every valid JSON
  object made the artifact contract algorithm-neutral.
- The deterministic policy converges in six epochs with final weights
  `[-2.0, 2.0, 1.0]`, making trace replay objective and easy to test.
- The static browser route, script output, tests, and committed trace now share
  one canonical JSON object.

## Follow-ups

No new task was filed. ADALINE and richer learning-rate/margin visualization
already remain visible in `docs/roadmap.md`; they should be reconsidered during
normal portfolio planning rather than automatically queued by this pilot.
