# GNL-1 Add a deterministic perceptron trace lab

**Priority:** high
**Blocked by:** nothing
**Touches:** `labs/perceptron-and/`, `data/experiments.json`, `data/algorithms.json`, `scripts/validate_lab.py`, `README.md`, `docs/roadmap.md`, `index.html`

## Context

The lab roadmap names “Perceptron and ADALINE basics” as its first near-term
candidate. The repository currently has one available Cascade-Correlation lab,
whose validator assumes a script exposes `growth_trace` before it will compare a
static trace snapshot. A perceptron is a smaller, non-growing baseline and is a
good first portfolio-cycle pilot: it has deterministic behavior, objective
tests, and no research or product decision that should be made unattended.

This task is the perceptron slice only. ADALINE should remain a later task rather
than being bundled into this first pilot.

## Goal

Add a compact, pure-Python perceptron lab that learns a linearly separable AND
truth table and exposes a deterministic training trace in both Python and a
static browser explainer.

## Acceptance criteria

- [ ] `labs/perceptron-and/perceptron_and.py` uses only the Python standard library, trains deterministically, emits one JSON object, and records enough epoch/update detail to explain convergence.
- [ ] Tests verify deterministic convergence, final truth-table predictions, and internally consistent trace updates.
- [ ] The lab includes `experiment.json`, `trace.json`, `README.md`, `index.html`, `demo.js`, and its tests, with cautious “toy mechanism / not a paper reproduction” framing.
- [ ] `data/experiments.json` and `data/algorithms.json` register the available lab and retain valid relative links to the main garden.
- [ ] `scripts/validate_lab.py` compares a declared trace artifact with any successful JSON-object script output; `growth_trace` remains list-validated when present but is no longer required for snapshot comparison.
- [ ] Root README, roadmap, and index expose the new lab without adding literature-review content.
- [ ] `python3 scripts/validate_lab.py` passes and directly executes all declared tests.
- [ ] Changed JSON passes `python3 -m json.tool`, changed JavaScript passes `node --check` when Node is available, `git diff --check` passes, and static routes are smoke-tested over a local HTTP server.

## Relevant files

- `docs/roadmap.md`
- `data/experiments.json`
- `data/algorithms.json`
- `scripts/validate_lab.py`
- `labs/cascade-correlation-xor/` as the existing artifact pattern
- `index.html`
- `README.md`

## Decisions already made

- Implement only the perceptron slice; do not add ADALINE in this task.
- Use the AND truth table so the mechanism is linearly separable and the expected result is objective.
- Keep training and output deterministic; no random initialization or external dependency.
- Treat this as a baseline learning-mechanism lab, not a growing-network algorithm and not a published-experiment reproduction.
- Keep one canonical committed `trace.json`; the browser must fetch it rather than duplicate numeric constants.

## Out of scope

- ADALINE, multilayer networks, XOR learning, neural growth, pruning, or structural plasticity.
- New paper reviews, bibliographic records, or PDF ingestion.
- Framework dependencies such as NumPy, PyTorch, or TensorFlow.
- Changes to the sibling `growing-neural-networks` repository.
