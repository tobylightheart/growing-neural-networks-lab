# Growing Neural Networks Lab

A companion lab for the [Growing Neural Networks](../growing-neural-networks/) literature garden.

The literature garden explains papers, historical context, taxonomies, and synthesis. This lab keeps the runnable side small, inspectable, and testable: pure-Python implementations, browser playgrounds, and tiny experiments that demonstrate mechanisms rather than claiming full paper reproduction.

## Remit

This project should do:

- Implement constructive, growing, pruning, and plastic neural-network mechanisms from scratch.
- Prefer pure Python and static browser demos over heavy ML frameworks.
- Keep every lab small enough to read in one sitting.
- Include validation or tests for every available lab.
- Link back to the literature garden for paper context.

This project should not do:

- Become a second literature-review site.
- Bulk-ingest papers or PDFs.
- Make strong reproduction claims unless a lab actually reproduces a published experiment.
- Replace the synthesis and review workflow in `growing-neural-networks`.

## Initial labs

- `labs/cascade-correlation-xor/` — a minimal cascade-correlation-style XOR experiment with a pure-Python script and static browser explainer.

## Validate

```bash
python3 scripts/validate_lab.py
```

The validator executes every test path declared by each available experiment,
as well as checking metadata, runnable script output, and static trace snapshots.

## Preview

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```
