# Cross-project remit

`growing-neural-networks` and `growing-neural-networks-lab` are deliberately paired but separate.

## growing-neural-networks

Primary role: literature garden.

It owns:

- paper records, reviews, bibliographic provenance, and review bundles
- historical synthesis and cautious conceptual taxonomies
- static modules/exercises when they are part of the review narrative
- PDF wanted lists and private-library planning

It should not become a repository of many standalone implementations.

## growing-neural-networks-lab

Primary role: runnable lab bench.

It owns:

- pure-Python reference implementations
- tiny toy datasets and algorithm traces
- static browser experiments that demonstrate mechanisms
- tests and validators showing that the examples run

It should not become a second paper catalogue. Literature claims should be short and link back to the main garden.

## Boundary rule

When adding something new, ask:

- Is the main value a paper, review, taxonomy, or synthesis? Put it in `growing-neural-networks`.
- Is the main value runnable code, a mechanism trace, or a toy experiment? Put it in `growing-neural-networks-lab`.
- Is it both? Put the source review/synthesis in the garden and the runnable artifact in the lab, linked both ways.
