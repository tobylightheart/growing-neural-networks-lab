# Perceptron on AND

A deterministic, dependency-free trace of the classic perceptron update rule on
the linearly separable AND truth table.

## What it demonstrates

- a bias plus two input weights;
- strict-positive thresholding;
- sample-by-sample error-driven updates;
- convergence after repeated passes over fixed data; and
- a fixed-capacity baseline against which constructive growth can be contrasted.

This is a toy mechanism, not a reproduction of a published experiment. It does
not add hidden units or otherwise grow network capacity. For broader algorithm
context, use the main Growing Neural Networks literature garden.

## Run

```bash
python3 perceptron_and.py
python3 tests/test_perceptron.py
```

The script emits the same JSON object committed as `trace.json`. The browser
explainer fetches that file so the Python run, tests, and static page share one
canonical trace.
