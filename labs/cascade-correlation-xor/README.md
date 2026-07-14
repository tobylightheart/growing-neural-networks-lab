# Cascade-Correlation on XOR

This lab demonstrates a small cascade-correlation-style idea: when a linear output model cannot solve XOR, add a hidden feature whose activation is correlated with the residual error, freeze that feature, and retrain the output layer.

The implementation is deliberately compact and deterministic. It is not a full reproduction of the original Cascade-Correlation algorithm; it is a toy mechanism trace suitable for reading and modification.

## Run

```bash
python3 cascade_correlation_xor.py
python3 tests/test_xor.py
```

## What to look for

- The baseline linear model stalls on XOR.
- A small candidate search finds a useful hidden feature.
- After adding the feature, the classifier reaches the XOR truth table.
- The `comparison` block reports the baseline-to-grown MSE reduction and confirms that the hidden feature is frozen before the output refit.
- The script's `growth_trace` rows align each XOR example with its baseline output, residual, hidden activation, grown output, and final prediction so the mechanism can be inspected without re-deriving the numbers.

## Companion context

For paper context and historical synthesis, use the main `growing-neural-networks` literature garden. This lab intentionally keeps the literature section short and focuses on runnable mechanics.
