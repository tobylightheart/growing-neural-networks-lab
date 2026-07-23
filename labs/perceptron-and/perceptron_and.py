#!/usr/bin/env python3
"""Deterministic perceptron training trace for the AND truth table.

This is a tiny educational baseline, not a paper reproduction and not a growing
network. It uses no random state or external dependencies.
"""
from __future__ import annotations

import json

AND_DATA = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 0),
    ([1.0, 0.0], 0),
    ([1.0, 1.0], 1),
]
LEARNING_RATE = 1.0
MAX_EPOCHS = 10


def activation(weights: list[float], inputs: list[float]) -> float:
    return weights[0] + sum(weight * value for weight, value in zip(weights[1:], inputs))


def predict(weights: list[float], inputs: list[float]) -> int:
    """Use the pilot's pinned strict-positive decision boundary."""
    return 1 if activation(weights, inputs) > 0.0 else 0


def predictions(weights: list[float]) -> list[int]:
    return [predict(weights, inputs) for inputs, _ in AND_DATA]


def run_experiment() -> dict:
    weights = [0.0, 0.0, 0.0]
    sample_trace: list[dict] = []
    epoch_trace: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        weights_at_start = weights[:]
        errors = 0

        for inputs, target in AND_DATA:
            weights_before = weights[:]
            score = activation(weights, inputs)
            prediction = 1 if score > 0.0 else 0
            error = target - prediction
            update = [
                LEARNING_RATE * error,
                LEARNING_RATE * error * inputs[0],
                LEARNING_RATE * error * inputs[1],
            ]
            weights = [weight + delta for weight, delta in zip(weights, update)]
            if error != 0:
                errors += 1

            sample_trace.append(
                {
                    "epoch": epoch,
                    "x": inputs,
                    "target": target,
                    "activation": score,
                    "prediction": prediction,
                    "error": error,
                    "weights_before": weights_before,
                    "update": update,
                    "weights_after": weights[:],
                }
            )

        predictions_after = predictions(weights)
        epoch_trace.append(
            {
                "epoch": epoch,
                "weights_before": weights_at_start,
                "weights_after": weights[:],
                "errors": errors,
                "predictions_after": predictions_after,
                "converged": errors == 0,
            }
        )
        if errors == 0:
            break

    targets = [target for _, target in AND_DATA]
    return {
        "algorithm": "perceptron",
        "claim_scope": ["toy-mechanism", "not-a-full-paper-reproduction"],
        "policy": {
            "dataset_order": [inputs for inputs, _ in AND_DATA],
            "initial_weights": [0.0, 0.0, 0.0],
            "weight_order": ["bias", "x1", "x2"],
            "learning_rate": LEARNING_RATE,
            "positive_when": "activation > 0",
            "max_epochs": MAX_EPOCHS,
        },
        "dataset": [{"x": inputs, "target": target} for inputs, target in AND_DATA],
        "epoch_trace": epoch_trace,
        "sample_trace": sample_trace,
        "final_model": {
            "weights": weights,
            "predictions": predictions(weights),
            "targets": targets,
            "converged": predictions(weights) == targets and epoch_trace[-1]["converged"],
            "epochs_run": len(epoch_trace),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
