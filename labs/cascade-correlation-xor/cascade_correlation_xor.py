#!/usr/bin/env python3
"""Tiny cascade-correlation-style XOR lab.

This is a toy mechanism demonstration, not a full reproduction of the original
Cascade-Correlation training algorithm. It uses a deterministic candidate search
for one hidden feature and least-squares output fitting.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass

XOR_DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax=b by Gaussian elimination with partial pivoting."""
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [rv - factor * cv for rv, cv in zip(aug[row], aug[col])]
    return [aug[row][-1] for row in range(n)]


def fit_least_squares(features: list[list[float]], targets: list[float], ridge: float = 1e-6) -> list[float]:
    cols = len(features[0])
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]
    for row, target in zip(features, targets):
        for i in range(cols):
            xty[i] += row[i] * target
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]
    for i in range(cols):
        xtx[i][i] += ridge
    return solve_linear_system(xtx, xty)


def dot(weights: list[float], features: list[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def mse(values: list[float], targets: list[float]) -> float:
    return sum((value - target) ** 2 for value, target in zip(values, targets)) / len(targets)


def correlation(values: list[float], residuals: list[float]) -> float:
    mean_v = sum(values) / len(values)
    mean_r = sum(residuals) / len(residuals)
    numerator = sum((v - mean_v) * (r - mean_r) for v, r in zip(values, residuals))
    denom_v = math.sqrt(sum((v - mean_v) ** 2 for v in values))
    denom_r = math.sqrt(sum((r - mean_r) ** 2 for r in residuals))
    if denom_v * denom_r == 0:
        return 0.0
    return numerator / (denom_v * denom_r)


@dataclass(frozen=True)
class CandidateUnit:
    bias: float
    w1: float
    w2: float
    residual_correlation: float
    score: float

    def activate(self, x1: float, x2: float) -> float:
        return sigmoid(self.bias + self.w1 * x1 + self.w2 * x2)


def input_features(data=XOR_DATA) -> list[list[float]]:
    return [[1.0, x[0], x[1]] for x, _ in data]


def fit_output(features: list[list[float]], data=XOR_DATA) -> tuple[list[float], list[float], float]:
    targets = [target for _, target in data]
    weights = fit_least_squares(features, targets)
    outputs = [dot(weights, row) for row in features]
    return weights, outputs, mse(outputs, targets)


def find_best_candidate(residuals: list[float], data=XOR_DATA) -> CandidateUnit:
    best: CandidateUnit | None = None
    search_values = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    for bias, w1, w2 in itertools.product(search_values, repeat=3):
        activations = [sigmoid(bias + w1 * x[0] + w2 * x[1]) for x, _ in data]
        corr = abs(correlation(activations, residuals))
        # Prefer units that improve the readout, not just units with high raw
        # correlation. This keeps the toy trace honest: the candidate is only
        # useful if freezing it and refitting the output layer helps.
        grown_features = [[1.0, x[0], x[1], activation] for (x, _), activation in zip(data, activations)]
        _, _, candidate_mse = fit_output(grown_features, data)
        score = corr + (1.0 / (1.0 + candidate_mse))
        if best is None or score > best.score:
            best = CandidateUnit(float(bias), float(w1), float(w2), corr, score)
    assert best is not None
    return best


def run_experiment() -> dict:
    targets = [target for _, target in XOR_DATA]
    base_features = input_features()
    base_weights, base_outputs, base_mse = fit_output(base_features)
    residuals = [target - output for target, output in zip(targets, base_outputs)]

    candidate = find_best_candidate(residuals)
    grown_features = []
    hidden_values = []
    for x, _ in XOR_DATA:
        hidden = candidate.activate(x[0], x[1])
        hidden_values.append(hidden)
        grown_features.append([1.0, x[0], x[1], hidden])

    grown_weights, grown_outputs, grown_mse = fit_output(grown_features)
    predictions = [1 if output >= 0.5 else 0 for output in grown_outputs]
    growth_trace = []
    for (x, target), base_output, residual, hidden, grown_output, prediction in zip(
        XOR_DATA, base_outputs, residuals, hidden_values, grown_outputs, predictions
    ):
        growth_trace.append(
            {
                "x": x,
                "target": target,
                "baseline_output": base_output,
                "residual": residual,
                "hidden_value": hidden,
                "grown_output": grown_output,
                "prediction": prediction,
            }
        )

    return {
        "baseline": {"weights": base_weights, "outputs": base_outputs, "mse": base_mse},
        "candidate": {
            "bias": candidate.bias,
            "weights": [candidate.w1, candidate.w2],
            "residual_correlation": candidate.residual_correlation,
            "candidate_score": candidate.score,
            "hidden_values": hidden_values,
        },
        "grown_model": {
            "weights": grown_weights,
            "outputs": grown_outputs,
            "predictions": predictions,
            "mse": grown_mse,
            "solves_xor": predictions == [0, 1, 1, 0],
        },
        "growth_trace": growth_trace,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
