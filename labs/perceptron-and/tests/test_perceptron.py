#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from perceptron_and import AND_DATA, LEARNING_RATE, activation, predict, run_experiment  # noqa: E402


def test_perceptron_converges_deterministically() -> None:
    first = run_experiment()
    second = run_experiment()
    final = first["final_model"]

    assert first == second
    assert final["converged"] is True
    assert final["epochs_run"] == 6
    assert final["weights"] == [-2.0, 2.0, 1.0]
    assert final["predictions"] == [0, 0, 0, 1]
    assert final["targets"] == [target for _, target in AND_DATA]


def test_sample_trace_replays_every_weight_update() -> None:
    result = run_experiment()
    weights = result["policy"]["initial_weights"][:]
    rows_per_epoch = len(AND_DATA)

    assert len(result["sample_trace"]) == result["final_model"]["epochs_run"] * rows_per_epoch

    for row in result["sample_trace"]:
        assert row["weights_before"] == weights
        expected_activation = activation(weights, row["x"])
        expected_prediction = predict(weights, row["x"])
        expected_error = row["target"] - expected_prediction
        expected_update = [
            LEARNING_RATE * expected_error,
            LEARNING_RATE * expected_error * row["x"][0],
            LEARNING_RATE * expected_error * row["x"][1],
        ]
        assert row["activation"] == expected_activation
        assert row["prediction"] == expected_prediction
        assert row["error"] == expected_error
        assert row["update"] == expected_update
        weights = [weight + delta for weight, delta in zip(weights, expected_update)]
        assert row["weights_after"] == weights

    assert weights == result["final_model"]["weights"]


def test_epoch_trace_matches_sample_trace() -> None:
    result = run_experiment()
    sample_trace = result["sample_trace"]

    for epoch in result["epoch_trace"]:
        rows = [row for row in sample_trace if row["epoch"] == epoch["epoch"]]
        assert len(rows) == len(AND_DATA)
        assert epoch["weights_before"] == rows[0]["weights_before"]
        assert epoch["weights_after"] == rows[-1]["weights_after"]
        assert epoch["errors"] == sum(row["error"] != 0 for row in rows)
        assert epoch["predictions_after"] == [predict(epoch["weights_after"], x) for x, _ in AND_DATA]
        assert epoch["converged"] is (epoch["errors"] == 0)


if __name__ == "__main__":
    test_perceptron_converges_deterministically()
    test_sample_trace_replays_every_weight_update()
    test_epoch_trace_matches_sample_trace()
    print("perceptron AND tests passed")
