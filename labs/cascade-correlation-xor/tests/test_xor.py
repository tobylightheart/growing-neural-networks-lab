#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from cascade_correlation_xor import XOR_DATA, run_experiment  # noqa: E402


def test_cascade_correlation_xor_solves_truth_table() -> None:
    result = run_experiment()
    assert result["grown_model"]["solves_xor"] is True
    assert result["grown_model"]["predictions"] == [0, 1, 1, 0]
    assert result["grown_model"]["mse"] < 0.05
    assert result["candidate"]["residual_correlation"] > 0.5


def test_growth_trace_records_real_improvement() -> None:
    result = run_experiment()

    baseline = result["baseline"]
    candidate = result["candidate"]
    grown = result["grown_model"]
    comparison = result["comparison"]

    trace = result["growth_trace"]
    assert len(baseline["outputs"]) == len(XOR_DATA)
    assert len(candidate["hidden_values"]) == len(XOR_DATA)
    assert len(grown["outputs"]) == len(XOR_DATA)
    assert len(trace) == len(XOR_DATA)

    for index, row in enumerate(trace):
        x, target = XOR_DATA[index]
        assert row["x"] == x
        assert row["target"] == target
        assert row["baseline_output"] == baseline["outputs"][index]
        assert row["residual"] == target - baseline["outputs"][index]
        assert row["hidden_value"] == candidate["hidden_values"][index]
        assert row["grown_output"] == grown["outputs"][index]
        assert row["prediction"] == grown["predictions"][index]
        if target == 1.0:
            expected_margin = row["grown_output"] - 0.5
        else:
            expected_margin = 0.5 - row["grown_output"]
        assert row["margin_from_threshold"] == expected_margin
        assert row["margin_from_threshold"] > 0.49

    # A bias-plus-inputs linear readout should stall near the XOR mean.
    assert all(0.49 < output < 0.51 for output in baseline["outputs"])
    assert baseline["mse"] > 0.24

    # The selected hidden feature must be useful after it is frozen and the
    # output layer is refit, not merely correlated with residuals in isolation.
    assert grown["mse"] < baseline["mse"] / 1000
    assert comparison["hidden_feature_frozen_before_refit"] is True
    assert comparison["mse_reduction"] == baseline["mse"] - grown["mse"]
    assert comparison["error_reduction_factor"] == baseline["mse"] / grown["mse"]
    assert comparison["error_reduction_factor"] > 1_000_000
    assert comparison["min_output_margin"] == min(row["margin_from_threshold"] for row in trace)
    assert comparison["min_output_margin"] > 0.49
    for output, (_, target) in zip(grown["outputs"], XOR_DATA):
        if target == 1.0:
            assert output > 0.99
        else:
            assert output < 0.01


if __name__ == "__main__":
    test_cascade_correlation_xor_solves_truth_table()
    test_growth_trace_records_real_improvement()
    print("cascade-correlation XOR tests passed")
