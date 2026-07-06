#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from cascade_correlation_xor import run_experiment  # noqa: E402


def test_cascade_correlation_xor_solves_truth_table() -> None:
    result = run_experiment()
    assert result["grown_model"]["solves_xor"] is True
    assert result["grown_model"]["predictions"] == [0, 1, 1, 0]
    assert result["grown_model"]["mse"] < 0.05
    assert result["candidate"]["residual_correlation"] > 0.5


if __name__ == "__main__":
    test_cascade_correlation_xor_solves_truth_table()
    print("cascade-correlation XOR test passed")
