#!/usr/bin/env python3
"""Validate growing-neural-networks-lab metadata and available lab files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    experiments = load_json(ROOT / "data" / "experiments.json")
    algorithms = load_json(ROOT / "data" / "algorithms.json")
    algorithm_ids = {item["id"] for item in algorithms}
    experiment_ids = set()

    for experiment in experiments:
        eid = experiment.get("id")
        require(bool(eid), "experiment missing id", errors)
        require(eid not in experiment_ids, f"duplicate experiment id: {eid}", errors)
        experiment_ids.add(eid)
        require(experiment.get("algorithm_id") in algorithm_ids, f"{eid}: unknown algorithm_id", errors)
        if experiment.get("status") == "available":
            for field in ("entry", "script"):
                rel = experiment.get(field)
                require(bool(rel), f"{eid}: missing {field}", errors)
                if rel:
                    require((ROOT / rel).exists(), f"{eid}: {field} not found: {rel}", errors)
            for test_path in experiment.get("tests", []):
                require((ROOT / test_path).exists(), f"{eid}: test not found: {test_path}", errors)
            readme = ROOT / "labs" / eid / "README.md"
            require(readme.exists(), f"{eid}: missing lab README.md", errors)

    for algorithm in algorithms:
        for lab_id in algorithm.get("lab_ids", []):
            require(lab_id in experiment_ids, f"algorithm {algorithm['id']}: unknown lab_id {lab_id}", errors)

    if errors:
        print("Lab validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Lab validation passed: {len(experiments)} experiment(s), {len(algorithms)} algorithm(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
