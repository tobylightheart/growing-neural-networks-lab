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
            require(
                bool(experiment.get("summary")),
                f"{eid}: available experiment missing summary",
                errors,
            )
            require(
                bool(experiment.get("main_garden_links")),
                f"{eid}: available experiment missing main_garden_links",
                errors,
            )
            for link in experiment.get("main_garden_links", []):
                require(
                    isinstance(link, str) and link.startswith("../growing-neural-networks/"),
                    f"{eid}: main garden link should stay a relative cross-project link: {link}",
                    errors,
                )
            claims = set(experiment.get("claims", []))
            require(
                "toy-mechanism" in claims,
                f"{eid}: available experiment should declare toy-mechanism claim scope",
                errors,
            )
            require(
                "not-a-full-paper-reproduction" in claims,
                f"{eid}: available experiment should declare not-a-full-paper-reproduction claim scope",
                errors,
            )

            for field in ("entry", "script"):
                rel = experiment.get(field)
                require(bool(rel), f"{eid}: missing {field}", errors)
                if rel:
                    require((ROOT / rel).exists(), f"{eid}: {field} not found: {rel}", errors)
            tests = experiment.get("tests", [])
            require(bool(tests), f"{eid}: available experiment missing tests", errors)
            for test_path in tests:
                require((ROOT / test_path).exists(), f"{eid}: test not found: {test_path}", errors)

            lab_dir = ROOT / "labs" / str(eid)
            readme = lab_dir / "README.md"
            require(readme.exists(), f"{eid}: missing lab README.md", errors)

            lab_metadata_path = lab_dir / "experiment.json"
            require(lab_metadata_path.exists(), f"{eid}: missing lab experiment.json", errors)
            if lab_metadata_path.exists():
                lab_metadata = load_json(lab_metadata_path)
                require(
                    lab_metadata.get("id") == eid,
                    f"{eid}: lab experiment.json id does not match data/experiments.json",
                    errors,
                )
                require(
                    lab_metadata.get("algorithm_id") == experiment.get("algorithm_id"),
                    f"{eid}: lab experiment.json algorithm_id does not match data/experiments.json",
                    errors,
                )
                require(
                    lab_metadata.get("status") == experiment.get("status"),
                    f"{eid}: lab experiment.json status does not match data/experiments.json",
                    errors,
                )
                require(
                    bool(lab_metadata.get("limitations")),
                    f"{eid}: lab experiment.json should document limitations",
                    errors,
                )

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
