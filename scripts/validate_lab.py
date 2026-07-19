#!/usr/bin/env python3
"""Validate growing-neural-networks-lab metadata and available lab files."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_script_output(rel: str, eid: str, errors: list[str]) -> dict | None:
    """Run an available lab script and require a JSON experiment trace.

    Lab scripts are intentionally tiny and dependency-light, so the metadata
    validator can afford to execute them. This catches stale script paths and
    non-JSON regressions before the static site points readers at a broken lab.
    """
    script_path = ROOT / rel
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        errors.append(f"{eid}: script timed out: {rel}")
        return None

    if completed.returncode != 0:
        errors.append(f"{eid}: script failed ({completed.returncode}): {rel}")
        if completed.stderr.strip():
            errors.append(f"{eid}: script stderr: {completed.stderr.strip()}")
        return None

    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"{eid}: script did not emit JSON: {rel} ({exc})")
        return None

    require(isinstance(output, dict), f"{eid}: script JSON output should be an object", errors)
    require(bool(output), f"{eid}: script JSON output should not be empty", errors)
    if isinstance(output, dict) and "growth_trace" in output:
        require(isinstance(output["growth_trace"], list), f"{eid}: growth_trace should be a list", errors)
        return output
    return None


def validate_trace_snapshot(
    trace_path: Path,
    script_output: dict | None,
    eid: str,
    errors: list[str],
) -> None:
    """Keep a static trace snapshot in lockstep with the runnable script."""
    snapshot = load_json(trace_path)
    trace_label = trace_path.name
    require(isinstance(snapshot, dict), f"{eid}: {trace_label} should be a JSON object", errors)
    if script_output is None or not isinstance(snapshot, dict):
        return
    require(
        snapshot == script_output,
        f"{eid}: {trace_label} is stale; regenerate it from the lab script",
        errors,
    )


def validate_artifacts(
    lab_dir: Path,
    artifacts: object,
    script_output: dict | None,
    eid: str,
    errors: list[str],
) -> None:
    """Validate lab-local artifact metadata for static-site routes.

    Artifact paths are intentionally lab-local so an available experiment can be
    served from a static file host without depending on generated files outside
    the lab directory.
    """
    require(isinstance(artifacts, dict), f"{eid}: artifacts should be a JSON object", errors)
    if not isinstance(artifacts, dict):
        return

    for artifact_name, artifact_rel in artifacts.items():
        require(isinstance(artifact_rel, str), f"{eid}: artifact {artifact_name} path should be a string", errors)
        if not isinstance(artifact_rel, str):
            continue
        artifact_path = Path(artifact_rel)
        require(
            not artifact_path.is_absolute() and ".." not in artifact_path.parts,
            f"{eid}: artifact {artifact_name} should stay lab-local: {artifact_rel}",
            errors,
        )
        full_path = lab_dir / artifact_path
        require(full_path.exists(), f"{eid}: artifact {artifact_name} not found: {artifact_rel}", errors)
        if artifact_name == "trace" and full_path.exists():
            validate_trace_snapshot(full_path, script_output, eid, errors)


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
            script_output = None
            script_rel = experiment.get("script")
            if script_rel and (ROOT / script_rel).exists():
                script_output = validate_script_output(script_rel, str(eid), errors)
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
                validate_artifacts(
                    lab_dir,
                    lab_metadata.get("artifacts", {}),
                    script_output,
                    str(eid),
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
