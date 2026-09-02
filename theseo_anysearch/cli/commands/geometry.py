"""Ray-free geometry validation, inspection, and deterministic sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Validate, inspect, and sample experiment geometry without Ray.")


def _resolve(config_path: Path, seed: int, sample_index: int = 0) -> tuple[Any, dict[str, Any], tuple[tuple[int, int, int], ...]]:
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig

    experiment = load_experiment(config_path)
    if not isinstance(experiment, ExperimentConfig):
        raise ValueError("geometry commands require one experiment, not a sweep")
    runtime = experiment.env.to_runtime_dict()
    if runtime.get("compiled_world_path"):
        compiled = Path(str(runtime["compiled_world_path"]))
        if not compiled.is_absolute():
            compiled = config_path.parent.joinpath(compiled).resolve()
        runtime["compiled_world_path"] = str(compiled)
    if runtime.get("waypoints_file"):
        waypoint_path = Path(str(runtime["waypoints_file"]))
        if not waypoint_path.is_absolute():
            waypoint_path = config_path.parent.joinpath(waypoint_path).resolve()
        runtime["waypoints"] = json.loads(waypoint_path.read_text(encoding="utf-8"))
    extent = tuple(experiment.env.geometry.extent or (experiment.env.geometry.grid_size,) * 3)
    if experiment.env.geometry.compiled_world_path is not None:
        return experiment, runtime, ()
    provider = experiment.env.geometry.provider
    if provider is not None:
        from theseo_anysearch.experiments.custom_geometry import (
            GeometryContext,
            GeometryTaskRequirements,
            _EmptyWorld,
            discover_geometry_source,
            load_geometry_provider,
        )

        source = discover_geometry_source(config_path, provider.name)
        loaded = load_geometry_provider(source, provider.name)
        if loaded is None:
            raise ValueError("native providers require a compiled extension and live native host")
        context = GeometryContext(
            seed=seed + sample_index, attempt=1, extent=extent,
            task=GeometryTaskRequirements(
                max_steps=experiment.env.max_steps, action_mode=experiment.env.action.mode,
            ), parameters=provider.parameters, world=_EmptyWorld(extent),
        )
        proposal = loaded.generate(context)
        runtime["geometry_sources"] = [item.model_dump(mode="json") for item in proposal.sources]
        runtime["geometry_proposal"] = proposal.model_dump(mode="json")
    if (runtime.get("geometry_pool") or {}).get("pool_dir"):
        from theseo_anysearch.environments.geometry_pool import GeometryPool
        import numpy as np

        pool = GeometryPool(runtime["geometry_pool"]["pool_dir"], seed=seed + sample_index)
        cells = GeometryPool.grid_to_cells(pool.sample(rng=np.random.default_rng(seed + sample_index)))
        return experiment, runtime, tuple(cells)
    from theseo_anysearch.environments.geometry_sources import resolve_geometry_sources
    from theseo_anysearch.environments.pettingzoo.multi_voxel_env import _load_stl_geometry

    cells = resolve_geometry_sources(runtime, grid_size=max(extent), load_stl=_load_stl_geometry)
    return experiment, runtime, tuple(cells)


def geometry_report(config_path: Path, seed: int = 42, sample_index: int = 0) -> dict[str, Any]:
    """Build a deterministic, machine-readable report without constructing Ray."""
    experiment, runtime, coordinates = _resolve(config_path, seed, sample_index)
    from theseo_anysearch.environments.task_identity import configured_geometry_identity
    from theseo_anysearch.environments.validation import BoundedWorldRead, validate_geometry, validate_task_feasibility
    from theseo_anysearch.environments.action_spaces import offsets_for_mode
    from theseo_anysearch.worlds.extent import resolve_extent

    extent = resolve_extent(runtime)
    compiled_path = (
        Path(str(runtime["compiled_world_path"]))
        if runtime.get("compiled_world_path")
        else None
    )
    if compiled_path is not None:
        from theseo_anysearch.worlds.artifacts import load_geometry_artifact
        artifact = load_geometry_artifact(compiled_path)
        occupancy = sum(chunk.occupied_voxels for chunk in artifact.compiled_world.manifest.chunks)  # type: ignore[union-attr]
        geometry_valid = True
        geometry_reason = None
        identity = artifact.manifest.identity_sha256
    else:
        result = validate_geometry(coordinates, extent)
        occupancy = len(coordinates)
        geometry_valid = result.valid
        geometry_reason = result.rejection_reason
        identity = configured_geometry_identity(runtime)
    waypoints = runtime.get("waypoints") or {}
    start = tuple(waypoints["start"]) if waypoints.get("start") else None
    goal = tuple(waypoints["goal"]) if waypoints.get("goal") else None
    feasibility: dict[str, Any] | None = None
    if compiled_path is None and start is not None and goal is not None and geometry_valid:
        occupied = set(coordinates)
        planned = validate_task_feasibility(
            BoundedWorldRead(lambda coordinate: coordinate in occupied),
            start=start, goal=goal, extent=extent,
            directions=offsets_for_mode(runtime.get("action_mode", "discrete_26")),
            action_mode=runtime.get("action_mode", "discrete_26"),
            maximum_search_nodes=int((runtime.get("geometry_validation") or {}).get("maximum_search_nodes", 100_000)),
            maximum_steps=int(runtime.get("max_steps", 256)),
        )
        feasibility = planned.model_dump(mode="json")
    feasible = feasibility is None or bool(feasibility["feasible"])
    return {
        "seed": seed + sample_index, "extent": list(extent), "occupancy_count": occupancy,
        "geometry_identity": identity,
        "geometry_validity": {"valid": geometry_valid, "rejection_reason": geometry_reason},
        "task_feasibility": feasibility,
        "training_suitability": {"suitable": geometry_valid and feasible, "reason": None if geometry_valid and feasible else geometry_reason or "infeasible"},
        "evaluation_suitability": {"suitable": geometry_valid and feasible and identity is not None, "stable_identity": identity},
        "proposal": runtime.get("geometry_proposal"),
        "bounded_large_world_read": compiled_path is not None,
    }


def _emit(report: dict[str, Any], json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    typer.echo(f"geometry: {'valid' if report['geometry_validity']['valid'] else 'INVALID'}")
    typer.echo(f"identity: {report['geometry_identity']}")
    typer.echo(f"extent: {report['extent']}  occupied: {report['occupancy_count']}")
    feasibility = report["task_feasibility"]
    typer.echo("task: not configured" if feasibility is None else f"task: {'feasible' if feasibility['feasible'] else 'INFEASIBLE'}")


@app.command("inspect")
def inspect_geometry(config: Path, seed: int = 42, json_output: bool = typer.Option(False, "--json")) -> None:
    """Inspect resolved geometry and task metadata."""
    try:
        _emit(geometry_report(config, seed), json_output)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command("validate")
def validate_geometry_command(config: Path, seed: int = 42, json_output: bool = typer.Option(False, "--json")) -> None:
    """Fail when geometry or its configured navigation task is invalid."""
    report = geometry_report(config, seed)
    _emit(report, json_output)
    if not report["training_suitability"]["suitable"]:
        raise typer.Exit(1)


@app.command("sample")
def sample_geometry(config: Path, count: int = typer.Option(10, min=1), seed: int = 42, output: Path | None = None) -> None:
    """Sample a deterministic geometry distribution and optionally save JSON."""
    reports = [geometry_report(config, seed, index) for index in range(count)]
    payload = {"seed": seed, "count": count, "accepted": sum(item["training_suitability"]["suitable"] for item in reports), "samples": reports}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    typer.echo(encoded)
