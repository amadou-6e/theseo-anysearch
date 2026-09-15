"""Provider-neutral voxel world generation and experiment selection CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click
import typer
import yaml
from typer.core import TyperGroup

from theseo_anysearch.world_providers.api import installed_providers, load_provider, provider_errors
from theseo_anysearch.world_providers.service import (
    generate_world,
    load_verified_world,
    local_worlds,
    remote_catalog,
)


class ProviderGroup(TyperGroup):
    """Resolve installed provider names without hard-coding them in the core CLI."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        existing = super().get_command(ctx, cmd_name)
        if existing is not None:
            return existing
        try:
            provider = load_provider(cmd_name)
        except ValueError as exc:
            if cmd_name in provider_errors():
                message = str(exc)

                def broken_provider() -> None:
                    raise click.ClickException(message)

                return click.Command(cmd_name, callback=broken_provider)
            return None
        options: list[click.Parameter] = [
            click.Option(["--seed"], type=click.INT, required=True),
            click.Option(["--output"], type=click.Path(path_type=Path, file_okay=False), required=True),
        ]
        for parameter in provider.info.parameters:
            kind = {
                "integer": click.INT, "number": click.FLOAT,
                "text": click.STRING, "boolean": click.BOOL,
            }[parameter.kind]
            options.append(click.Option(
                [f"--{parameter.name}"], type=kind, required=parameter.required,
                default=parameter.default, help=parameter.help,
            ))

        def invoke_provider(**kwargs: object) -> None:
            seed = kwargs.pop("seed")
            output = kwargs.pop("output")
            try:
                report = generate_world(
                    cmd_name, seed=int(seed), output=Path(output),
                    parameters={key.replace("_", "-"): value for key, value in kwargs.items() if value is not None},
                )
            except (OSError, ValueError, PermissionError) as exc:
                raise click.ClickException(str(exc)) from exc
            click.echo(json.dumps({
                "output": str(Path(output).resolve()),
                "world_identity_sha256": report["world_identity_sha256"],
                "tasks_verified": len(list(Path(output).glob("task-*.json"))),
                "rejected_task_strata": report["rejected_task_strata"],
                "topology_family": report["topology_family"],
                "limitations": report["limitations"],
                "previews": sorted(report["previews"]),
            }, sort_keys=True))

        return click.Command(cmd_name, params=options, callback=invoke_provider, help=provider.info.description)


app = typer.Typer(cls=ProviderGroup, help="List, generate, and select verified voxel worlds.")


@app.command("list")
def list_worlds(remote: bool = typer.Option(False, "--remote", help="Show downloadable providers.")) -> None:
    try:
        if remote:
            for row in remote_catalog():
                typer.echo(f"{row['name']}  {row['distribution']}=={row['version']}")
            return
        for name, provider in sorted(installed_providers().items()):
            info = provider.info
            parameters = ", ".join(item.name for item in info.parameters) or "none"
            extent = "variable" if info.native_extent is None else "x".join(map(str, info.native_extent))
            typer.echo(
                f"provider {name} v{info.version}  native={extent} @ "
                f"{info.native_meters_per_voxel:g} m/voxel  parameters={parameters}"
            )
        for name, error in sorted(provider_errors().items()):
            typer.echo(f"provider {name}  unavailable: {error}")
        for row in local_worlds():
            root = Path(row["path"])
            try:
                load_verified_world(root)
                state = "verified"
            except (OSError, ValueError):
                state = "missing-or-changed"
            typer.echo(f"world {row.get('root_geometry_id', '?')}  {state}  {root}")
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("add")
def add_world(
    world: Path = typer.Argument(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Select a verified world in an existing training or tuning YAML."""

    from theseo_anysearch.world_providers.selection import add_to_experiment

    try:
        result = add_to_experiment(world, config)
    except (OSError, ValueError, PermissionError, yaml.YAMLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    diff = result.pop("diff", "")
    if diff:
        typer.echo(diff, nl=False)
    typer.echo(json.dumps(result, sort_keys=True))
