"""Checkpoint execution-recipe commands."""
import json
from pathlib import Path
from typing import Literal
import typer
from theseo_anysearch.experiments.execution_recipe import ExecutionRecipe, clone as clone_recipe, make_portable, validate


def clone(checkpoint: Path = typer.Option(..., exists=True, file_okay=False),
          scope: Literal["evaluation", "continuation", "fine_tuning"] = typer.Option(...),
          output: Path = typer.Option(...),
          bundle_dir: Path | None = typer.Option(None, "--bundle-dir")) -> None:
    """Clone checkpoint provenance and policy-facing settings into a recipe."""
    recipe = clone_recipe(checkpoint, scope)
    if bundle_dir is not None:
        recipe = make_portable(recipe, bundle_dir)
        output = bundle_dir / output.name
    recipe.save(output)
    typer.echo(str(output))


def apply(recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
          world: Path | None = typer.Option(None, exists=True, dir_okay=False),
          dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Validate a recipe. Execution remains blocked until the executor is implemented."""
    result = validate(ExecutionRecipe.load(recipe), world, recipe.parent)
    print(json.dumps(result, indent=2))
    if not dry_run:
        typer.echo("Refusing execution: use --dry-run; policy execution is not implemented yet.", err=True)
        raise typer.Exit(2)
