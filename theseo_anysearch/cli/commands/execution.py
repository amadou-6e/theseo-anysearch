"""Checkpoint execution-recipe commands."""
import json
from pathlib import Path
from typing import Literal
import typer
import yaml
from theseo_anysearch.experiments.execution_recipe import (
    ExecutionRecipe,
    GeometryDecision,
    clone as clone_recipe,
    make_portable,
    validate,
)


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
          task: Literal["preserve", "clear", "replace"] | None = typer.Option(None, "--task"),
          task_config: Path | None = typer.Option(None, "--task-config", exists=True, dir_okay=False),
          routes: Literal["preserve", "clear", "replace"] | None = typer.Option(None, "--routes"),
          routes_config: Path | None = typer.Option(None, "--routes-config", exists=True, dir_okay=False),
          disable_capability: list[str] | None = typer.Option(None, "--disable-capability"),
          replace_capability: list[str] | None = typer.Option(None, "--replace-capability"),
          output_dir: Path | None = typer.Option(None, "--output-dir"),
          episodes: int | None = typer.Option(None, "--episodes", min=1),
          seed: int | None = typer.Option(None, "--seed"),
          iterations: int | None = typer.Option(None, "--iterations", min=1),
          dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Validate and execute an evaluation, continuation, or fine-tuning recipe."""
    loaded = ExecutionRecipe.load(recipe)
    if task is not None:
        value = yaml.safe_load(task_config.read_text(encoding="utf-8")) if task_config else None
        loaded.overrides.task = GeometryDecision(mode=task, value=value)
    if routes is not None:
        value = yaml.safe_load(routes_config.read_text(encoding="utf-8")) if routes_config else None
        loaded.overrides.routes = GeometryDecision(mode=routes, value=value)
    if disable_capability is not None:
        loaded.overrides.disabled_capabilities = disable_capability
    if replace_capability is not None:
        replacements = {}
        for item in replace_capability:
            source, separator, target = item.partition("=")
            if not separator:
                raise typer.BadParameter("capability replacement must be source=target")
            replacements[source] = target
        loaded.overrides.replacements = replacements
    result = validate(loaded, world, recipe.parent)
    print(json.dumps(result, indent=2))
    if not dry_run:
        if output_dir is None:
            raise typer.BadParameter("--output-dir is required for execution")
        if loaded.scope == "evaluation":
            if iterations is not None:
                raise typer.BadParameter("--iterations applies only to training scopes")
            from theseo_anysearch.experiments.execution_evaluate import evaluate

            destination = evaluate(loaded, base=recipe.parent, output_dir=output_dir,
                                   episodes=episodes or result["effective_config"]["evaluation"]["episodes"],
                                   seed=seed if seed is not None else result["effective_config"]["evaluation"]["seed"],
                                   world=world)
        else:
            if iterations is None:
                raise typer.BadParameter("--iterations is required for continuation and fine-tuning")
            if episodes is not None or seed is not None:
                raise typer.BadParameter("--episodes and --seed apply only to evaluation")
            from theseo_anysearch.experiments.execution_train import train_recipe

            destination = train_recipe(loaded, base=recipe.parent, output_dir=output_dir,
                                       iterations=iterations, world=world)
        typer.echo(str(destination))
