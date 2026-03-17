import typer

from theseo_anysearch.cli.commands import experiment as experiment_cmd
from theseo_anysearch.cli.commands import mlflow_ui as mlflow_cmd
from theseo_anysearch.cli.commands import ray_cmd
from theseo_anysearch.cli.commands import replay as replay_cmd
from theseo_anysearch.cli.commands import train as train_cmd
from theseo_anysearch.cli.commands import tune as tune_cmd

app = typer.Typer(
    name="anysearch",
    help="Theseo AnySearch — train and tune Rust-backed RL environments.",
    no_args_is_help=True,
)

app.add_typer(train_cmd.app, name="train")
app.add_typer(tune_cmd.app, name="tune")
app.add_typer(experiment_cmd.app, name="experiment")
app.add_typer(replay_cmd.app, name="replay")
app.add_typer(mlflow_cmd.app, name="mlflow")
app.add_typer(ray_cmd.app, name="ray")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
