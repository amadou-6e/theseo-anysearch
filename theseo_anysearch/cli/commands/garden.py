"""anysearch garden — pre-trained voxel encoder commands."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console(highlight=False)
err_console = Console(file=sys.stderr, highlight=False)

app = typer.Typer(
    name="garden",
    help="Pre-trained voxel encoder model garden.",
    no_args_is_help=True,
)

_DEFAULT_GARDEN = Path("runtime/garden")
_PRESETS_DIR = Path(__file__).parent.parent.parent / "garden" / "presets"


def _resolve_config_path(config: str) -> Path | None:
    """Resolve a config arg to a YAML Path.

    Tries in order:
      1. Literal file path that exists.
      2. Built-in presets (r1, r2, r3, r4).
      3. Config registry (name registered via ``garden add <file.yaml>``).
    Returns None if neither resolves.
    """
    p = Path(config)
    if p.exists():
        return p
    preset = _PRESETS_DIR / f"{config}.yaml"
    if preset.exists():
        return preset
    from theseo_anysearch.garden.store import resolve_config
    return resolve_config(config)


def _list_presets() -> list[dict]:
    """Return metadata for all built-in presets, sorted by radius."""
    import yaml
    result = []
    for f in sorted(_PRESETS_DIR.glob("*.yaml"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
        raw = yaml.safe_load(f.read_text()) or {}
        result.append({
            "name": raw.get("name", f.stem),
            "box_radius": raw.get("encoder", {}).get("box_radius"),
            "channels": raw.get("encoder", {}).get("conv_channels"),
            "latent_dim": raw.get("encoder", {}).get("latent_dim"),
            "path": f,
        })
    return result


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

@app.command()
def presets() -> None:
    """List built-in encoder presets (r1–r4) with their architecture and use-case."""
    from rich.text import Text

    n_col = ("n", "box_radius", "channels", "latent_dim", "params (enc)", "use case")
    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 2, 0, 0))
    tbl.add_column("preset", style="bold cyan")
    tbl.add_column("radius", justify="right")
    tbl.add_column("n", justify="right")
    tbl.add_column("channels")
    tbl.add_column("latent", justify="right")
    tbl.add_column("use case", style="dim")

    use_cases = {
        "r1": "collision check / tight spaces",
        "r2": "standard local nav (default)",
        "r3": "corner planning / wider view",
        "r4": "complex geometry / full local env",
        "r8": "extended range / open terrain",
        "r16": "macro spatial reasoning (GPU required)",
    }

    for p in _list_presets():
        r = p["box_radius"]
        n = 2 * r + 1 if r else "?"
        tbl.add_row(
            p["name"],
            str(r),
            str(n),
            str(p["channels"]),
            str(p["latent_dim"]),
            use_cases.get(p["name"], ""),
        )

    console.print("\n[bold]Built-in garden presets[/bold]\n")
    console.print(tbl)
    console.print(
        "\n[dim]Train any preset:[/dim]  anysearch garden train r2 --tag v1"
        "\n[dim]Use in RL config:[/dim]  pretrained_encoder: r2:v1\n"
    )


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@app.command()
def extract(
    config: str = typer.Argument(..., help="Garden YAML, data YAML, or registered model name[:tag]."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Override cache location."),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Re-collect, ignoring existing cache."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be collected without writing."),
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir"),
) -> None:
    """Collect and cache observation data without training an encoder."""
    import json as _json
    from theseo_anysearch.garden.data_config import load_data_config, load_garden_config, DataConfig
    from theseo_anysearch.garden.store import GardenStore

    config_path = _resolve_config_path(config)

    if config_path is not None:
        # File path: garden YAML or standalone data YAML
        try:
            garden_cfg = load_garden_config(config_path)
            data_cfg = garden_cfg.data if not isinstance(garden_cfg.data, Path) else load_data_config(garden_cfg.data)
            box_radius = garden_cfg.encoder.box_radius
        except Exception:
            data_cfg = load_data_config(config_path)
            box_radius = 2
    else:
        # Model name[:tag] — look up in garden registry
        try:
            model_dir, _ = GardenStore.resolve_ref(config)
        except KeyError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        meta_file = model_dir / "meta.json"
        if not meta_file.exists():
            err_console.print(f"[red]Error:[/red] meta.json not found in {model_dir}")
            raise typer.Exit(1)
        meta = _json.loads(meta_file.read_text())
        data_sources = meta.get("data_sources")
        if not data_sources:
            err_console.print(f"[red]Error:[/red] no data_sources in meta for {config!r}")
            raise typer.Exit(1)
        data_cfg = DataConfig.model_validate(data_sources)
        box_radius = meta.get("box_radius", 2)
        if cache_dir is None and garden_dir:
            cache_dir = garden_dir / "cache"

    if cache_dir:
        data_cfg.cache.path = cache_dir
    if force_refresh:
        data_cfg.cache.force_refresh = True

    if dry_run:
        _print_dry_run(data_cfg)
        return

    _run_extract(data_cfg, box_radius)


def _print_dry_run(data_cfg) -> None:
    from theseo_anysearch.garden.collect import _obs_cache_path
    from theseo_anysearch.garden.data_config import PositionSampleSourceConfig, TrainingRunSourceConfig

    table = Table(box=None, show_header=True, padding=(0, 2, 0, 0))
    table.add_column("#", style="dim")
    table.add_column("Type")
    table.add_column("Geometry / Run")
    table.add_column("Samples", justify="right")
    table.add_column("Cached")

    for i, src in enumerate(data_cfg.sources, 1):
        cache_path = _obs_cache_path(data_cfg.cache, src)
        cached = "yes" if cache_path.exists() else "no"
        if isinstance(src, PositionSampleSourceConfig):
            geo = src.geometry
            if geo.env:
                desc = f"env ({geo.env.geometry__stl_path or 'geometry_boxes'})"
                n = src.num_samples or 10000
            elif geo.sampler:
                desc = f"{geo.sampler.type}"
                n = geo.sampler.total_samples
            else:
                desc, n = "?", 0
            table.add_row(str(i), "position_sample", desc, str(n), cached)
        elif isinstance(src, TrainingRunSourceConfig):
            table.add_row(str(i), "training_run", src.run, "?", cached)

    meta = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    meta.add_column(style="dim", min_width=12)
    meta.add_column()
    meta.add_row("config", "dry run")
    meta.add_row("cache dir", str(data_cfg.cache.path))

    from rich.console import Group
    console.print(Panel(Group(meta), title="[bold]Garden extract — dry run[/bold]", title_align="left"))
    console.print(table)


def _run_extract(data_cfg, box_radius: int) -> None:
    from theseo_anysearch.garden.collect import collect_dataset

    total_collected = 0

    with Progress(
        SpinnerColumn("line"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting dataset", total=None)

        def cb(label: str, n: int) -> None:
            nonlocal total_collected
            total_collected += n
            progress.advance(task, n)
            progress.update(task, description=f"[cyan]{label}[/cyan]")

        grids = collect_dataset(data_cfg, box_radius=box_radius, progress_cb=cb)

    n_val = max(1, int(len(grids) * data_cfg.split.val_fraction))
    n_train = len(grids) - n_val

    summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    summary.add_column(style="dim", min_width=18)
    summary.add_column()
    summary.add_row("total samples", str(len(grids)))
    summary.add_row("train", f"{n_train}  ({100 * n_train // len(grids)}%)")
    summary.add_row("val", f"{n_val}  ({100 * n_val // len(grids)}%)")
    summary.add_row("cache dir", str(data_cfg.cache.path))

    console.print(Panel(summary, title="[bold]Garden extract complete[/bold]", title_align="left"))


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    config: str = typer.Argument(..., help="Garden YAML config path or registered name."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override garden store location."),
    tag: Optional[str] = typer.Option(None, "--tag", help="Version tag for this model (e.g. v1, v2)."),
    resume: bool = typer.Option(False, "--resume", help="Resume from last checkpoint."),
    device: str = typer.Option("auto", "--device", help="Torch device: cpu | cuda | cuda:0"),
) -> None:
    """Pre-train an encoder and save it to the model garden."""
    from theseo_anysearch.garden.collect import collect_dataset
    from theseo_anysearch.garden.data_config import load_garden_config
    from theseo_anysearch.garden.dataset import make_multi_radius_train_val, make_train_val_datasets
    from theseo_anysearch.garden.encoder import build_model
    from theseo_anysearch.garden.store import GardenStore
    from theseo_anysearch.garden.trainer import train as run_train

    config_path = _resolve_config_path(config)
    if config_path is None:
        err_console.print(f"[red]Error:[/red] config not found: {config!r}  (pass a file path or a registered name)")
        raise typer.Exit(1)
    config = config_path  # type: ignore[assignment]

    cfg = load_garden_config(config)
    garden_dir = output_dir or _DEFAULT_GARDEN
    store = GardenStore(garden_dir)
    dev = _resolve_device(device)

    if isinstance(cfg.data, Path):
        from theseo_anysearch.garden.data_config import load_data_config
        data_cfg = load_data_config(cfg.data)
    else:
        data_cfg = cfg.data

    radii = cfg.encoder.box_radii or [cfg.encoder.box_radius]

    # --- Collect data ---
    console.print(f"\n[bold]Collecting data[/bold]  ({len(data_cfg.sources)} source(s), radii={radii})")
    with Progress(
        SpinnerColumn("line"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting", total=None)

        def data_cb(label: str, n: int) -> None:
            progress.advance(task, n)
            progress.update(task, description=f"[cyan]{label}[/cyan]")

        if len(radii) == 1:
            grids = collect_dataset(data_cfg, box_radius=radii[0], progress_cb=data_cb)
            train_ds, val_ds = make_train_val_datasets(grids, cfg.augmentation, data_cfg.split)
        else:
            grids_per_radius = {
                r: collect_dataset(data_cfg, box_radius=r, progress_cb=data_cb)
                for r in radii
            }
            train_ds, val_ds = make_multi_radius_train_val(grids_per_radius, cfg.augmentation, data_cfg.split)

    console.print(f"  train {len(train_ds)}  val {len(val_ds)}\n")

    # --- Build model ---
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model[/bold]  {cfg.architecture}  {cfg.approach}  {n_params:,} params  device={dev}\n")

    checkpoint_path = garden_dir / cfg.name / "_checkpoint.pt" if resume else None

    # --- Train ---
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("train [green]{task.fields[train]:.4f}[/green]  val [cyan]{task.fields[val]:.4f}[/cyan]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Training",
            total=cfg.training.epochs,
            train=float("inf"),
            val=float("inf"),
        )

        def epoch_cb(epoch: int, train_loss: float, val_loss: float) -> None:
            progress.update(task, advance=1, train=train_loss, val=val_loss)

        result = run_train(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg.training,
            approach=cfg.approach,
            device=dev,
            epoch_cb=epoch_cb,
            checkpoint_path=checkpoint_path,
        )

    # --- Save ---
    ckpt_path = store.save(
        name=cfg.name,
        model=model,
        cfg=cfg,
        epochs_trained=result.epochs_trained,
        loss_curve=result.loss_curve,
        final_val_loss=result.final_val_loss,
        tag=tag,
    )

    ref = f"{cfg.name}:{tag}" if tag else cfg.name
    early = f"  (early stop, patience {cfg.training.early_stop_patience})" if result.stopped_early else ""
    summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    summary.add_column(style="dim", min_width=18)
    summary.add_column()
    summary.add_row("model", ref)
    summary.add_row("approach", cfg.approach)
    summary.add_row("epochs", f"{result.epochs_trained} / {cfg.training.epochs}{early}")
    summary.add_row("final val loss", f"{result.final_val_loss:.4f}")
    summary.add_row("saved to", str(ckpt_path))
    summary.add_row("use in RL", f"pretrained_encoder: {ref}")

    console.print(Panel(summary, title="[bold]Garden train complete[/bold]", title_align="left"))


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

@app.command()
def add(
    run_ref: str = typer.Argument(..., help="Garden YAML path to register, or a training run ref (e.g. multi_agent_ppo_asha:v3)."),
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir", help="Garden store location."),
    obs_key: str = typer.Option("local_grid", "--obs-key", help="Observation field to extract (training runs only)."),
    box_radius: int = typer.Option(2, "--box-radius", help="Box radius used during training (training runs only)."),
) -> None:
    """Register a garden YAML by name, or register a completed training run as a data source."""
    from theseo_anysearch.garden.store import register_config

    # If it's a YAML file path, register it as a named config.
    yaml_path = Path(run_ref)
    if yaml_path.suffix in (".yaml", ".yml") and yaml_path.exists():
        name = yaml_path.stem
        register_config(name, yaml_path)
        summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
        summary.add_column(style="dim", min_width=16)
        summary.add_column()
        summary.add_row("registered", name)
        summary.add_row("path", str(yaml_path.resolve()))
        summary.add_row("use as", f"anysearch garden extract {name}")
        console.print(Panel(summary, title="[bold]Garden add[/bold]", title_align="left"))
        return

    # Otherwise treat as a training run reference.
    from theseo_anysearch.garden.collect import TrainingRunCollector, _source_hash
    from theseo_anysearch.garden.data_config import CacheConfig, TrainingRunSourceConfig
    from theseo_anysearch.garden.store import GardenStore
    from theseo_anysearch.cli.registry import resolve_ref
    from theseo_anysearch.experiments.runner import _find_run_dir

    gdir = garden_dir or _DEFAULT_GARDEN
    store = GardenStore(gdir)
    cache_cfg = CacheConfig(path=gdir / "cache")

    # Resolve run directory
    try:
        experiment_dir, run_id = resolve_ref(run_ref)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] could not resolve {run_ref!r}: {exc}")
        raise typer.Exit(1)

    search = experiment_dir
    try:
        run_dir = _find_run_dir(search, run_id) if run_id else experiment_dir
    except FileNotFoundError:
        run_dir = experiment_dir

    source = TrainingRunSourceConfig(type="training_run", run=run_ref, obs_key=obs_key)

    with Progress(SpinnerColumn("line"), TextColumn("{task.description}"), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task(f"Extracting from {run_ref}", total=None)

        def cb(label: str, n: int) -> None:
            progress.advance(task, n)

        collector = TrainingRunCollector(source, cache_cfg, box_radius)
        grids = collector.collect(run_dir, lambda n: cb("", n))

    cache_hash = _source_hash(source)
    store.register_run(run_ref, run_dir, len(grids), cache_hash)

    summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    summary.add_column(style="dim", min_width=16)
    summary.add_column()
    summary.add_row("run", run_ref)
    summary.add_row("run dir", str(run_dir))
    summary.add_row("observations", f"{len(grids)} {obs_key} samples extracted")
    summary.add_row("cached to", str(cache_cfg.path / "obs" / f"{cache_hash}.npz"))
    summary.add_row("use as source", f"type: training_run\nrun: {run_ref}")

    console.print(Panel(summary, title="[bold]Garden add[/bold]", title_align="left"))


# ---------------------------------------------------------------------------
# list helpers
# ---------------------------------------------------------------------------

def _display_path(raw: str) -> str:
    """Return path relative to CWD when it requires fewer than 2 levels up, otherwise absolute."""
    import os
    rel = os.path.relpath(raw)
    dotdots = rel.replace("\\", "/").split("/").count("..")
    return rel if dotdots < 2 else raw


def _format_stl_list(paths: list[str], n, verbose: bool) -> str:
    """Format a list of STL paths: n samples header, then dirs grouped with one file per line."""
    import os
    from collections import defaultdict

    display = [_display_path(p) for p in paths]
    cap = len(display) if verbose else 10
    shown, hidden = display[:cap], display[cap:]

    # Group files by parent directory
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in shown:
        by_dir[os.path.dirname(p)].append(os.path.basename(p))

    lines = [f"[dim]{n} samples[/dim]"]
    for dir_path, filenames in by_dir.items():
        lines.append(f"[dim]{dir_path}[/dim]{os.sep}")
        for fname in filenames:
            lines.append(f"  {fname}")

    if hidden:
        lines.append(f"[dim]+{len(hidden)} more  (-v for all)[/dim]")

    return "\n".join(lines)


def _format_data_sources(data: dict, verbose: bool = False) -> list[tuple[str, str]]:
    """Return (label, text) rows summarising data sources for garden list output."""
    sources = data.get("sources", [])
    rows: list[tuple[str, str]] = []
    for i, src in enumerate(sources):
        label = "data" if i == 0 else ""
        stype = src.get("type", "?")
        if stype == "position_sample":
            geo = src.get("geometry", {})
            sampler = geo.get("sampler") or {}
            stype_s = sampler.get("type", "")
            if stype_s == "stl_list":
                paths = sampler.get("paths") or []
                n = sampler.get("total_samples", "?")
                rows.append((label, _format_stl_list(paths, n, verbose)))
            elif stype_s == "random_boxes":
                rows.append((label, f"random_boxes  {sampler.get('total_samples', '?')} samples"))
            elif geo.get("env"):
                rows.append((label, f"env  {src.get('num_samples', '?')} samples"))
            else:
                rows.append((label, "position_sample"))
        elif stype == "training_run":
            rows.append((label, f"training_run  {src.get('run', '?')}"))
        else:
            rows.append((label, stype))
    return rows


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_models(
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show all source files (default: max 10)."),
) -> None:
    """List all pre-trained encoders in the garden store."""
    from theseo_anysearch.garden.store import GardenStore

    store = GardenStore(garden_dir or _DEFAULT_GARDEN)
    models = store.list_models()

    if not models:
        console.print("[dim]No models in garden store.[/dim]")
        raise typer.Exit()

    for m in models:
        row = Table(box=None, show_header=False, padding=(0, 1, 0, 0))
        row.add_column(style="dim", min_width=12, no_wrap=True)
        row.add_column(overflow="fold")
        row.add_row("approach", m.get("approach", "?"))
        row.add_row("architecture", m.get("architecture", "?"))
        row.add_row("latent_dim", str(m.get("latent_dim", "?")))
        row.add_row("val_loss", f"{m['final_val_loss']:.4f}" if "final_val_loss" in m else "?")
        row.add_row("created", m.get("created_at", "?")[:10])
        for label, text in _format_data_sources(m.get("data_sources", {}), verbose=verbose):
            row.add_row(label, text)
        name = m.get("name", "?")
        tag = m.get("tag")
        title = f"[bold cyan]{name}[/bold cyan]" + (f"[dim]:{tag}[/dim]" if tag else "")
        console.print(Panel(row, title=title, title_align="left"))

    console.print("\n  To use in RL:   [dim]pretrained_encoder: <name>[/dim]")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@app.command()
def inspect(
    ref: str = typer.Argument(..., help="Model name[:tag] or path to model directory."),
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir"),
) -> None:
    """Print metadata and loss curve for a pre-trained encoder."""
    import json as _json
    from theseo_anysearch.garden.store import GardenStore

    # Resolve via global registry first, then fall back to store-local lookup
    try:
        model_dir, _ = GardenStore.resolve_ref(ref)
    except KeyError:
        # Not in global registry — try as a bare name in the local store
        store = GardenStore(garden_dir or _DEFAULT_GARDEN)
        model_dir = store._model_dir(ref)

    meta_file = model_dir / "meta.json"
    curve_file = model_dir / "loss_curve.json"
    if not meta_file.exists():
        err_console.print(f"[red]Error:[/red] no model found for {ref!r}")
        raise typer.Exit(1)
    meta = _json.loads(meta_file.read_text())
    curve = _json.loads(curve_file.read_text()) if curve_file.exists() else []

    detail = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    detail.add_column(style="dim", min_width=18)
    detail.add_column()
    for k, v in meta.items():
        if k == "data_sources":
            continue  # shown separately via _format_data_sources
        if k == "augmentation":
            detail.add_row(k, "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
        else:
            detail.add_row(k, str(v))
    for label, text in _format_data_sources(meta.get("data_sources", {}), verbose=True):
        detail.add_row(label, text)

    panel_title = f"[bold]{meta.get('name', ref)}[/bold]" + (f"[dim]:{meta['tag']}[/dim]" if meta.get("tag") else "")
    console.print(Panel(detail, title=panel_title, title_align="left"))

    if curve:
        loss_table = Table(padding=(0, 2, 0, 0))
        loss_table.add_column("Epoch", justify="right", style="dim")
        loss_table.add_column("Train loss", justify="right")
        loss_table.add_column("Val loss", justify="right")
        step = max(1, len(curve) // 10)
        for entry in curve[::step]:
            loss_table.add_row(
                str(entry["epoch"]),
                f"{entry['train']:.4f}",
                f"{entry['val']:.4f}",
            )
        if curve[-1] not in curve[::step]:
            e = curve[-1]
            loss_table.add_row(str(e["epoch"]), f"{e['train']:.4f}", f"{e['val']:.4f}")
        console.print(loss_table)


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

@app.command()
def tune(
    config: str = typer.Argument(..., help="Garden YAML (with tune: block) or registered config name."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Ray Tune trial storage dir."),
    tag: Optional[str] = typer.Option(None, "--tag", help="Tag appended to the run name."),
    save_best: bool = typer.Option(True, "--save-best/--no-save-best", help="Save best trial to garden store."),
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir"),
) -> None:
    """Hyperparameter search over garden encoder architecture and training config."""
    from theseo_anysearch.garden.tune_runner import GardenTuneRunner
    from theseo_anysearch.garden.store import GardenStore
    from theseo_anysearch.garden.data_config import load_garden_config

    config_path = _resolve_config_path(config)
    if config_path is None:
        err_console.print(f"[red]Error:[/red] config not found: {config!r}")
        raise typer.Exit(1)

    out = output_dir or Path("runtime/garden/tune")
    gdir = garden_dir or _DEFAULT_GARDEN

    runner = GardenTuneRunner(config_path, tag=tag)
    tc = runner._cfg.tune

    summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    summary.add_column(style="dim", min_width=18)
    summary.add_column()
    summary.add_row("model", runner._cfg.name)
    summary.add_row("trials", str(tc.num_samples))
    summary.add_row("max concurrent", str(tc.max_concurrent))
    summary.add_row("epochs / trial", str(runner._cfg.training.epochs))
    summary.add_row("storage", str(out.resolve()))
    console.print(Panel(summary, title="[bold]Garden tune[/bold]", title_align="left"))

    result = runner.run(out)

    best_summary = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    best_summary.add_column(style="dim", min_width=18)
    best_summary.add_column()
    best_summary.add_row("trials run", str(result["num_trials"]))
    best_summary.add_row("best val_loss", f"{result['best_val_loss']:.4f}")
    for k, v in result["best_config"].items():
        best_summary.add_row(k, str(v))
    console.print(Panel(best_summary, title="[bold]Garden tune — best trial[/bold]", title_align="left"))

    if save_best:
        # Re-train with best config and save to store
        import copy
        from theseo_anysearch.garden.collect import collect_dataset
        from theseo_anysearch.garden.data_config import GardenConfig, load_data_config
        from theseo_anysearch.garden.dataset import make_train_val_datasets
        from theseo_anysearch.garden.encoder import build_model
        from theseo_anysearch.garden.trainer import train as run_train
        from theseo_anysearch.garden.tune_runner import _deep_set

        cfg_dict = runner._cfg.model_dump(mode="json")
        cfg_dict.pop("tune", None)
        for k, v in result["best_config"].items():
            _deep_set(cfg_dict, k, v)
        best_cfg = GardenConfig.model_validate(cfg_dict)

        if isinstance(best_cfg.data, Path):
            data_cfg = load_data_config(best_cfg.data)
        else:
            data_cfg = best_cfg.data

        console.print("\n[bold]Re-training best config...[/bold]")
        grids = collect_dataset(data_cfg, box_radius=best_cfg.encoder.box_radius)
        train_ds, val_ds = make_train_val_datasets(grids, best_cfg.augmentation, data_cfg.split)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(best_cfg)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(),
            TextColumn("val [cyan]{task.fields[val]:.4f}[/cyan]"),
            TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task("Training best", total=best_cfg.training.epochs, val=float("inf"))
            def epoch_cb(epoch, train_loss, val_loss):
                progress.update(task, advance=1, val=val_loss)
            result_tr = run_train(model=model, train_ds=train_ds, val_ds=val_ds,
                                  cfg=best_cfg.training, approach=best_cfg.approach,
                                  device=device, epoch_cb=epoch_cb)

        store = GardenStore(gdir)
        ckpt = store.save(name=best_cfg.name, model=model, cfg=best_cfg,
                          epochs_trained=result_tr.epochs_trained,
                          loss_curve=result_tr.loss_curve,
                          final_val_loss=result_tr.final_val_loss,
                          tag=tag)
        console.print(f"\n  Saved best encoder -> [bold]{ckpt}[/bold]")


# ---------------------------------------------------------------------------
# show-data
# ---------------------------------------------------------------------------

@app.command(name="show-data")
def show_data(
    config: str = typer.Argument(..., help="Data YAML, garden YAML, or registered config name."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name[:tag] to overlay reconstructions."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory containing .npz files."),
    garden_dir: Optional[Path] = typer.Option(None, "--garden-dir"),
) -> None:
    """Launch an interactive 3-D viewer for cached garden observation samples.

    Pass --model <name> to also display reconstructions from the trained autoencoder.
    """
    import json as _json
    import os
    import struct
    import tempfile
    import subprocess
    import numpy as np
    from theseo_anysearch.garden.collect import _obs_cache_path
    from theseo_anysearch.garden.data_config import DataConfig, load_data_config, load_garden_config
    from theseo_anysearch.garden.store import GardenStore

    # --- Resolve DataConfig ---
    config_path = _resolve_config_path(config)
    box_radius = 2

    if config_path is not None:
        try:
            garden_cfg = load_garden_config(config_path)
            data_cfg = garden_cfg.data if not isinstance(garden_cfg.data, Path) else load_data_config(garden_cfg.data)
            box_radius = garden_cfg.encoder.box_radius
        except Exception:
            data_cfg = load_data_config(config_path)
    else:
        try:
            model_dir, _ = GardenStore.resolve_ref(config)
        except KeyError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        meta_file = model_dir / "meta.json"
        if not meta_file.exists():
            err_console.print(f"[red]Error:[/red] meta.json not found in {model_dir}")
            raise typer.Exit(1)
        meta = _json.loads(meta_file.read_text())
        data_sources = meta.get("data_sources")
        if not data_sources:
            err_console.print(f"[red]Error:[/red] no data_sources in meta for {config!r}")
            raise typer.Exit(1)
        data_cfg = DataConfig.model_validate(data_sources)
        box_radius = meta.get("box_radius", 2)

    if cache_dir:
        data_cfg.cache.path = cache_dir

    # --- Locate cached .npz files ---
    npz_paths = [_obs_cache_path(data_cfg.cache, src) for src in data_cfg.sources]
    npz_paths = [p for p in npz_paths if p.exists()]
    if not npz_paths:
        err_console.print("[red]Error:[/red] No cached .npz files found. Run [bold]anysearch garden extract[/bold] first.")
        raise typer.Exit(1)

    grids = np.concatenate([np.load(p)["observations"] for p in npz_paths])  # (N, n, n, n)
    N, n = len(grids), grids.shape[1]
    console.print(f"[dim]Loaded {N} samples  (n={n}, box_radius={box_radius})[/dim]")

    # --- Optional: compute reconstructions ---
    recon: np.ndarray | None = None
    if model:
        recon = _compute_reconstructions(grids, model, box_radius)

    # --- Write .gobs binary file ---
    # Header: "GOBS" (4) + u32 N + u8 n + u8 flags + 2 pad bytes
    # flags bit 0 = has_recon
    # Body:  f32[N*n³] inputs  [+ f32[N*n³] recons if has_recon]
    flags = 0x01 if recon is not None else 0x00
    with tempfile.NamedTemporaryFile(suffix=".gobs", delete=False) as tf:
        gobs_path = tf.name
        tf.write(b"GOBS")
        tf.write(struct.pack("<I", N))
        tf.write(struct.pack("<B", n))
        tf.write(struct.pack("<B", flags))
        tf.write(b"\x00\x00")  # 2 pad bytes
        tf.write(grids.astype(np.float32).tobytes())
        if recon is not None:
            tf.write(recon.astype(np.float32).tobytes())

    # --- Find the Rust binary ---
    import importlib.util
    core_spec = importlib.util.find_spec("theseo_core")
    binary_name = "garden-show-data.exe" if sys.platform == "win32" else "garden-show-data"

    candidate_dirs = []
    if core_spec and core_spec.origin:
        pkg_dir = Path(core_spec.origin).parent
        candidate_dirs += [pkg_dir, pkg_dir.parent]
    candidate_dirs += [
        Path(__file__).parents[3] / "theseo_anysearch" / "core" / "target" / "release",
        Path(__file__).parents[3] / "theseo_anysearch" / "core" / "target" / "debug",
    ]

    binary_path = next((d / binary_name for d in candidate_dirs if (d / binary_name).exists()), None)
    if binary_path is None:
        err_console.print(
            f"[red]Error:[/red] {binary_name} not found.\n"
            "Build it with:\n  cd theseo_anysearch/core && cargo build --release --bin garden-show-data"
        )
        os.unlink(gobs_path)
        raise typer.Exit(1)

    console.print(f"[dim]Launching viewer: {binary_path}[/dim]")
    subprocess.run([str(binary_path), gobs_path], check=False)
    os.unlink(gobs_path)


def _compute_reconstructions(grids: "np.ndarray", model_ref: str, box_radius: int) -> "np.ndarray | None":
    """Run the full AE forward pass on *grids* and return reconstructions (N, n, n, n)."""
    import json as _json
    import numpy as np
    import torch
    from theseo_anysearch.garden.store import GardenStore

    try:
        model_dir, _ = GardenStore.resolve_ref(model_ref)
    except KeyError as exc:
        err_console.print(f"[yellow]Warning:[/yellow] --model {model_ref!r} not found: {exc}")
        return None

    meta_file = model_dir / "meta.json"
    model_pt = model_dir / "model.pt"
    if not meta_file.exists() or not model_pt.exists():
        err_console.print(
            "[yellow]Warning:[/yellow] model.pt not found — re-train to enable reconstructions.\n"
            "  (Only encoder.pt is saved for older runs.)"
        )
        return None

    meta = _json.loads(meta_file.read_text())
    architecture = meta.get("architecture", "voxel_box_3dcnn")
    if architecture != "voxel_box_3dcnn":
        err_console.print(
            f"[yellow]Warning:[/yellow] Reconstruction display requires voxel_box_3dcnn (got {architecture!r})."
        )
        return None

    from theseo_anysearch.garden.models.ae import VoxelAE
    n = 2 * box_radius + 1
    ae = VoxelAE(architecture=architecture, n=n, channels=meta["conv_channels"], latent_dim=meta["latent_dim"])
    ae.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
    ae.eval()

    batch_size = 256
    recon_parts = []
    with torch.no_grad():
        for i in range(0, len(grids), batch_size):
            x = torch.from_numpy(grids[i:i + batch_size])
            r, _ = ae(x)              # (B, 1, n, n, n)
            recon_parts.append(r.squeeze(1).numpy())

    recon = np.concatenate(recon_parts)
    console.print(f"[dim]Reconstructions computed ({len(recon)} samples)[/dim]")
    return recon
