"""Artifact and interactive report generation for resource benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from theseo_anysearch.benchmarking.models import (
    CandidateSummary,
    ResourceBenchmarkResult,
)


def write_progress_report(
    *,
    environment_candidates: list[CandidateSummary],
    worker_candidates: list[CandidateSummary],
    output_dir: Path,
    max_envs_per_worker: int,
    max_workers: int,
) -> Path:
    """Atomically refresh the inspectable HTML while a benchmark is running."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "report.html"
    temporary_path = output_dir / "report.progress.html"
    figure = make_subplots(
        rows=4,
        cols=1,
        specs=[[{"secondary_y": True}] for _ in range(4)],
        subplot_titles=(
            "Environment vectorization",
            "Environment sweep resources",
            "Rollout worker scaling",
            "Worker sweep resources",
        ),
        vertical_spacing=0.08,
    )

    for throughput_row, resource_row, candidates, x_title, waiting in (
        (1, 2, environment_candidates, "Environments per worker",
         "Waiting for the first environment candidate"),
        (3, 4, worker_candidates, "Rollout workers",
         "Waiting for the environment sweep to finish"),
    ):
        if not candidates:
            figure.add_annotation(
                text=waiting,
                x=0.5,
                y=0.5,
                xref=f"x{throughput_row} domain",
                yref=f"y{throughput_row} domain",
                showarrow=False,
                font={"color": "#868e96"},
            )
            continue
        x_values = [candidate.candidate for candidate in candidates]
        gpu = [candidate.gpu_utilization_percent for candidate in candidates]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[candidate.steps_per_second for candidate in candidates],
                name=f"{x_title} steps/s",
                mode="lines+markers",
                line={"color": "#087f5b", "width": 3},
            ),
            row=throughput_row,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[candidate.speedup for candidate in candidates],
                name=f"{x_title} speedup",
                mode="lines+markers",
                line={"color": "#e67700", "dash": "dot", "width": 2},
            ),
            row=throughput_row,
            col=1,
            secondary_y=True,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[candidate.cpu_percent for candidate in candidates],
                name=f"{x_title} CPU %",
                mode="lines+markers",
                line={"color": "#1971c2", "width": 2},
            ),
            row=resource_row,
            col=1,
            secondary_y=False,
        )
        if any(value is not None for value in gpu):
            figure.add_trace(
                go.Bar(
                    x=x_values,
                    y=gpu,
                    name=f"{x_title} GPU average %",
                    text=gpu,
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    marker_color="rgba(116,192,252,0.35)",
                ),
                row=resource_row,
                col=1,
                secondary_y=False,
            )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[candidate.memory_mb for candidate in candidates],
                name=f"{x_title} memory MB",
                mode="lines+markers",
                line={"color": "#862e9c", "dash": "dot", "width": 2},
            ),
            row=resource_row,
            col=1,
            secondary_y=True,
        )
        for row in (throughput_row, resource_row):
            figure.update_xaxes(title_text=x_title, dtick=1, row=row, col=1)

    figure.update_layout(
        title={
            "text": (
                "AnySearch resource benchmark — in progress"
                f"<br><sup>Environment ticks: {len(environment_candidates)}/"
                f"{max_envs_per_worker}; worker ticks: {len(worker_candidates)}/"
                f"{max_workers}. This page refreshes every 5 seconds.</sup>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        height=1680,
        hovermode="x unified",
        margin={"l": 80, "r": 80, "t": 170, "b": 70},
    )
    figure.write_html(temporary_path, include_plotlyjs=True, full_html=True)
    document = temporary_path.read_text(encoding="utf-8")
    document = document.replace(
        "<head>",
        '<head><meta http-equiv="refresh" content="5">',
        1,
    )
    temporary_path.write_text(document, encoding="utf-8")
    temporary_path.replace(html_path)
    _add_diagnostic_links(html_path)
    return html_path


def write_benchmark_artifacts(
    result: ResourceBenchmarkResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write JSON, CSV, YAML, and standalone Plotly HTML artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    yaml_path = output_dir / "recommended.yaml"
    html_path = output_dir / "report.html"

    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    _write_csv(result, csv_path)
    _write_recommendation(result, yaml_path)
    _write_html(result, html_path)
    return {
        "json": json_path,
        "csv": csv_path,
        "yaml": yaml_path,
        "html": html_path,
    }


def _write_csv(result: ResourceBenchmarkResult, path: Path) -> None:
    fields = [
        "phase",
        "candidate",
        "repeat",
        "num_env_runners",
        "num_envs_per_env_runner",
        "wall_seconds",
        "sampled_steps",
        "steps_per_second",
        "speedup",
        "cpu_percent",
        "memory_mb",
        "gpu_utilization_percent",
        "gpu_memory_mb",
        "gpu_power_watts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sweep in (result.environment_sweep, result.worker_sweep):
            for candidate in sweep.candidates:
                for sample in candidate.samples:
                    row = sample.model_dump()
                    row["speedup"] = candidate.speedup
                    writer.writerow(
                        {field: row.get(field)
                         for field in fields})


def _write_recommendation(result: ResourceBenchmarkResult, path: Path) -> None:
    recommendation = result.recommendation
    path.write_text(
        "training:\n"
        f"  num_env_runners: {recommendation.num_env_runners}\n"
        f"  num_envs_per_env_runner: {recommendation.num_envs_per_env_runner}\n"
        "  num_gpus_per_env_runner: 0.0\n",
        encoding="utf-8",
    )


def _write_html(result: ResourceBenchmarkResult, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_prediction = result.prediction is not None
    row_count = 5 if has_prediction else 4
    specs = [[{"secondary_y": True}] for _ in range(4)]
    subplot_titles = [
        "Environment vectorization",
        "Environment sweep resources",
        "Rollout worker scaling",
        "Worker sweep resources",
    ]
    if has_prediction:
        specs.append([{"secondary_y": False}])
        subplot_titles.append(
            "Calibration: predicted vs. confirmed throughput")

    figure = make_subplots(
        rows=row_count,
        cols=1,
        specs=specs,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.06,
    )

    for throughput_row, resource_row, sweep, x_title in (
        (1, 2, result.environment_sweep, "Environments per worker"),
        (3, 4, result.worker_sweep, "Rollout workers"),
    ):
        throughput_legend = "legend" if throughput_row == 1 else "legend3"
        resource_legend = f"legend{resource_row}"
        x_values = [candidate.candidate for candidate in sweep.candidates]
        throughput = [
            candidate.steps_per_second for candidate in sweep.candidates
        ]
        speedup = [candidate.speedup for candidate in sweep.candidates]
        gpu = [
            candidate.gpu_utilization_percent for candidate in sweep.candidates
        ]
        cpu = [candidate.cpu_percent for candidate in sweep.candidates]
        memory = [candidate.memory_mb for candidate in sweep.candidates]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=throughput,
                name="Steps/s",
                legend=throughput_legend,
                mode="lines+markers",
                line={
                    "color": "#087f5b",
                    "width": 3
                },
            ),
            row=throughput_row,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=speedup,
                name="Speedup",
                legend=throughput_legend,
                mode="lines+markers",
                line={
                    "color": "#e67700",
                    "dash": "dot",
                    "width": 2
                },
            ),
            row=throughput_row,
            col=1,
            secondary_y=True,
        )
        stop_candidate = sweep.candidates[-1]
        figure.add_trace(
            go.Scatter(
                x=[stop_candidate.candidate],
                y=[stop_candidate.steps_per_second],
                name="Sweep stop",
                legend=throughput_legend,
                mode="markers",
                marker={
                    "color": "#212529",
                    "size": 11,
                    "symbol": "x",
                },
                hovertemplate=(f"{sweep.stop_reason}<br>"
                               "%{x}: %{y:.1f} steps/s<extra></extra>"),
            ),
            row=throughput_row,
            col=1,
            secondary_y=False,
        )
        if any(value is not None for value in cpu):
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=cpu,
                    name="Rollout CPU %",
                    legend=resource_legend,
                    mode="lines+markers",
                    line={
                        "color": "#1971c2",
                        "width": 2
                    },
                ),
                row=resource_row,
                col=1,
                secondary_y=False,
            )
        if any(value is not None for value in memory):
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=memory,
                    name="Rollout memory MB",
                    legend=resource_legend,
                    mode="lines+markers",
                    line={
                        "color": "#862e9c",
                        "dash": "dot",
                        "width": 2
                    },
                ),
                row=resource_row,
                col=1,
                secondary_y=True,
            )
        if any(value is not None for value in gpu):
            figure.add_trace(
                go.Bar(
                    x=x_values,
                    y=gpu,
                    name="GPU average %",
                    legend=resource_legend,
                    text=gpu,
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    textfont={
                        "color": "#212529",
                        "size": 12
                    },
                    cliponaxis=False,
                    marker_color="rgba(116,192,252,0.35)",
                ),
                row=resource_row,
                col=1,
                secondary_y=False,
            )
        figure.add_vline(
            x=sweep.peak_candidate,
            line_color="#c92a2a",
            line_dash="dash",
            row=throughput_row,
            col=1,
        )
        if sweep.phase == "workers":
            figure.add_hline(
                y=result.max_gpu_utilization,
                line_color="#495057",
                line_dash="dot",
                annotation_text="GPU target",
                annotation_position="top left",
                row=resource_row,
                col=1,
            )
        figure.add_vline(
            x=sweep.peak_candidate,
            line_color="#c92a2a",
            line_dash="dash",
            row=resource_row,
            col=1,
        )
        for row in (throughput_row, resource_row):
            figure.update_xaxes(title_text=x_title, dtick=1, row=row, col=1)
        figure.update_yaxes(title_text="Sampled steps / second",
                            row=throughput_row,
                            col=1)
        figure.update_yaxes(
            title_text="Speedup",
            row=throughput_row,
            col=1,
            secondary_y=True,
        )
        figure.update_yaxes(title_text="CPU / GPU utilization %",
                            row=resource_row,
                            col=1)
        figure.update_yaxes(
            title_text="Rollout memory MB",
            row=resource_row,
            col=1,
            secondary_y=True,
        )

    if has_prediction:
        _add_prediction_panel(figure, result, row=5)

    recommendation = result.recommendation
    environment_rule = (
        "Automatic environment sweep: stop after "
        f"{result.decline_patience} throughput decline(s) beyond "
        f"{result.decline_tolerance:.0%} tolerance")
    worker_rule = (
        "Automatic worker sweep: stop at "
        f"{result.max_gpu_utilization:g}% average GPU duty cycle or the worker limit"
    )
    duration_rule = (
        f"Wall-clock budget: {result.max_duration_minutes:g} minutes; "
        f"elapsed {result.elapsed_seconds / 60.0:.1f} minutes")
    legends = {}
    for row in range(1, row_count + 1):
        subplot = figure.get_subplot(row, 1)
        legend_name = "legend" if row == 1 else f"legend{row}"
        legends[legend_name] = {
            "x": 0.995,
            "y": subplot.yaxis.domain[1] - 0.005,
            "xanchor": "right",
            "yanchor": "top",
            "orientation": "v",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#dee2e6",
            "borderwidth": 1,
            "font": {
                "size": 11
            },
        }
    figure.update_layout(
        title={
            "text":
            ("AnySearch resource benchmark"
             f"<br><sup>Recommended: {recommendation.num_env_runners} workers x "
             f"{recommendation.num_envs_per_env_runner} environments</sup>"
             f"<br><sup>{environment_rule}</sup>"
             f"<br><sup>Observed: {result.environment_sweep.stop_reason}</sup>"
             f"<br><sup>{worker_rule}</sup>"
             f"<br><sup>Observed: {result.worker_sweep.stop_reason}</sup>"
             f"<br><sup>{duration_rule}</sup>"),
            "x":
            0.03,
        },
        template="plotly_white",
        height=1680 + (360 if has_prediction else 0),
        hovermode="x unified",
        barmode="group",
        **legends,
        margin={
            "l": 80,
            "r": 80,
            "t": 310,
            "b": 70
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text":
                f"Worker GPU average target: {result.max_gpu_utilization:g}%",
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.08,
                "showarrow": False,
                "xanchor": "right",
            },
        ],
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    _add_diagnostic_links(path)


def _add_prediction_panel(
    figure: "go.Figure",
    result: ResourceBenchmarkResult,
    *,
    row: int,
) -> None:
    """Chart the roofline calibration's guess against what the sweep confirmed.

    Helps a user judge, at a glance, how much to trust the calibration on
    their hardware: bars close together mean the prediction that narrowed
    the sweep was a good guess; a large gap means the confirming sweep did
    most of the real work, which is exactly what it is there for.
    """
    import plotly.graph_objects as go

    prediction = result.prediction
    assert prediction is not None
    legend_name = f"legend{row}"

    phases = ["Environments", "Workers"]
    predicted = [
        prediction.environment_predicted.predicted_steps_per_second,
        prediction.worker_predicted.predicted_steps_per_second,
    ]
    confirmed = [
        result.environment_sweep.peak_steps_per_second,
        result.worker_sweep.peak_steps_per_second,
    ]
    bottlenecks = [
        prediction.environment_predicted.bottleneck,
        prediction.worker_predicted.bottleneck,
    ]
    figure.add_trace(
        go.Bar(
            x=phases,
            y=predicted,
            name="Calibration prediction",
            legend=legend_name,
            marker_color="rgba(230,119,0,0.55)",
            text=[
                f"{value:.1f} steps/s<br>bottleneck: {bottleneck}"
                for value, bottleneck in zip(predicted, bottlenecks)
            ],
            textposition="outside",
        ),
        row=row,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=phases,
            y=confirmed,
            name="Confirmed sweep peak",
            legend=legend_name,
            marker_color="rgba(8,127,91,0.75)",
            text=[f"{value:.1f} steps/s" for value in confirmed],
            textposition="outside",
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(title_text="Sampled steps / second", row=row, col=1)

    costs = prediction.stage_costs
    gil_text = (f"{costs.gil_contention_ratio:.1%}"
                if costs.gil_contention_ratio is not None else "n/a")
    scheduler_text = (f"{costs.scheduler_queue_seconds * 1000:.2f} ms"
                      if costs.scheduler_queue_seconds is not None else "n/a")
    summary = (
        f"Calibration took {prediction.calibration_seconds:.1f}s · "
        f"env step {costs.env_step_seconds * 1000:.2f} ms · "
        f"inference {costs.inference_seconds_per_env * 1000:.2f} ms · "
        f"learner {costs.learner_seconds_per_batch * 1000:.1f} ms/batch · "
        f"transfer {costs.transfer_seconds_per_mb * 1000:.2f} ms/MB · "
        f"GIL contention {gil_text} · "
        f"scheduler queue {scheduler_text} · "
        f"oversubscription correction {prediction.correction_exponent:.2f}")
    figure.add_annotation(
        text=summary,
        xref=f"x{row} domain",
        yref=f"y{row} domain",
        x=0.02,
        y=1.12,
        showarrow=False,
        align="left",
        font={"size": 11, "color": "#495057"},
    )


def _add_diagnostic_links(path: Path) -> None:
    """Add stable links to captured benchmark output without altering plots."""
    document = path.read_text(encoding="utf-8")
    diagnostics = (
        '<aside style="position:fixed;right:12px;bottom:12px;z-index:9999;'
        'padding:9px 12px;border:1px solid #ced4da;border-radius:6px;'
        'background:rgba(255,255,255,.94);font:13px sans-serif;color:#343a40">'
        'Diagnostics: <a href="benchmark.stdout.log">stdout</a> · '
        '<a href="benchmark.stderr.log">stderr</a></aside>'
    )
    document = document.replace("<body>", f"<body>{diagnostics}", 1)
    path.write_text(document, encoding="utf-8")
