"""Render the self-contained Experiment Report (HTML) and write the registry-ready run folder."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import plotly.graph_objects as go
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__
from .runner import ExperimentResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ACCENT = "#2563eb"
_FLAT_COLOR = "#dc2626"


def _build_pareto_figure(result: ExperimentResult) -> go.Figure:
    """p50 latency (x, log) vs Recall@k (y); HNSW error bars + Flat reference line."""
    xs = [p.latency.p50_ms for p in result.sweep]
    ys = [p.recall_mean for p in result.sweep]
    err_y = [p.recall_std for p in result.sweep]
    err_x = [p.latency.iqr_ms / 2 for p in result.sweep]
    labels = [f"ef={p.ef_search}" for p in result.sweep]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+lines+text",
            name="HNSW",
            text=labels,
            textposition="bottom center",
            marker=dict(size=10, color=_ACCENT),
            line=dict(color=_ACCENT, width=1, dash="dot"),
            error_y=dict(type="data", array=err_y, visible=True, color=_ACCENT),
            error_x=dict(type="data", array=err_x, visible=True, color=_ACCENT),
            hovertemplate="%{text}<br>p50 %{x:.3f} ms<br>recall %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=result.flat.recall,
        line=dict(color=_FLAT_COLOR, width=2, dash="dash"),
        annotation_text=f"Flat (exact) — recall {result.flat.recall:.2f}, "
        f"p50 {result.flat.latency.p50_ms:.2f} ms",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="p50 latency (ms, log scale)", type="log"),
        yaxis=dict(title=f"Recall@{result.k} (vs exact search)"),
        margin=dict(l=60, r=30, t=30, b=60),
        height=460,
        showlegend=False,
    )
    return fig


def _summary_rows(result: ExperimentResult) -> list[dict]:
    rows = [
        {
            "label": "Flat (exact)",
            "recall": f"{result.flat.recall:.3f}",
            "p50": f"{result.flat.latency.p50_ms:.3f}",
            "p95": f"{result.flat.latency.p95_ms:.3f}",
            "iqr": f"{result.flat.latency.iqr_ms:.3f}",
            "build": f"{result.flat.build_time_s:.3f}",
            "size": f"{result.flat.size_mb:.2f}",
            "is_flat": True,
        }
    ]
    for p in result.sweep:
        rows.append(
            {
                "label": f"HNSW ef={p.ef_search}",
                "recall": f"{p.recall_mean:.3f} ± {p.recall_std:.3f}",
                "p50": f"{p.latency.p50_ms:.3f}",
                "p95": f"{p.latency.p95_ms:.3f}",
                "iqr": f"{p.latency.iqr_ms:.3f}",
                "build": f"{p.build_time_s_mean:.3f}",
                "size": f"{p.size_mb:.2f}",
                "is_flat": False,
            }
        )
    return rows


def render_report(result: ExperimentResult, config_yaml: str) -> str:
    """Render the full self-contained HTML report string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("experiment_report.html.j2")
    fig = _build_pareto_figure(result)
    chart_html = fig.to_html(include_plotlyjs="inline", full_html=False)

    return template.render(
        result=result,
        config_yaml=config_yaml,
        chart_html=chart_html,
        rows=_summary_rows(result),
        version=__version__,
        reproduce_cmd="vectorbench run config.yaml",
        k=result.k,
    )


def _results_json(result: ExperimentResult) -> dict:
    return result.to_json_dict()


def write_outputs(result: ExperimentResult, out_dir: Path, config_yaml: str) -> Path:
    """Write the registry-ready run folder; return the path to the Experiment Report."""
    ts = result.timestamp.replace(":", "").replace("-", "").split(".")[0]
    run_dir = Path(out_dir) / f"{ts}_{result.short_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolved config as actually run.
    (run_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")

    # results.json — raw per-rep numbers, schema-versioned.
    (run_dir / "results.json").write_text(
        json.dumps(_results_json(result), indent=2), encoding="utf-8"
    )

    # metadata.json — flat index source for a future `vectorbench list`.
    metadata = {
        "experiment_type": result.experiment_type,
        "name": result.name,
        "full_hash": result.full_hash,
        "short_hash": result.short_hash,
        "timestamp": result.timestamp,
        "duration_s": result.duration_s,
        "dataset": result.dataset_summary,
        "machine_info": result.machine_info,
        "schema_version": result.schema_version,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # The Experiment Report.
    report_path = run_dir / "experiment_report.html"
    report_path.write_text(render_report(result, config_yaml), encoding="utf-8")
    return report_path


def resolved_config_yaml(config) -> str:
    """Serialize the validated config to YAML for the run folder and report config block."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
