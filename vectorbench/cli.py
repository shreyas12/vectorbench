"""Typer CLI: `vectorbench run <config.yaml> [--debug]` and `--version`."""

from __future__ import annotations

from pathlib import Path

import typer

from . import __version__
from .config import ConfigError, load_config
from .report import resolved_config_yaml, write_outputs
from .runner import run_experiment

app = typer.Typer(add_completion=False, help="Design, run, compare, and visualize "
                  "retrieval experiments on your own data.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vectorbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """VectorBench CLI."""


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to an experiment YAML config."),
    debug: bool = typer.Option(False, "--debug", help="Print full tracebacks on error."),
) -> None:
    """Run a retrieval experiment and write a self-contained Experiment Report."""
    short_hash = "????????"
    try:
        config = load_config(config_path)
        typer.echo(f"Experiment: {config.name}  [flat-vs-hnsw]")
        typer.echo(f"Dataset: {config.dataset.kind}")

        result = run_experiment(config, log=typer.echo)
        short_hash = result.short_hash
        config_yaml = resolved_config_yaml(config)
        report_path = write_outputs(result, config.output.dir, config_yaml)
        typer.echo(f"Experiment report: {report_path}")
    except (ConfigError, ValueError, RuntimeError, FileNotFoundError) as exc:
        if debug:
            raise
        typer.secho(
            f"Error during experiment {short_hash}: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
