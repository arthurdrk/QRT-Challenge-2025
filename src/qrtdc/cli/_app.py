from __future__ import annotations

import sys
from typing import Optional

import typer

from ._console import Console

app = typer.Typer(
    name="qrtdc",
    help="QRT Data Challenge 2025 CLI — utilities and experiments.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def _app(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Increase verbosity (-v, -vv).",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show the application version and exit.",
        callback=lambda value: _print_version_and_exit(value),
        is_eager=True,
    ),
):
    """Top-level options for the CLI."""
    ctx.obj = {"verbose": verbose}


def _print_version_and_exit(value: Optional[bool]) -> None:
    if value:
        try:
            # Use package metadata if available
            import importlib.metadata as importlib_metadata  # Python 3.8+
        except Exception:  # pragma: no cover - fallback for very old Pythons
            import importlib_metadata  # type: ignore
        try:
            version = importlib_metadata.version("qrt-challenge-2025")
        except importlib_metadata.PackageNotFoundError:
            version = "0.1.0"
        Console.print(f"[title]qrtdc[/title] version [success]{version}[/success]")
        raise typer.Exit(code=0)

# Register subcommands from the qrtdc package
try:
    from qrtdc.cli._preprocess import app as preprocess_app

    app.add_typer(preprocess_app, name="preprocess", help="Data preprocessing pipeline.")
except Exception as exc:  # be resilient if importing optional cmds fails
    if "pytest" not in sys.modules:
        Console.print(f"[warning]Some subcommands failed to load: {exc}[/warning]")
