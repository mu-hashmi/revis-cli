"""Revis CLI."""

import logging
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from revis.config import get_config_template, load_config, parse_duration
from revis.store.sqlite import SQLiteRunStore
from revis.types import Budget

app = typer.Typer(help="Revis - Autonomous ML iteration engine")
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

REVIS_DIR = ".revis"
CONFIG_FILE = "revis.yaml"
DB_FILE = f"{REVIS_DIR}/revis.db"


def get_store() -> SQLiteRunStore:
    """Get the run store, initializing if needed."""
    db_path = Path(DB_FILE)
    if not db_path.exists():
        console.print("[red]Error:[/red] Revis not initialized. Run 'revis init' first.")
        raise typer.Exit(1)
    return SQLiteRunStore(db_path)


@app.command()
def init():
    """Initialize Revis in the current directory."""
    revis_dir = Path(REVIS_DIR)
    config_file = Path(CONFIG_FILE)

    if config_file.exists():
        console.print(f"[yellow]Warning:[/yellow] {CONFIG_FILE} already exists.")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    # Create .revis directory structure
    revis_dir.mkdir(exist_ok=True)
    (revis_dir / "artifacts").mkdir(exist_ok=True)
    (revis_dir / "logs").mkdir(exist_ok=True)

    # Initialize database
    db_path = Path(DB_FILE)
    store = SQLiteRunStore(db_path)
    store.initialize()

    # Write config template
    config_file.write_text(get_config_template())

    # Add .revis to .gitignore if it exists
    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text()
        if REVIS_DIR not in content:
            with open(gitignore, "a") as f:
                f.write(f"\n# Revis\n{REVIS_DIR}/\n")
            console.print(f"Added {REVIS_DIR}/ to .gitignore")
    else:
        gitignore.write_text(f"# Revis\n{REVIS_DIR}/\n")
        console.print(f"Created .gitignore with {REVIS_DIR}/")

    console.print(f"[green]Initialized Revis.[/green]")
    console.print(f"  Config: {CONFIG_FILE}")
    console.print(f"  Database: {DB_FILE}")
    console.print(f"\nEdit {CONFIG_FILE} to configure your training setup.")


@app.command()
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 5 seconds"),
):
    """Show current session status."""
    store = get_store()

    def show_status():
        session = store.get_running_session()

        if session is None:
            # Check for orphaned sessions
            orphaned = store.get_orphaned_sessions()
            if orphaned:
                console.print("[yellow]Warning:[/yellow] Found orphaned session(s):")
                for s in orphaned:
                    console.print(f"  - {s.id} (branch: {s.branch})")
                console.print("\nUse 'revis stop' to clean up.")
            else:
                console.print("No active session.")
            return

        # Session info
        console.print(f"[bold]Session:[/bold] {session.id}")
        console.print(f"[bold]Branch:[/bold] {session.branch}")
        console.print(f"[bold]Status:[/bold] {session.status}")
        console.print(f"[bold]Iteration:[/bold] {session.iteration_count}")

        # Budget
        budget = session.budget
        if budget.type == "time":
            used_str = format_duration(budget.used)
            total_str = format_duration(budget.value)
            pct = (budget.used / budget.value * 100) if budget.value > 0 else 0
            console.print(f"[bold]Budget:[/bold] {used_str} / {total_str} ({pct:.0f}%)")
        else:
            console.print(f"[bold]Budget:[/bold] {budget.used} / {budget.value} runs")

        console.print(f"[bold]Retries left:[/bold] {session.retry_budget}")
        console.print(f"[bold]LLM cost:[/bold] ${session.llm_cost_usd:.2f}")

        # Recent runs
        runs = store.query_runs(session_id=session.id, limit=5)
        if runs:
            console.print("\n[bold]Recent runs:[/bold]")
            table = Table()
            table.add_column("#")
            table.add_column("Status")
            table.add_column("Duration")

            for run in reversed(runs):
                duration = ""
                if run.started_at and run.ended_at:
                    secs = (run.ended_at - run.started_at).total_seconds()
                    duration = format_duration(int(secs))
                elif run.started_at:
                    duration = "running..."

                table.add_row(
                    str(run.iteration_number),
                    run.status,
                    duration,
                )
            console.print(table)

    if watch:
        try:
            while True:
                console.clear()
                show_status()
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        show_status()


@app.command()
def loop(
    budget: str = typer.Option(..., help="Budget: time (e.g., 8h) or run count (e.g., 10)"),
    budget_type: str = typer.Option("time", "--type", "-t", help="Budget type: time or runs"),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Baseline run ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Start the autonomous iteration loop."""
    setup_logging(verbose)
    store = get_store()

    # Check for existing session
    existing = store.get_running_session()
    if existing:
        console.print(f"[red]Error:[/red] Session {existing.id} is already running.")
        console.print("Use 'revis status' to check progress or 'revis stop' to stop it.")
        raise typer.Exit(1)

    # Load config
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        console.print(f"[red]Error:[/red] {CONFIG_FILE} not found. Run 'revis init' first.")
        raise typer.Exit(1)

    config = load_config(config_path)

    # Parse budget
    if budget_type == "time":
        budget_seconds = parse_duration(budget)
        budget_obj = Budget(type="time", value=budget_seconds)
    else:
        budget_obj = Budget(type="runs", value=int(budget))

    console.print(f"[green]Starting Revis loop[/green]")
    console.print(f"  Budget: {budget} ({budget_type})")
    console.print(f"  Model: {config.llm.model}")
    console.print(f"  Primary metric: {config.metrics.primary}")

    # Import and run the main loop
    from revis.loop import run_loop

    try:
        session = run_loop(config, budget_obj, baseline)
        console.print(f"\n[green]Session completed:[/green] {session.termination_reason}")
        if session.pr_url:
            console.print(f"PR: {session.pr_url}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Use 'revis stop' to gracefully stop.")
        raise typer.Exit(1)


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Resume a stopped session."""
    setup_logging(verbose)
    store = get_store()

    session = store.get_session(session_id)
    if session is None:
        console.print(f"[red]Error:[/red] Session {session_id} not found.")
        raise typer.Exit(1)

    if session.status == "running":
        console.print(f"[red]Error:[/red] Session {session_id} is already running.")
        raise typer.Exit(1)

    if session.status == "completed":
        console.print(f"[red]Error:[/red] Session {session_id} is already completed.")
        raise typer.Exit(1)

    console.print(f"[green]Resuming session {session_id}[/green]")
    console.print(f"  Remaining budget: {session.budget.remaining()}")
    console.print(f"  Iteration: {session.iteration_count}")

    # Load config and resume
    config_path = Path(CONFIG_FILE)
    config = load_config(config_path)

    from revis.loop import resume_loop

    session = resume_loop(config, session)
    console.print(f"\n[green]Session completed:[/green] {session.termination_reason}")
    if session.pr_url:
        console.print(f"PR: {session.pr_url}")


@app.command()
def stop():
    """Stop the current session gracefully."""
    store = get_store()

    session = store.get_running_session()
    if session is None:
        console.print("No active session to stop.")
        raise typer.Exit(0)

    console.print(f"[yellow]Stopping session {session.id}[/yellow]")

    # Create stop signal file
    stop_file = Path(REVIS_DIR) / "stop_signal"
    stop_file.touch()

    console.print("Stop signal sent. The session will stop after the current run completes.")
    console.print("Use 'revis status' to monitor.")


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


if __name__ == "__main__":
    app()
