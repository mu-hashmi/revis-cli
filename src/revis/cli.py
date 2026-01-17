"""Revis CLI."""

import logging
import os
import shutil
import subprocess
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

TMUX_SESSION_PREFIX = "revis-"


def get_tmux_session_name(name: str) -> str:
    """Get tmux session name for a revis session."""
    return f"{TMUX_SESSION_PREFIX}{name}"


def tmux_session_exists(session_name: str) -> bool:
    """Check if a tmux session exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    return result.returncode == 0


def setup_logging(verbose: bool = False, session_name: str | None = None):
    """Configure logging with rich handler and optional file output.

    If session_name is provided, also logs to .revis/logs/<session_name>.log
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True)
    ]

    # Add file handler for session logs
    if session_name:
        log_dir = Path(".revis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{session_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture debug in file
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,  # Override any existing config
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

    console.print("[green]Initialized Revis.[/green]")
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
                    console.print(f"  - {s.name} (branch: {s.branch})")
                console.print("\nUse 'revis stop' to clean up.")
            else:
                console.print("No active session.")
            return

        # Session info
        console.print(f"[bold]Session:[/bold] {session.name}")
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
    name: str = typer.Option(..., "--name", "-n", help="Session name (used for branch name)"),
    budget: str = typer.Option(..., help="Budget: time (e.g., 8h) or run count (e.g., 10)"),
    budget_type: str = typer.Option("time", "--type", "-t", help="Budget type: time or runs"),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Baseline run ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    background: bool = typer.Option(False, "--background", "--bg", help="Run in background (tmux)"),
    _in_tmux: bool = typer.Option(False, "--_in-tmux", hidden=True, help="Internal: already in tmux"),
):
    """Start the autonomous iteration loop."""
    # Handle background mode - launch in tmux and exit
    if background and not _in_tmux:
        if not shutil.which("tmux"):
            console.print("[red]Error:[/red] tmux not installed. Install it or run without --background.")
            raise typer.Exit(1)

        tmux_name = get_tmux_session_name(name)
        if tmux_session_exists(tmux_name):
            console.print(f"[red]Error:[/red] tmux session '{tmux_name}' already exists.")
            console.print(f"Use 'revis watch {name}' to attach or 'tmux kill-session -t {tmux_name}' to remove it.")
            raise typer.Exit(1)

        # Build command to re-run ourselves in tmux
        cmd_parts = ["revis", "loop", "--name", name, "--budget", budget, "--type", budget_type, "--_in-tmux"]
        if baseline:
            cmd_parts.extend(["--baseline", baseline])
        if verbose:
            cmd_parts.append("--verbose")

        # Escape for shell
        cmd = " ".join(cmd_parts)
        cwd = os.getcwd()

        # Launch in tmux
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", tmux_name, "-c", cwd, cmd],
            check=True,
        )

        console.print("[green]Revis loop started in background[/green]")
        console.print(f"  Session: {name}")
        console.print(f"  Budget: {budget} ({budget_type})")
        console.print()
        console.print("[bold]Commands:[/bold]")
        console.print(f"  revis watch {name}    - attach to live output")
        console.print(f"  revis logs {name}     - show recent output")
        console.print("  revis status         - check session status")
        console.print("  revis stop           - stop the loop")
        return

    setup_logging(verbose, session_name=name)
    store = get_store()

    # Validate name
    if not name.replace("-", "").replace("_", "").isalnum():
        console.print("[red]Error:[/red] Session name must be alphanumeric (dashes/underscores allowed)")
        raise typer.Exit(1)

    # Check for name conflict
    if store.session_name_exists(name):
        console.print(f"[red]Error:[/red] Session '{name}' already exists.")
        console.print("Use 'revis list' to see existing sessions or choose a different name.")
        raise typer.Exit(1)

    # Check for existing running session
    existing = store.get_running_session()
    if existing:
        console.print(f"[red]Error:[/red] Session '{existing.name}' is already running.")
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

    console.print(f"[green]Starting Revis session '{name}'[/green]")
    console.print(f"  Branch: revis/{name}")
    console.print(f"  Budget: {budget} ({budget_type})")
    console.print(f"  Model: {config.llm.model}")
    console.print(f"  Primary metric: {config.metrics.primary}")

    from revis.loop import run_loop

    try:
        session = run_loop(config, name, budget_obj, console, baseline)
        console.print(f"\n[green]Session completed:[/green] {session.termination_reason}")
        console.print(f"\nTo export: revis export {name}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Use 'revis stop' to gracefully stop.")
        raise typer.Exit(1)


@app.command()
def resume(
    name: str = typer.Argument(..., help="Session name to resume"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Resume a stopped session."""
    setup_logging(verbose)
    store = get_store()

    session = store.get_session_by_name(name)
    if session is None:
        console.print(f"[red]Error:[/red] Session '{name}' not found.")
        raise typer.Exit(1)

    if session.status == "running":
        console.print(f"[red]Error:[/red] Session '{name}' is already running.")
        raise typer.Exit(1)

    if session.status == "completed":
        console.print(f"[red]Error:[/red] Session '{name}' is already completed.")
        raise typer.Exit(1)

    console.print(f"[green]Resuming session '{name}'[/green]")
    console.print(f"  Remaining budget: {session.budget.remaining()}")
    console.print(f"  Iteration: {session.iteration_count}")

    config_path = Path(CONFIG_FILE)
    config = load_config(config_path)

    from revis.loop import resume_loop

    session = resume_loop(config, session, console)
    console.print(f"\n[green]Session completed:[/green] {session.termination_reason}")
    console.print(f"\nTo export: revis export {name}")


@app.command()
def stop():
    """Stop the current session gracefully."""
    store = get_store()

    session = store.get_running_session()
    if session is None:
        console.print("No active session to stop.")
        raise typer.Exit(0)

    console.print(f"[yellow]Stopping session '{session.name}'[/yellow]")

    # Create stop signal file
    stop_file = Path(REVIS_DIR) / "stop_signal"
    stop_file.touch()

    console.print("Stop signal sent. The session will stop after the current run completes.")
    console.print("Use 'revis status' to monitor.")


@app.command()
def watch(
    name: str = typer.Argument(..., help="Session name to watch"),
):
    """Attach to a running loop's tmux session."""
    if not shutil.which("tmux"):
        console.print("[red]Error:[/red] tmux not installed.")
        raise typer.Exit(1)

    tmux_name = get_tmux_session_name(name)
    if not tmux_session_exists(tmux_name):
        console.print(f"[red]Error:[/red] No tmux session '{tmux_name}' found.")
        console.print("The loop may not be running in background mode, or has already finished.")
        console.print("\nCheck 'revis status' or 'revis list' for session info.")
        raise typer.Exit(1)

    # Attach to tmux session (replaces current process)
    os.execvp("tmux", ["tmux", "attach-session", "-t", tmux_name])


@app.command()
def logs(
    name: str = typer.Argument(..., help="Session name to show logs for"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow output (like tail -f)"),
):
    """Show recent output from a session (from log file or tmux)."""
    log_file = Path(f".revis/logs/{name}.log")
    tmux_name = get_tmux_session_name(name)
    tmux_running = shutil.which("tmux") and tmux_session_exists(tmux_name)

    # Prefer log file (persists after session ends)
    if log_file.exists():
        if follow:
            # Use tail -f on the log file
            try:
                os.execvp("tail", ["tail", "-f", "-n", str(lines), str(log_file)])
            except KeyboardInterrupt:
                pass
        else:
            # Show last N lines
            content = log_file.read_text()
            log_lines = content.strip().split("\n")
            for line in log_lines[-lines:]:
                console.print(line)
            if tmux_running:
                console.print(f"\n[dim]Use 'revis watch {name}' to attach live, or 'revis logs {name} -f' to follow[/dim]")
            else:
                console.print(f"\n[dim]Session finished. Showing last {min(lines, len(log_lines))} lines from log file.[/dim]")
        return

    # Fall back to tmux if log file doesn't exist
    if not shutil.which("tmux"):
        console.print(f"[red]Error:[/red] No log file found at {log_file}")
        raise typer.Exit(1)

    if not tmux_running:
        console.print(f"[red]Error:[/red] No log file '{log_file}' and no tmux session '{tmux_name}' found.")
        console.print("The session may not have started, or logs were not captured.")
        raise typer.Exit(1)

    if follow:
        try:
            while True:
                os.system("clear")
                result = subprocess.run(
                    ["tmux", "capture-pane", "-t", tmux_name, "-p", "-S", f"-{lines}"],
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)
                print(f"\n[Following {tmux_name} - Ctrl+C to stop]")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", tmux_name, "-p", "-S", f"-{lines}"],
            capture_output=True,
            text=True,
        )
        console.print(result.stdout)
        console.print(f"\n[dim]Use 'revis watch {name}' to attach, or 'revis logs {name} -f' to follow[/dim]")


@app.command("list")
def list_sessions(
    all_sessions: bool = typer.Option(False, "--all", "-a", help="Show all sessions including completed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show additional details"),
):
    """List all Revis sessions."""
    store = get_store()

    sessions = store.list_sessions(limit=100)

    if not sessions:
        console.print("No sessions found.")
        return

    table = Table(title="Revis Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Iterations", justify="right")
    table.add_column("Budget")
    table.add_column("Exported", justify="center")

    if verbose:
        table.add_column("Branch")
        table.add_column("Started")

    for session in sessions:
        status_style = {
            "running": "green",
            "completed": "blue",
            "stopped": "yellow",
            "failed": "red",
        }.get(session.status, "white")

        # Budget display
        if session.budget.type == "time":
            budget_str = f"{format_duration(session.budget.used)}/{format_duration(session.budget.value)}"
        else:
            budget_str = f"{session.budget.used}/{session.budget.value} runs"

        exported = "[green]\u2713[/green]" if session.exported_at else "[dim]-[/dim]"

        row = [
            session.name,
            f"[{status_style}]{session.status}[/{status_style}]",
            str(session.iteration_count),
            budget_str,
            exported,
        ]

        if verbose:
            row.extend([
                session.branch,
                session.started_at.strftime("%Y-%m-%d %H:%M"),
            ])

        table.add_row(*row)

    console.print(table)


@app.command()
def show(
    name: str = typer.Argument(..., help="Session name to show"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show agent tool call traces"),
):
    """Show detailed information about a session."""
    store = get_store()

    session = store.get_session_by_name(name)
    if session is None:
        console.print(f"[red]Error:[/red] Session '{name}' not found.")
        raise typer.Exit(1)

    # Session header
    console.print(f"\n[bold]Session: {session.name}[/bold]")
    console.print(f"  ID: {session.id}")
    console.print(f"  Branch: {session.branch}")
    console.print(f"  Status: {session.status}")
    if session.termination_reason:
        console.print(f"  Termination: {session.termination_reason.value}")

    # Times
    console.print("\n[bold]Timeline:[/bold]")
    console.print(f"  Started: {session.started_at}")
    if session.ended_at:
        console.print(f"  Ended: {session.ended_at}")
        duration = session.ended_at - session.started_at
        console.print(f"  Duration: {format_duration(int(duration.total_seconds()))}")
    if session.exported_at:
        console.print(f"  Exported: {session.exported_at}")
        if session.pr_url:
            console.print(f"  PR: {session.pr_url}")

    # Budget
    console.print("\n[bold]Budget:[/bold]")
    if session.budget.type == "time":
        console.print(f"  Time: {format_duration(session.budget.used)} / {format_duration(session.budget.value)}")
    else:
        console.print(f"  Runs: {session.budget.used} / {session.budget.value}")
    console.print(f"  LLM Cost: ${session.llm_cost_usd:.2f}")
    console.print(f"  Retries Remaining: {session.retry_budget}")

    runs = store.query_runs(session_id=session.id, limit=50)
    if not runs:
        return

    if trace:
        _show_traces(store, runs)
    else:
        _show_runs_table(store, runs, session.iteration_count)


def _show_runs_table(store, runs: list, iteration_count: int) -> None:
    """Show runs as a summary table."""
    console.print(f"\n[bold]Iterations ({iteration_count}):[/bold]")

    table = Table()
    table.add_column("#", justify="right")
    table.add_column("Status")
    table.add_column("Metrics")
    table.add_column("Change")
    table.add_column("Duration")

    for run in reversed(runs):
        metrics = store.get_run_metrics(run.id)
        metrics_str = ", ".join(f"{m.name}={m.value:.4f}" for m in metrics[:3])
        if not metrics_str:
            metrics_str = "N/A"

        decisions = store.get_decisions(run.id)
        if decisions:
            rationale = decisions[0].rationale
            change_str = rationale[:40] + "..." if len(rationale) > 40 else rationale
        else:
            change_str = "Initial"

        duration_str = ""
        if run.started_at and run.ended_at:
            secs = (run.ended_at - run.started_at).total_seconds()
            duration_str = format_duration(int(secs))
        elif run.started_at:
            duration_str = "running..."

        status_style = {"completed": "green", "failed": "red", "running": "yellow"}.get(run.status, "white")

        table.add_row(
            str(run.iteration_number),
            f"[{status_style}]{run.status}[/{status_style}]",
            metrics_str,
            change_str,
            duration_str,
        )

    console.print(table)


def _show_traces(store, runs: list) -> None:
    """Show agent tool call traces for each run."""
    for run in reversed(runs):
        console.print(f"\n[bold]─── Iteration {run.iteration_number} ───[/bold]")

        traces = store.get_traces(run.id)
        if not traces:
            console.print("  [dim]No traces recorded[/dim]")
            continue

        for trace in traces:
            event_type = trace["event_type"]
            data = trace["data"]

            if event_type == "tool_call":
                tool = data.get("tool", "?")
                args = data.get("args", {})
                console.print(f"[dim]→[/dim] [cyan]{tool}[/cyan]", end="")
                _print_trace_args(tool, args)
            # Skip tool_result - they're interleaved and make output noisy

        decisions = store.get_decisions(run.id)
        if decisions:
            rationale = decisions[0].rationale
            console.print(f"[green]Rationale:[/green] {rationale}")


def _print_trace_args(tool: str, args: dict) -> None:
    """Format tool arguments for trace display."""
    if tool == "read_file":
        console.print(f"([yellow]{args.get('path', '?')}[/yellow])")
    elif tool == "write_file":
        path = args.get("path", "?")
        content = args.get("content", "")
        console.print(f"([yellow]{path}[/yellow]) [dim]{len(content)} bytes[/dim]")
    elif tool == "search_codebase":
        console.print(f"([yellow]{args.get('pattern', '?')}[/yellow])")
    elif tool == "find_definition":
        console.print(f"([yellow]{args.get('name', '?')}[/yellow])")
    elif tool == "list_directory":
        path = args.get("path", ".")
        recursive = args.get("recursive", False)
        suffix = " [dim]recursive[/dim]" if recursive else ""
        console.print(f"([yellow]{path}[/yellow]){suffix}")
    elif tool == "run_command":
        cmd = args.get("command", "?")
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."
        console.print(f"([yellow]{cmd}[/yellow])")
    else:
        console.print(f"({args})")


@app.command()
def export(
    name: str = typer.Argument(..., help="Session name to export"),
    no_pr: bool = typer.Option(False, "--no-pr", help="Push branch but don't create PR"),
    force: bool = typer.Option(False, "--force", "-f", help="Force push (use with caution)"),
):
    """Export a session to remote and optionally create a PR."""
    store = get_store()

    session = store.get_session_by_name(name)
    if session is None:
        console.print(f"[red]Error:[/red] Session '{name}' not found.")
        raise typer.Exit(1)

    if session.status == "running":
        console.print("[red]Error:[/red] Cannot export running session. Stop it first with 'revis stop'.")
        raise typer.Exit(1)

    if session.exported_at and not force:
        console.print(f"[yellow]Warning:[/yellow] Session already exported at {session.exported_at}")
        if session.pr_url:
            console.print(f"  PR: {session.pr_url}")
        if not typer.confirm("Export again?"):
            raise typer.Exit(0)

    # Load config for PR body
    config_path = Path(CONFIG_FILE)
    config = load_config(config_path)

    from revis.analyzer.compare import RunAnalyzer
    from revis.github.pr import GitConfig, GitHubManager, GitManager, format_pr_body

    repo_path = Path.cwd()
    git = GitManager(GitConfig(repo_path=repo_path))

    # Ensure we're on the session branch for push
    current_branch = git.get_current_branch()
    if current_branch != session.branch:
        console.print(f"Checking out branch {session.branch}...")
        git.checkout(session.branch)

    # Push to remote
    console.print(f"Pushing branch {session.branch} to origin...")
    try:
        git.push(session.branch, force=force)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to push: {e}")
        if current_branch != session.branch:
            git.checkout(current_branch)
        raise typer.Exit(1)

    pr_url = None

    if not no_pr:
        try:
            github = GitHubManager()
        except ValueError:
            console.print("[yellow]Warning:[/yellow] GITHUB_TOKEN not set - skipping PR creation.")
            console.print("Set GITHUB_TOKEN environment variable to enable PR creation.")
        else:
            console.print("Creating pull request...")

            runs = store.query_runs(session_id=session.id, limit=100)
            runs = list(reversed(runs))

            run_metrics = {}
            decisions = {}
            for run in runs:
                metrics = store.get_run_metrics(run.id)
                run_metrics[run.id] = {m.name: m.value for m in metrics}
                run_decisions = store.get_decisions(run.id)
                if run_decisions:
                    decisions[run.id] = run_decisions[0].rationale

            analyzer = RunAnalyzer(
                store=store,
                primary_metric=config.metrics.primary,
                minimize=config.metrics.minimize,
            )
            metric_history = analyzer.get_metric_history(session.id)
            baseline_value = metric_history[0] if metric_history else None

            owner, repo = git.get_repo_info()

            title = f"Revis/{session.name}"
            if session.termination_reason:
                title = f"Revis/{session.name}: {session.termination_reason.value.replace('_', ' ').title()}"

            body = format_pr_body(
                session=session,
                runs=runs,
                run_metrics=run_metrics,
                decisions=decisions,
                primary_metric=config.metrics.primary,
                baseline_value=baseline_value,
            )

            try:
                pr_url = github.create_pr(
                    owner=owner,
                    repo=repo,
                    title=title,
                    body=body,
                    head=session.branch,
                    base="main",
                )
                console.print(f"[green]Created PR:[/green] {pr_url}")
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to create PR: {e}")

    # Mark as exported
    store.mark_session_exported(session.id, pr_url)

    # Return to original branch
    if current_branch != session.branch:
        git.checkout(current_branch)

    console.print(f"[green]Session '{name}' exported successfully.[/green]")
    if pr_url:
        console.print(f"  PR: {pr_url}")


@app.command()
def delete(
    names: list[str] = typer.Argument(..., help="Session name(s) to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    keep_branch: bool = typer.Option(False, "--keep-branch", help="Don't delete local git branch"),
):
    """Delete local session(s) and their branches."""
    store = get_store()

    # Validate all sessions exist first
    sessions_to_delete = []
    for name in names:
        session = store.get_session_by_name(name)
        if session is None:
            console.print(f"[red]Error:[/red] Session '{name}' not found.")
            raise typer.Exit(1)
        if session.status == "running" and not force:
            console.print(f"[red]Error:[/red] Cannot delete running session '{name}'. Stop it first or use --force.")
            raise typer.Exit(1)
        sessions_to_delete.append(session)

    # Confirm
    if not force:
        console.print("Will delete the following sessions:")
        for session in sessions_to_delete:
            exported = " (exported)" if session.exported_at else ""
            console.print(f"  - {session.name}: {session.iteration_count} iterations{exported}")
        if not typer.confirm("Continue?"):
            raise typer.Exit(0)

    from revis.github.pr import GitConfig, GitManager

    repo_path = Path.cwd()
    git = GitManager(GitConfig(repo_path=repo_path))
    current_branch = git.get_current_branch()

    for session in sessions_to_delete:
        # Delete git branch if requested
        if not keep_branch:
            if current_branch == session.branch:
                console.print(f"[yellow]Warning:[/yellow] Cannot delete branch '{session.branch}' - currently checked out.")
            elif git.branch_exists(session.branch):
                try:
                    git._run("branch", "-D", session.branch)
                    console.print(f"  Deleted branch: {session.branch}")
                except Exception as e:
                    console.print(f"  [yellow]Warning:[/yellow] Could not delete branch: {e}")

        # Delete from database
        store.delete_session(session.id, force=force)
        console.print(f"[green]Deleted session:[/green] {session.name}")


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
