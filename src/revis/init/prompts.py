"""Interactive prompts for revis init using InquirerPy."""

import os
import shutil
from dataclasses import dataclass, field

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console

from revis.init.metrics.eval_json import EvalJsonMetricsSource
from revis.init.metrics.wandb import WandbMetricsSource
from revis.init.ssh_config import SSHHost, parse_ssh_config

console = Console()


@dataclass
class InitConfig:
    """Configuration gathered from interactive prompts."""

    train_command: str = "python train.py"

    metrics_source: str = "eval_json"
    metrics_project: str | None = None
    metrics_entity: str | None = None

    primary_metric: str = "loss"
    minimize: bool = True

    executor_type: str = "local"
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_port: int = 22
    ssh_key_path: str | None = None

    coding_agent_type: str = "auto"

    extra_deny_patterns: list[str] = field(default_factory=list)


def detect_coding_agent() -> str | None:
    """Detect available coding agents."""
    if shutil.which("claude"):
        return "claude-code"
    return None


def prompt_training_command() -> str:
    """Prompt for training command."""
    return inquirer.text(
        message="Training command:",
        default="python train.py",
        validate=lambda x: len(x.strip()) > 0,
        invalid_message="Command cannot be empty",
    ).execute()


def prompt_metrics_source() -> tuple[str, WandbMetricsSource | EvalJsonMetricsSource]:
    """Prompt for metrics source selection."""
    choices = []

    wandb_available = WandbMetricsSource.detect_auth()
    if wandb_available:
        choices.append(Choice(value="wandb", name="Weights & Biases"))
    else:
        choices.append(
            Choice(value="wandb_disabled", name="Weights & Biases (requires: wandb login)")
        )

    choices.append(Choice(value="eval_json", name="eval.json (manual)"))

    source_type = inquirer.select(
        message="Metrics source:",
        choices=choices,
        default="wandb" if wandb_available else "eval_json",
    ).execute()

    if source_type == "wandb_disabled":
        console.print("[yellow]Run 'wandb login' first, then re-run 'revis init'[/yellow]")
        console.print("[dim]See: https://github.com/mu-hashmi/revis-cli#evaljson-manual[/dim]")
        return "eval_json", EvalJsonMetricsSource()

    if source_type == "wandb":
        source = WandbMetricsSource()
        console.print("[dim]Connecting to W&B...[/dim]", end=" ")
        if source.connect():
            console.print("[green]connected[/green]")
            return "wandb", source
        else:
            console.print("[red]failed[/red]")
            console.print("[yellow]Falling back to eval.json[/yellow]")
            console.print("[dim]See: https://github.com/mu-hashmi/revis-cli#evaljson-manual[/dim]")
            return "eval_json", EvalJsonMetricsSource()

    # eval_json selected directly
    console.print("[dim]See: https://github.com/mu-hashmi/revis-cli#evaljson-manual[/dim]")
    return "eval_json", EvalJsonMetricsSource()


def prompt_wandb_project(source: WandbMetricsSource) -> str | None:
    """Prompt for W&B project selection."""
    projects = source.list_projects()

    if not projects:
        console.print("[yellow]No projects found. Enter project name manually.[/yellow]")
        return inquirer.text(
            message="W&B project name:",
            validate=lambda x: len(x.strip()) > 0,
        ).execute()

    choices = [Choice(value=p, name=p) for p in projects[:20]]
    choices.append(Choice(value="__other__", name="Other (enter manually)"))

    selected = inquirer.select(
        message="W&B project:",
        choices=choices,
    ).execute()

    if selected == "__other__":
        return inquirer.text(
            message="Project name:",
            validate=lambda x: len(x.strip()) > 0,
        ).execute()

    return selected


def prompt_primary_metric(
    source: WandbMetricsSource | EvalJsonMetricsSource,
    project: str | None,
) -> str:
    """Prompt for primary metric selection."""
    with console.status("[dim]Fetching metrics...[/dim]"):
        metrics = source.list_metrics(project or "")

    if metrics:
        choices = [Choice(value=m, name=m) for m in metrics[:15]]
        choices.append(Choice(value="__other__", name="Other (enter manually)"))

        selected = inquirer.select(
            message="Primary metric to optimize:",
            choices=choices,
        ).execute()

        if selected == "__other__":
            return inquirer.text(
                message="Metric name:",
                default="loss",
                validate=lambda x: len(x.strip()) > 0,
            ).execute()

        return selected
    else:
        return inquirer.text(
            message="Primary metric to optimize:",
            default="loss",
            validate=lambda x: len(x.strip()) > 0,
        ).execute()


def prompt_objective() -> bool:
    """Prompt for optimization objective (minimize/maximize)."""
    return inquirer.select(
        message="Objective:",
        choices=[
            Choice(value=True, name="minimize"),
            Choice(value=False, name="maximize"),
        ],
        default=True,
    ).execute()


def prompt_executor() -> tuple[str, SSHHost | None]:
    """Prompt for execution environment."""
    ssh_hosts = parse_ssh_config()

    choices = [Choice(value="local", name="Local (tmux)")]

    if ssh_hosts:
        for host in ssh_hosts[:10]:
            choices.append(Choice(value=host, name=f"{host.name} (from ~/.ssh/config)"))

    choices.append(Choice(value="ssh_manual", name="Remote SSH (enter manually)"))

    selected = inquirer.select(
        message="Execution environment:",
        choices=choices,
        default="local",
    ).execute()

    if selected == "local":
        return "local", None

    if isinstance(selected, SSHHost):
        return "ssh", selected

    host = inquirer.text(
        message="SSH hostname:",
        validate=lambda x: len(x.strip()) > 0,
    ).execute()

    user = inquirer.text(
        message="SSH username:",
        validate=lambda x: len(x.strip()) > 0,
    ).execute()

    port_str = inquirer.text(
        message="SSH port:",
        default="22",
    ).execute()

    try:
        port = int(port_str)
    except ValueError:
        port = 22

    return "ssh", SSHHost(name=host, hostname=host, user=user, port=port)


def prompt_coding_agent() -> str:
    """Prompt for coding agent selection."""
    detected = detect_coding_agent()

    choices = []
    if detected:
        choices.append(Choice(value="auto", name=f"auto ({detected} detected)"))
    else:
        choices.append(Choice(value="auto", name="auto (none detected)"))

    choices.extend(
        [
            Choice(value="claude-code", name="claude-code"),
            Choice(value="none", name="none (pause for manual changes)"),
        ]
    )

    return inquirer.select(
        message="Coding agent (for code changes):",
        choices=choices,
        default="auto",
    ).execute()


def prompt_context_deny() -> list[str]:
    """Prompt for additional files to hide from the LLM."""
    hide_files = inquirer.select(
        message="The LLM can read your repo. Hide any files?",
        choices=[
            Choice(value=False, name="No, allow full access (recommended)"),
            Choice(value=True, name="Yes, select files to hide"),
        ],
        default=False,
    ).execute()

    if not hide_files:
        return []

    selected_patterns: list[str] = []
    current_path = "."

    while True:
        # Build choices for current directory
        choices = []

        # Add back option if not at root
        if current_path != ".":
            choices.append(Choice(value="__back__", name="← Back"))

        # Add done option
        choices.append(
            Choice(value="__done__", name=f"✓ Done ({len(selected_patterns)} selected)")
        )

        # List entries in current directory
        try:
            entries = sorted(os.listdir(current_path))
        except OSError:
            entries = []

        dirs = []
        files = []
        for entry in entries:
            if entry.startswith("."):
                continue
            full_path = os.path.join(current_path, entry)
            rel_path = full_path[2:] if full_path.startswith("./") else full_path
            if os.path.isdir(full_path):
                dirs.append((entry, rel_path))
            else:
                files.append((entry, rel_path))

        # Add directories first (navigable)
        for name, rel_path in dirs:
            pattern = f"{rel_path}/**"
            if pattern in selected_patterns:
                choices.append(Choice(value=f"__dir__:{rel_path}", name=f"[x] {name}/"))
            else:
                choices.append(Choice(value=f"__dir__:{rel_path}", name=f"[ ] {name}/"))

        # Add files
        for name, rel_path in files:
            if rel_path in selected_patterns:
                choices.append(Choice(value=f"__file__:{rel_path}", name=f"[x] {name}"))
            else:
                choices.append(Choice(value=f"__file__:{rel_path}", name=f"[ ] {name}"))

        if len(choices) <= 2:  # Only back and done
            console.print("[dim]No files in this directory.[/dim]")
            if current_path == ".":
                return selected_patterns
            current_path = os.path.dirname(current_path) or "."
            continue

        # Show current path
        display_path = current_path if current_path != "." else "(root)"
        result = inquirer.select(
            message=f"Hide files [{display_path}]:",
            choices=choices,
        ).execute()

        if result == "__done__":
            return selected_patterns
        elif result == "__back__":
            current_path = os.path.dirname(current_path) or "."
        elif result.startswith("__dir__:"):
            rel_path = result[8:]
            pattern = f"{rel_path}/**"
            # Show submenu for directory
            dir_action = inquirer.select(
                message=f"Directory: {rel_path}/",
                choices=[
                    Choice(value="toggle", name="Toggle hide entire directory"),
                    Choice(value="enter", name="Browse contents"),
                    Choice(value="cancel", name="Cancel"),
                ],
            ).execute()
            if dir_action == "toggle":
                if pattern in selected_patterns:
                    selected_patterns.remove(pattern)
                else:
                    selected_patterns.append(pattern)
            elif dir_action == "enter":
                current_path = rel_path
        elif result.startswith("__file__:"):
            rel_path = result[9:]
            if rel_path in selected_patterns:
                selected_patterns.remove(rel_path)
            else:
                selected_patterns.append(rel_path)


def run_interactive_init() -> InitConfig:
    """Run the full interactive init flow."""
    config = InitConfig()

    console.print()

    config.train_command = prompt_training_command()
    console.print()

    source_type, source = prompt_metrics_source()
    config.metrics_source = source_type

    if source_type == "wandb" and isinstance(source, WandbMetricsSource):
        config.metrics_entity = source.get_entity()
        config.metrics_project = prompt_wandb_project(source)
        console.print()

    config.primary_metric = prompt_primary_metric(source, config.metrics_project)
    console.print()

    config.minimize = prompt_objective()
    console.print()

    executor_type, ssh_host = prompt_executor()
    config.executor_type = executor_type

    if ssh_host:
        config.ssh_host = ssh_host.hostname
        config.ssh_user = ssh_host.user
        config.ssh_port = ssh_host.port
        config.ssh_key_path = ssh_host.identity_file
    console.print()

    config.coding_agent_type = prompt_coding_agent()
    console.print()

    config.extra_deny_patterns = prompt_context_deny()
    console.print()

    return config
