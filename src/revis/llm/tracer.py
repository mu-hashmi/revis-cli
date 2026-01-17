"""Agent tracing with rich terminal output and persistence."""

from typing import Protocol

from rich.console import Console
from rich.panel import Panel


class TracerBackend(Protocol):
    """Protocol for trace persistence."""

    def log_trace(self, run_id: str, event_type: str, data: dict) -> None: ...


class AgentTracer:
    """Traces agent tool calls with rich terminal output and persistence."""

    def __init__(
        self,
        console: Console,
        backend: TracerBackend,
        run_id: str,
    ):
        self.console = console
        self.backend = backend
        self.run_id = run_id

    def on_tool_call(self, name: str, args: dict) -> None:
        self.console.print(f"[dim]→[/dim] [cyan]{name}[/cyan]", end="")
        self._print_args(name, args)
        self.backend.log_trace(self.run_id, "tool_call", {"tool": name, "args": args})

    def on_tool_result(self, name: str, result: str) -> None:
        self._print_result_summary(name, result)
        self.backend.log_trace(
            self.run_id,
            "tool_result",
            {"tool": name, "result": result[:1000]},
        )

    def on_iteration_complete(self, iteration: int, rationale: str, significant: bool) -> None:
        style = "bold green" if significant else "green"
        marker = " [SIGNIFICANT]" if significant else ""
        self.console.print(Panel(rationale + marker, title=f"Iteration {iteration}", style=style))

    def _print_args(self, name: str, args: dict) -> None:
        if name == "read_file":
            path = args.get("path", "?")
            self.console.print(f"([yellow]{path}[/yellow])")
        elif name == "write_file":
            path = args.get("path", "?")
            content = args.get("content", "")
            self.console.print(f"([yellow]{path}[/yellow]) [dim]{len(content)} bytes[/dim]")
        elif name == "search_codebase":
            pattern = args.get("pattern", "?")
            self.console.print(f"([yellow]{pattern}[/yellow])")
        elif name == "find_definition":
            target = args.get("name", "?")
            self.console.print(f"([yellow]{target}[/yellow])")
        elif name == "list_directory":
            path = args.get("path", ".")
            recursive = args.get("recursive", False)
            suffix = " [dim]recursive[/dim]" if recursive else ""
            self.console.print(f"([yellow]{path}[/yellow]){suffix}")
        elif name == "run_command":
            cmd = args.get("command", "?")
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            self.console.print(f"([yellow]{cmd}[/yellow])")
        else:
            self.console.print(f"({args})")

    def _print_result_summary(self, name: str, result: str) -> None:
        if name == "search_codebase":
            if result == "No matches found":
                self.console.print("    [dim]no matches[/dim]")
            else:
                lines = result.strip().split("\n")
                self.console.print(f"    [dim]{len(lines)} matches[/dim]")
        elif name == "find_definition":
            if result == "No matches found":
                self.console.print("    [dim]not found[/dim]")
            else:
                lines = result.strip().split("\n")
                self.console.print(f"    [dim]{len(lines)} matches[/dim]")
        elif name == "read_file":
            if result.startswith("File not found") or result.startswith("Access denied"):
                self.console.print(f"    [red]{result}[/red]")
            else:
                lines = result.strip().split("\n")
                self.console.print(f"    [dim]{len(lines)} lines[/dim]")
        elif name == "list_directory":
            if result == "(empty)":
                self.console.print("    [dim]empty[/dim]")
            else:
                items = result.strip().split("\n")
                self.console.print(f"    [dim]{len(items)} items[/dim]")
        elif name == "run_command":
            if result == "(no output)":
                self.console.print("    [dim]ok[/dim]")
            elif "Error" in result or "error" in result:
                preview = result[:60].replace("\n", " ")
                self.console.print(f"    [red]{preview}...[/red]")
            else:
                preview = result[:60].replace("\n", " ")
                self.console.print(f"    [dim]{preview}[/dim]")
