"""Executor protocol for remote command execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


@dataclass
class ExitResult:
    """Result of a completed process."""

    exit_code: int
    failed: bool
    error_message: str | None = None


class Executor(Protocol):
    """Protocol for remote command execution."""

    def launch(
        self,
        command: str,
        env: dict[str, str],
        session_name: str,
    ) -> str:
        """Launch a command in a persistent session. Returns process identifier."""
        ...

    def stream_logs(self, process_id: str, tail_lines: int = 200) -> Iterator[str]:
        """Stream logs from a running process."""
        ...

    def get_log_tail(self, log_path: str, lines: int = 200) -> str:
        """Get the last N lines of a log file."""
        ...

    def wait(self, process_id: str, timeout: int | None = None) -> ExitResult:
        """Wait for process completion."""
        ...

    def kill(self, process_id: str) -> None:
        """Kill a running process."""
        ...

    def is_running(self, process_id: str) -> bool:
        """Check if process is still running."""
        ...

    def collect_artifacts(self, patterns: list[str], since_timestamp: float) -> list[Path]:
        """Collect artifacts matching patterns modified after timestamp."""
        ...

    def sync_code(self, local_path: Path, remote_path: str) -> None:
        """Sync local code to remote."""
        ...

    def reconnect(self) -> bool:
        """Attempt to reconnect after connection loss."""
        ...

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        ...

    def read_file(self, path: str) -> str:
        """Read file content."""
        ...

    def close(self) -> None:
        """Close executor and cleanup resources."""
        ...
