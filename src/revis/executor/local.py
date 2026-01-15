"""Local executor for running training on the same machine."""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from revis.executor.base import ExitResult


@dataclass
class LocalConfig:
    """Local executor configuration."""

    work_dir: str = "."


class LocalExecutor:
    """Local executor using tmux for persistence."""

    def __init__(self, config: LocalConfig):
        self.config = config
        self._work_dir = Path(config.work_dir).expanduser().resolve()

    def _run(self, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        """Run a shell command locally."""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self._work_dir,
        )
        return result.returncode, result.stdout, result.stderr

    def launch(
        self,
        command: str,
        env: dict[str, str],
        session_name: str,
    ) -> str:
        """Launch command in local tmux session."""
        env_exports = " && ".join(f"export {k}=\"{v}\"" for k, v in env.items())
        if env_exports:
            env_exports = f"{env_exports} && "

        exit_file = self._work_dir / ".revis_exit"
        exit_file.unlink(missing_ok=True)

        tmux_cmd = (
            f"tmux new-session -d -s {session_name} "
            f"'cd {self._work_dir} && {env_exports}{command}; echo \"EXIT_CODE=$?\" >> .revis_exit'"
        )

        exit_code, _, stderr = self._run(tmux_cmd)
        if exit_code != 0:
            self._run(f"tmux kill-session -t {session_name} 2>/dev/null")
            exit_code, _, stderr = self._run(tmux_cmd)
            if exit_code != 0:
                raise RuntimeError(f"Failed to create tmux session: {stderr}")

        return session_name

    def is_running(self, process_id: str) -> bool:
        """Check if tmux session exists."""
        exit_code, _, _ = self._run(f"tmux has-session -t {process_id} 2>/dev/null")
        return exit_code == 0

    def wait(self, process_id: str, timeout: int | None = None) -> ExitResult:
        """Wait for process completion."""
        exit_file = self._work_dir / ".revis_exit"
        start_time = time.time()
        poll_interval = 5

        while True:
            if not self.is_running(process_id):
                if exit_file.exists():
                    content = exit_file.read_text()
                    exit_file.unlink(missing_ok=True)
                    if "EXIT_CODE=" in content:
                        code = int(content.split("EXIT_CODE=")[1].strip())
                        return ExitResult(
                            exit_code=code,
                            failed=code != 0,
                            error_message=None if code == 0 else f"Process exited with code {code}",
                        )
                # No exit file - session ended but we can't verify exit code
                return ExitResult(
                    exit_code=-1,
                    failed=True,
                    error_message="Process ended but exit code unavailable",
                )

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.kill(process_id)
                    return ExitResult(
                        exit_code=-1,
                        failed=True,
                        error_message=f"Process timed out after {timeout}s",
                    )

            time.sleep(poll_interval)

    def kill(self, process_id: str) -> None:
        """Kill tmux session."""
        self._run(f"tmux kill-session -t {process_id} 2>/dev/null")

    def stream_logs(self, process_id: str, tail_lines: int = 200) -> Iterator[str]:
        """Stream logs from tmux session."""
        while self.is_running(process_id):
            exit_code, output, _ = self._run(
                f"tmux capture-pane -t {process_id} -p -S -{tail_lines}"
            )
            if exit_code == 0:
                yield output
            time.sleep(2)

    def get_log_tail(self, log_path: str, lines: int = 200) -> str:
        """Get last N lines of log file."""
        full_path = self._work_dir / log_path
        if full_path.exists():
            exit_code, output, _ = self._run(f"tail -n {lines} {full_path}")
            return output
        return ""

    def get_tmux_output(self, session_name: str, lines: int = 200) -> str:
        """Get output from tmux pane."""
        exit_code, output, _ = self._run(
            f"tmux capture-pane -t {session_name} -p -S -{lines}"
        )
        return output if exit_code == 0 else ""

    def sync_code(self, local_path: Path, remote_path: str) -> None:
        """No-op for local executor - code is already here."""
        pass

    def collect_artifacts(
        self,
        patterns: list[str],
        since_timestamp: float,
        local_dest: Path,
    ) -> list[Path]:
        """Collect artifacts matching patterns."""
        import glob
        collected = []

        for pattern in patterns:
            full_pattern = str(self._work_dir / pattern)
            for file_path in glob.glob(full_pattern, recursive=True):
                path = Path(file_path)
                if path.is_file() and path.stat().st_mtime >= since_timestamp:
                    rel_path = path.relative_to(self._work_dir)
                    dest_path = local_dest / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if path != dest_path:
                        import shutil
                        shutil.copy2(path, dest_path)
                    collected.append(dest_path)

        return collected

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return (self._work_dir / path).exists()

    def read_file(self, path: str) -> str:
        """Read file content."""
        full_path = self._work_dir / path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_text()

    def reconnect(self) -> bool:
        """No-op for local executor."""
        return True

    def close(self) -> None:
        """No-op for local executor."""
        pass
