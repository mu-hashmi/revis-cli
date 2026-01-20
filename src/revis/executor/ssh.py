"""SSH executor implementation using paramiko."""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import paramiko

from revis.executor.base import ExitResult


@dataclass
class SSHConfig:
    """SSH connection configuration."""

    host: str
    user: str
    port: int = 22
    key_path: str | None = None
    work_dir: str = "~/revis-work"


class SSHExecutor:
    """SSH-based remote executor with tmux persistence."""

    def __init__(self, config: SSHConfig):
        self.config = config
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    @property
    def client(self) -> paramiko.SSHClient:
        """Get or create SSH client."""
        if self._client is None:
            self._client = self._connect()
        return self._client

    @property
    def sftp(self) -> paramiko.SFTPClient:
        """Get or create SFTP client."""
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    def _connect(self) -> paramiko.SSHClient:
        """Establish SSH connection."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.user,
        }

        if self.config.key_path:
            connect_kwargs["key_filename"] = os.path.expanduser(self.config.key_path)
        else:
            # Use SSH agent
            connect_kwargs["allow_agent"] = True

        client.connect(**connect_kwargs)
        return client

    def reconnect(self) -> bool:
        """Attempt to reconnect after connection loss."""
        self.close()
        try:
            self._client = self._connect()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close SSH connection."""
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def _exec(self, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        """Execute command and return (exit_code, stdout, stderr)."""
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(), stderr.read().decode()

    def _expand_path(self, path: str) -> str:
        """Expand ~ in remote path."""
        if path.startswith("~"):
            exit_code, home, _ = self._exec("echo $HOME")
            if exit_code == 0:
                return path.replace("~", home.strip(), 1)
        return path

    # Process management with tmux

    def launch(
        self,
        command: str,
        env: dict[str, str],
        session_name: str,
    ) -> str:
        """Launch command in tmux session. Returns session name as process ID."""
        work_dir = self._expand_path(self.config.work_dir)

        # Build environment exports (each var needs separate export for safety)
        env_exports = " && ".join(f'export {k}="{v}"' for k, v in env.items())
        if env_exports:
            env_exports = f"{env_exports} && "

        # Create tmux session with the command
        tmux_cmd = (
            f"tmux new-session -d -s {session_name} "
            f"'cd {work_dir} && {env_exports}{command}; echo \"EXIT_CODE=$?\" >> .revis_exit'"
        )

        exit_code, _, stderr = self._exec(tmux_cmd)
        if exit_code != 0:
            # Session might already exist, try killing and recreating
            self._exec(f"tmux kill-session -t {session_name} 2>/dev/null")
            exit_code, _, stderr = self._exec(tmux_cmd)
            if exit_code != 0:
                raise RuntimeError(f"Failed to create tmux session: {stderr}")

        return session_name

    def is_running(self, process_id: str) -> bool:
        """Check if tmux session is still running."""
        exit_code, _, _ = self._exec(f"tmux has-session -t {process_id} 2>/dev/null")
        return exit_code == 0

    def wait(self, process_id: str, timeout: int | None = None) -> ExitResult:
        """Wait for process completion."""
        work_dir = self._expand_path(self.config.work_dir)
        exit_file = f"{work_dir}/.revis_exit"

        start_time = time.time()
        poll_interval = 5  # seconds

        while True:
            # Check if session still exists
            if not self.is_running(process_id):
                # Session ended, check exit file
                exit_code, content, _ = self._exec(
                    f"cat {exit_file} 2>/dev/null && rm -f {exit_file}"
                )
                if exit_code == 0 and "EXIT_CODE=" in content:
                    code = int(content.split("EXIT_CODE=")[1].strip())
                    return ExitResult(
                        exit_code=code,
                        failed=code != 0,
                        error_message=None if code == 0 else f"Process exited with code {code}",
                    )
                # No exit file - session ended but we can't verify exit code
                # Assume failure to be safe
                return ExitResult(
                    exit_code=-1,
                    failed=True,
                    error_message="Process ended but exit code unavailable",
                )

            # Check timeout
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
        self._exec(f"tmux kill-session -t {process_id} 2>/dev/null")

    # Log collection

    def stream_logs(self, process_id: str, tail_lines: int = 200) -> Iterator[str]:
        """Stream logs from tmux session."""
        while self.is_running(process_id):
            # Capture current pane content
            exit_code, output, _ = self._exec(
                f"tmux capture-pane -t {process_id} -p -S -{tail_lines}"
            )
            if exit_code == 0:
                yield output
            time.sleep(2)

    def get_log_tail(self, log_path: str, lines: int = 200) -> str:
        """Get last N lines of a log file."""
        work_dir = self._expand_path(self.config.work_dir)
        full_path = f"{work_dir}/{log_path}"

        exit_code, output, stderr = self._exec(f"tail -n {lines} {full_path} 2>/dev/null")
        if exit_code != 0:
            # Try getting from tmux pane as fallback
            exit_code, output, _ = self._exec(
                f"tmux capture-pane -t revis -p -S -{lines} 2>/dev/null"
            )
        return output

    def get_tmux_output(self, session_name: str, lines: int = 200) -> str:
        """Get output from tmux pane."""
        exit_code, output, _ = self._exec(f"tmux capture-pane -t {session_name} -p -S -{lines}")
        return output if exit_code == 0 else ""

    # Code sync

    def sync_code(self, local_path: Path, remote_path: str) -> None:
        """Sync local code to remote using rsync."""
        remote_path = self._expand_path(remote_path)

        # Ensure remote directory exists
        self._exec(f"mkdir -p {remote_path}")

        # Build rsync command
        # Respect .gitignore by using --filter
        # NOTE: No --delete flag - preserves data/artifacts that exist only on remote
        rsync_cmd = [
            "rsync",
            "-avz",
            "--filter=:- .gitignore",  # Respect .gitignore
            "--exclude=.git",
            "--exclude=.revis",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.venv",
            "--exclude=venv",
            "-e",
            f"ssh -p {self.config.port}",
            f"{local_path}/",
            f"{self.config.user}@{self.config.host}:{remote_path}/",
        ]

        # Add key if specified
        if self.config.key_path:
            key_path = os.path.expanduser(self.config.key_path)
            rsync_cmd[rsync_cmd.index("-e") + 1] = f"ssh -p {self.config.port} -i {key_path}"

        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rsync failed: {result.stderr}")

    # Artifact collection

    def collect_artifacts(
        self,
        patterns: list[str],
        since_timestamp: float,
        local_dest: Path,
    ) -> list[Path]:
        """Collect artifacts matching patterns modified after timestamp."""
        work_dir = self._expand_path(self.config.work_dir)
        collected = []

        for pattern in patterns:
            # Find files matching pattern modified after timestamp
            # Using find with -newer would require a reference file, so use stat
            find_cmd = f"find {work_dir}/{pattern} -type f 2>/dev/null"
            exit_code, output, _ = self._exec(find_cmd)

            if exit_code != 0 or not output.strip():
                continue

            for remote_file in output.strip().split("\n"):
                if not remote_file:
                    continue

                # Check modification time
                stat_cmd = f"stat -c %Y {remote_file} 2>/dev/null"
                exit_code, mtime_str, _ = self._exec(stat_cmd)

                if exit_code != 0:
                    # Try macOS stat format
                    stat_cmd = f"stat -f %m {remote_file} 2>/dev/null"
                    exit_code, mtime_str, _ = self._exec(stat_cmd)

                if exit_code == 0:
                    try:
                        mtime = float(mtime_str.strip())
                        if mtime < since_timestamp:
                            continue  # Skip old files
                    except ValueError:
                        pass  # Include if we can't parse

                # Download the file
                rel_path = remote_file.replace(work_dir + "/", "")
                local_file = local_dest / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    self.sftp.get(remote_file, str(local_file))
                    collected.append(local_file)
                except Exception as e:
                    # Log but continue
                    print(f"Warning: Failed to collect {remote_file}: {e}")

        return collected

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download a single file from remote."""
        work_dir = self._expand_path(self.config.work_dir)
        full_remote = f"{work_dir}/{remote_path}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(full_remote, str(local_path))

    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on remote."""
        work_dir = self._expand_path(self.config.work_dir)
        full_path = f"{work_dir}/{remote_path}"
        exit_code, _, _ = self._exec(f"test -f {full_path}")
        return exit_code == 0

    def read_file(self, remote_path: str) -> str:
        """Read file content from remote."""
        work_dir = self._expand_path(self.config.work_dir)
        full_path = f"{work_dir}/{remote_path}"
        exit_code, content, _ = self._exec(f"cat {full_path}")
        if exit_code != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        return content
