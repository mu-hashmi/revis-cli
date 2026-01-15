"""Executor module for remote command execution."""

from revis.executor.base import Executor, ExitResult
from revis.executor.local import LocalConfig, LocalExecutor
from revis.executor.ssh import SSHConfig, SSHExecutor

__all__ = [
    "Executor",
    "ExitResult",
    "LocalConfig",
    "LocalExecutor",
    "SSHConfig",
    "SSHExecutor",
]
