"""MetricsSource protocol for init-time metric source detection."""

from typing import Protocol


class MetricsSource(Protocol):
    """Protocol for metrics source backends used during init."""

    @staticmethod
    def detect_auth() -> bool:
        """Check if authentication is available (env var, netrc, etc.)."""
        ...

    def connect(self) -> bool:
        """Attempt to connect and authenticate. Returns True on success."""
        ...

    def list_projects(self) -> list[str]:
        """List available projects/experiments."""
        ...

    def list_metrics(self, project: str, limit_runs: int = 10) -> list[str]:
        """Get metric keys from recent runs in a project."""
        ...

    def get_source_type(self) -> str:
        """Return the source type identifier (e.g., 'wandb', 'mlflow')."""
        ...

    def get_entity(self) -> str | None:
        """Return the entity/namespace (e.g., W&B username or team)."""
        ...
