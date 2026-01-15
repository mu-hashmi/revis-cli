"""Artifact storage protocol."""

from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    """Protocol for artifact storage."""

    def upload(self, run_id: str, local_path: Path, artifact_type: str) -> str:
        """Upload an artifact. Returns content-addressed path."""
        ...

    def download(self, uri: str, local_path: Path) -> None:
        """Download an artifact to local path."""
        ...

    def exists(self, uri: str) -> bool:
        """Check if artifact exists."""
        ...

    def list_artifacts(self, run_id: str, artifact_type: str | None = None) -> list[str]:
        """List artifacts for a run."""
        ...

    def get_url(self, uri: str) -> str:
        """Get URL for artifact (for PR links)."""
        ...
