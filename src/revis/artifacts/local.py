"""Local content-addressed artifact storage."""

import hashlib
import shutil
from pathlib import Path


class LocalArtifactStore:
    """Local filesystem artifact store with content-addressed storage."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload(self, run_id: str, local_path: Path, artifact_type: str) -> str:
        """Upload an artifact. Returns content-addressed path."""
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")

        # Compute SHA1 hash
        content_hash = self._compute_hash(local_path)

        # Store in content-addressed path: ab/cdef1234...
        prefix = content_hash[:2]
        suffix = content_hash[2:]
        store_dir = self.base_path / prefix
        store_dir.mkdir(exist_ok=True)
        store_path = store_dir / suffix

        # Only copy if not already stored (dedup)
        if not store_path.exists():
            shutil.copy2(local_path, store_path)

        # Return the content-addressed URI
        return f"local://{prefix}/{suffix}"

    def download(self, uri: str, local_path: Path) -> None:
        """Download an artifact to local path."""
        store_path = self._uri_to_path(uri)
        if not store_path.exists():
            raise FileNotFoundError(f"Artifact not found: {uri}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(store_path, local_path)

    def exists(self, uri: str) -> bool:
        """Check if artifact exists."""
        try:
            store_path = self._uri_to_path(uri)
            return store_path.exists()
        except ValueError:
            return False

    def list_artifacts(self, run_id: str, artifact_type: str | None = None) -> list[str]:
        """List all artifacts (not filtered by run_id - that's in the store)."""
        # For local storage, we don't maintain a run_id -> artifacts mapping
        # That's handled by the RunStore. This just lists all stored artifacts.
        artifacts = []
        for prefix_dir in self.base_path.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for artifact in prefix_dir.iterdir():
                    if artifact.is_file():
                        artifacts.append(f"local://{prefix_dir.name}/{artifact.name}")
        return artifacts

    def get_url(self, uri: str) -> str:
        """Get URL for artifact. For local storage, returns file:// URL."""
        store_path = self._uri_to_path(uri)
        return f"file://{store_path.absolute()}"

    def get_size(self, uri: str) -> int:
        """Get artifact size in bytes."""
        store_path = self._uri_to_path(uri)
        return store_path.stat().st_size

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA1 hash of file."""
        sha1 = hashlib.sha1()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()

    def _uri_to_path(self, uri: str) -> Path:
        """Convert URI to local path."""
        if not uri.startswith("local://"):
            raise ValueError(f"Invalid local artifact URI: {uri}")

        rel_path = uri[len("local://"):]
        return self.base_path / rel_path
