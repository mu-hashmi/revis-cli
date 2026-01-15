"""Artifact storage module."""

from revis.artifacts.base import ArtifactStore
from revis.artifacts.local import LocalArtifactStore

__all__ = ["ArtifactStore", "LocalArtifactStore"]
