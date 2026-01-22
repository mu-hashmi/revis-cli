"""Weights & Biases metrics source implementation."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDED_METRICS = {
    "epoch",
    "step",
    "global_step",
    "batch",
    "iteration",
    "learning_rate",
    "lr",
    "grad_norm",
    "gradient_norm",
    "samples_per_second",
    "steps_per_second",
    "samples_seen",
    "tokens_seen",
    "wall_time",
    "timestamp",
}


def is_optimizable_metric(key: str) -> bool:
    """Check if a metric is meaningful to optimize."""
    key_lower = key.lower()
    if key_lower in EXCLUDED_METRICS:
        return False
    if key_lower.endswith("_lr") or key_lower.endswith("_learning_rate"):
        return False
    if key_lower.startswith("lr_") or key_lower.startswith("learning_rate_"):
        return False
    return True


class WandbMetricsSource:
    """Weights & Biases metrics source for init."""

    def __init__(self):
        self._api = None
        self._entity: str | None = None

    @staticmethod
    def detect_auth() -> bool:
        """Check for W&B API key in env or netrc."""
        if os.environ.get("WANDB_API_KEY"):
            return True

        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            try:
                content = netrc_path.read_text()
                if "api.wandb.ai" in content:
                    return True
            except Exception:
                pass
        return False

    def connect(self) -> bool:
        """Connect to W&B API."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb package not installed - run 'pip install wandb'")
            return False

        try:
            self._api = wandb.Api()
            self._entity = self._api.default_entity
            return True
        except Exception as e:
            logger.warning(f"W&B connection failed: {e}")
            return False

    def list_projects(self) -> list[str]:
        """List user's W&B projects."""
        if self._api is None:
            return []

        try:
            projects = self._api.projects(entity=self._entity)
            return [p.name for p in projects]
        except Exception:
            return []

    def list_metrics(self, project: str, limit_runs: int = 10) -> list[str]:
        """Get metric keys from recent runs."""
        if self._api is None:
            return []

        try:
            runs = self._api.runs(
                path=f"{self._entity}/{project}",
                per_page=limit_runs,
            )

            metric_keys = set()
            for run in runs:
                for key in run.summary.keys():
                    if not key.startswith("_") and is_optimizable_metric(key):
                        metric_keys.add(key)

            return sorted(metric_keys)
        except Exception:
            return []

    def get_source_type(self) -> str:
        return "wandb"

    def get_entity(self) -> str | None:
        return self._entity
