"""Weights & Biases metrics collector for the training loop."""

import importlib.util
import logging
import re
import sys

logger = logging.getLogger(__name__)


def _import_wandb():
    """Import the real wandb package, avoiding local wandb/ directories.

    Many ML projects have a wandb/ directory where W&B stores run logs.
    This can shadow the actual wandb package when importing.
    """
    import wandb as _wandb

    # Check if we got the real package (has Api class) or a namespace/directory
    if hasattr(_wandb, "Api"):
        return _wandb

    # The normal import got a local wandb/ directory - find the real package
    # Search site-packages for the real wandb installation
    import os

    for path in sys.path:
        if "site-packages" in path:
            wandb_init = os.path.join(path, "wandb", "__init__.py")
            if os.path.isfile(wandb_init):
                spec = importlib.util.spec_from_file_location(
                    "wandb_real", wandb_init,
                    submodule_search_locations=[os.path.join(path, "wandb")]
                )
                if spec and spec.loader:
                    real_wandb = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(real_wandb)
                    return real_wandb

    raise ImportError(
        "wandb is not installed or is shadowed by a local 'wandb/' directory. "
        "Install wandb with: pip install wandb"
    )

# Pattern to match W&B run URL in training logs
# Matches: "🚀 View run at https://wandb.ai/entity/project/runs/abc123"
WANDB_RUN_URL_PATTERN = re.compile(
    r"View run at https://wandb\.ai/[^/]+/[^/]+/runs/([a-zA-Z0-9]+)"
)


class WandbCollector:
    """Collect metrics from W&B runs."""

    def __init__(self, project: str, entity: str | None = None):
        self.project = project
        self.entity = entity
        self._api = None

    def _get_api(self):
        if self._api is None:
            wandb = _import_wandb()
            self._api = wandb.Api()
            if self.entity is None:
                self.entity = self._api.default_entity
        return self._api

    def parse_run_id_from_log(self, log_content: str) -> str | None:
        """Parse W&B run ID from training log output.

        W&B prints a line like:
            wandb: 🚀 View run at https://wandb.ai/entity/project/runs/abc123

        This extracts the run ID (abc123) from that URL.
        """
        match = WANDB_RUN_URL_PATTERN.search(log_content)
        if match:
            run_id = match.group(1)
            logger.debug(f"Parsed W&B run ID from log: {run_id}")
            return run_id
        logger.warning("Could not find W&B run URL in training log")
        return None

    def _get_run_by_id(self, run_id: str):
        """Get a W&B run by its ID."""
        api = self._get_api()
        try:
            run_path = f"{self.entity}/{self.project}/{run_id}"
            return api.run(run_path)
        except Exception as e:
            logger.warning(f"Error fetching W&B run {run_id}: {e}")
            return None

    def get_metrics_from_log(self, log_content: str) -> dict[str, float] | None:
        """Get metrics by parsing W&B run ID from training log.

        This is the primary method for getting metrics - it doesn't require
        the user's training script to use any specific naming convention.
        """
        run_id = self.parse_run_id_from_log(log_content)
        if run_id is None:
            return None

        run = self._get_run_by_id(run_id)
        if run is None:
            return None

        if run.state not in ("finished", "crashed"):
            logger.warning(f"W&B run {run_id} is not finished (state: {run.state})")
            return None

        try:
            metrics = {}
            for key, value in run.summary.items():
                if not key.startswith("_") and isinstance(value, (int, float)):
                    metrics[key] = float(value)
            logger.info(f"Retrieved {len(metrics)} metrics from W&B run {run_id}")
            return metrics
        except Exception as e:
            logger.warning(f"Error getting W&B metrics: {e}")
            return None
