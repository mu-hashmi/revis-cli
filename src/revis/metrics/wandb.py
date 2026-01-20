"""Weights & Biases metrics collector for the training loop."""

import logging
import time

logger = logging.getLogger(__name__)


class WandbCollector:
    """Collect metrics from W&B runs."""

    def __init__(self, project: str, entity: str | None = None):
        self.project = project
        self.entity = entity
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                import wandb

                self._api = wandb.Api()
                if self.entity is None:
                    self.entity = self._api.default_entity
            except ImportError:
                raise ImportError("wandb is not installed. Install with: pip install wandb")
        return self._api

    def _find_run(self, run_name: str):
        """Find a run by name in the project."""
        api = self._get_api()
        try:
            runs = api.runs(
                path=f"{self.entity}/{self.project}",
                filters={"display_name": run_name},
                per_page=5,
            )
            for run in runs:
                if run.name == run_name:
                    return run
            return None
        except Exception as e:
            logger.warning(f"Error finding W&B run: {e}")
            return None

    def get_run_status(self, run_name: str) -> str | None:
        """Get the status of a run."""
        run = self._find_run(run_name)
        if run is None:
            return None
        return run.state

    def get_metrics(self, run_name: str) -> dict[str, float] | None:
        """Get metrics for a completed run."""
        run = self._find_run(run_name)
        if run is None:
            return None

        if run.state not in ("finished", "crashed"):
            return None

        try:
            metrics = {}
            for key, value in run.summary.items():
                if not key.startswith("_") and isinstance(value, (int, float)):
                    metrics[key] = float(value)
            return metrics
        except Exception as e:
            logger.warning(f"Error getting W&B metrics: {e}")
            return None

    def wait_for_metrics(
        self,
        run_name: str,
        timeout: int | None = None,
        poll_interval: int = 30,
    ) -> dict[str, float] | None:
        """Wait for metrics from a training run."""
        start_time = time.time()

        while True:
            status = self.get_run_status(run_name)

            if status == "finished":
                return self.get_metrics(run_name)

            if status == "crashed":
                logger.warning(f"W&B run {run_name} crashed")
                return None

            if status is None:
                logger.debug(f"W&B run {run_name} not found yet, waiting...")

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Timeout waiting for W&B run {run_name}")
                    return None

            time.sleep(poll_interval)
