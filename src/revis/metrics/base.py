"""MetricsCollector protocol for collecting metrics during training."""

from typing import Protocol


class MetricsCollector(Protocol):
    """Protocol for collecting metrics from training runs."""

    def wait_for_metrics(
        self,
        run_name: str,
        timeout: int | None = None,
        poll_interval: int = 30,
    ) -> dict[str, float] | None:
        """
        Wait for metrics from a training run.

        Args:
            run_name: Name/ID of the run to wait for
            timeout: Maximum seconds to wait (None = no timeout)
            poll_interval: Seconds between polls

        Returns:
            Metrics dict if available, None if timed out or run failed
        """
        ...

    def get_metrics(self, run_name: str) -> dict[str, float] | None:
        """
        Get metrics for a completed run.

        Args:
            run_name: Name/ID of the run

        Returns:
            Metrics dict if available, None if run not found or not complete
        """
        ...

    def get_run_status(self, run_name: str) -> str | None:
        """
        Get the status of a run.

        Returns:
            "running", "finished", "failed", "crashed", or None if not found
        """
        ...
