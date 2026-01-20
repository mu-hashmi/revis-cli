"""eval.json metrics collector for the training loop."""

import json
import logging
import time

logger = logging.getLogger(__name__)


class EvalJsonCollector:
    """Collect metrics from eval.json file written by training script."""

    def __init__(self, executor):
        self.executor = executor

    def get_run_status(self, run_name: str) -> str | None:
        """Check if training is still running."""
        if self.executor.is_running(run_name):
            return "running"

        if self.executor.file_exists("eval.json"):
            return "finished"

        return "failed"

    def get_metrics(self, run_name: str) -> dict[str, float] | None:
        """Get metrics from eval.json."""
        try:
            if not self.executor.file_exists("eval.json"):
                return None

            content = self.executor.read_file("eval.json")
            data = json.loads(content)

            if "metrics" not in data:
                logger.warning("eval.json missing 'metrics' field")
                return None

            metrics = {}
            for key, value in data["metrics"].items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

            return metrics
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid eval.json: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading eval.json: {e}")
            return None

    def wait_for_metrics(
        self,
        run_name: str,
        timeout: int | None = None,
        poll_interval: int = 30,
    ) -> dict[str, float] | None:
        """Wait for eval.json to appear after training completes."""
        start_time = time.time()

        while True:
            status = self.get_run_status(run_name)

            if status == "finished":
                return self.get_metrics(run_name)

            if status == "failed":
                logger.warning(f"Training run {run_name} failed without producing eval.json")
                return None

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Timeout waiting for training run {run_name}")
                    return None

            time.sleep(poll_interval)
