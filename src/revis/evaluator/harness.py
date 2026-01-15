"""Evaluation harness for collecting and validating eval results."""

import json
from pathlib import Path

from revis.executor.ssh import SSHExecutor
from revis.types import EvalResult


class EvalHarness:
    """Harness for running evaluation and collecting results."""

    def __init__(self, executor: SSHExecutor):
        self.executor = executor

    def run_eval(self, eval_command: str, session_name: str) -> EvalResult:
        """Run evaluation command and collect results."""
        # Launch eval in same tmux session or new one
        eval_session = f"{session_name}-eval"
        self.executor.launch(eval_command, {}, eval_session)

        # Wait for completion
        result = self.executor.wait(eval_session, timeout=3600)  # 1 hour max for eval

        if result.failed:
            raise RuntimeError(f"Evaluation failed: {result.error_message}")

        # Collect results
        return self.collect()

    def collect(self, eval_path: str = "eval.json") -> EvalResult:
        """Collect eval.json from remote and parse it."""
        if not self.executor.file_exists(eval_path):
            raise FileNotFoundError(f"Evaluation output not found: {eval_path}")

        content = self.executor.read_file(eval_path)
        return self.parse_eval_json(content)

    def parse_eval_json(self, content: str) -> EvalResult:
        """Parse eval.json content into EvalResult."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid eval.json: {e}")

        # Validate required fields
        if "metrics" not in data:
            raise ValueError("eval.json must contain 'metrics' field")

        metrics = data["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("eval.json 'metrics' must be a dictionary")

        # Validate metric values are numeric
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric '{name}' must be numeric, got {type(value).__name__}")

        return EvalResult(
            metrics=metrics,
            slices=data.get("slices", {}),
            plots=data.get("plots", []),
        )

    def collect_plots(self, plots: list[str], local_dest: Path) -> list[Path]:
        """Collect plot files from remote."""
        collected = []
        for plot in plots:
            local_path = local_dest / plot
            try:
                self.executor.download_file(plot, local_path)
                collected.append(local_path)
            except Exception as e:
                print(f"Warning: Failed to collect plot {plot}: {e}")
        return collected


def validate_eval_schema(data: dict) -> list[str]:
    """Validate eval.json schema and return list of errors."""
    errors = []

    if "metrics" not in data:
        errors.append("Missing required field: 'metrics'")
    elif not isinstance(data["metrics"], dict):
        errors.append("'metrics' must be a dictionary")
    else:
        for name, value in data["metrics"].items():
            if not isinstance(value, (int, float)):
                errors.append(f"Metric '{name}' must be numeric")

    if "slices" in data:
        if not isinstance(data["slices"], dict):
            errors.append("'slices' must be a dictionary")

    if "plots" in data:
        if not isinstance(data["plots"], list):
            errors.append("'plots' must be a list")
        elif not all(isinstance(p, str) for p in data["plots"]):
            errors.append("All items in 'plots' must be strings")

    return errors
