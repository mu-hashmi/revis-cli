"""eval.json fallback metrics source."""


class EvalJsonMetricsSource:
    """Fallback that uses eval.json written by training script."""

    @staticmethod
    def detect_auth() -> bool:
        """Always available - no auth needed."""
        return True

    def connect(self) -> bool:
        """No connection needed."""
        return True

    def list_projects(self) -> list[str]:
        """Not applicable for eval.json."""
        return []

    def list_metrics(self, project: str = "", limit_runs: int = 10) -> list[str]:
        """Return common metric suggestions."""
        return ["loss", "val_loss", "accuracy", "eval_loss", "perplexity"]

    def get_source_type(self) -> str:
        return "eval_json"

    def get_entity(self) -> str | None:
        return None
