"""Run comparison and analysis."""

from dataclasses import dataclass

from revis.store.sqlite import SQLiteRunStore
from revis.types import Analysis, EvalResult, Session


@dataclass
class RunComparison:
    """Comparison between two runs."""

    current_value: float
    previous_value: float | None
    baseline_value: float | None
    delta_from_previous: float | None
    delta_from_baseline: float | None
    improvement_from_previous: float | None  # Percentage
    improvement_from_baseline: float | None  # Percentage


@dataclass
class RunSummary:
    """Summary of a run for LLM context."""

    iteration: int
    metrics: dict[str, float]
    change_summary: str  # e.g., "Changed lr 1e-4 → 1e-3"
    result_summary: str  # e.g., "loss: 0.5 → 0.4 (-20%)"


class RunAnalyzer:
    """Analyzer for comparing runs and generating insights."""

    def __init__(self, store: SQLiteRunStore, primary_metric: str, minimize: bool = True):
        self.store = store
        self.primary_metric = primary_metric
        self.minimize = minimize

    def compare_to_previous(
        self,
        current_eval: EvalResult,
        previous_eval: EvalResult | None,
        baseline_eval: EvalResult | None = None,
    ) -> RunComparison:
        """Compare current run to previous and baseline."""
        current_value = current_eval.metrics.get(self.primary_metric, 0.0)

        previous_value = None
        delta_prev = None
        improvement_prev = None

        if previous_eval:
            previous_value = previous_eval.metrics.get(self.primary_metric)
            if previous_value is not None:
                delta_prev = current_value - previous_value
                if previous_value != 0:
                    improvement_prev = -delta_prev / abs(previous_value) if self.minimize else delta_prev / abs(previous_value)

        baseline_value = None
        delta_base = None
        improvement_base = None

        if baseline_eval:
            baseline_value = baseline_eval.metrics.get(self.primary_metric)
            if baseline_value is not None:
                delta_base = current_value - baseline_value
                if baseline_value != 0:
                    improvement_base = -delta_base / abs(baseline_value) if self.minimize else delta_base / abs(baseline_value)

        return RunComparison(
            current_value=current_value,
            previous_value=previous_value,
            baseline_value=baseline_value,
            delta_from_previous=delta_prev,
            delta_from_baseline=delta_base,
            improvement_from_previous=improvement_prev,
            improvement_from_baseline=improvement_base,
        )

    def get_metric_history(self, session_id: str) -> list[float]:
        """Get history of primary metric values for a session."""
        runs = self.store.query_runs(session_id=session_id, limit=100)
        history = []

        for run in reversed(runs):  # Oldest first
            metrics = self.store.get_run_metrics(run.id)
            for m in metrics:
                if m.name == self.primary_metric:
                    history.append(m.value)
                    break

        return history

    def get_initial_value(self, session_id: str) -> float | None:
        """Get the initial metric value for divergence detection."""
        history = self.get_metric_history(session_id)
        return history[0] if history else None

    def analyze_run(
        self,
        session: Session,
        current_eval: EvalResult,
        previous_eval: EvalResult | None,
    ) -> Analysis:
        """Analyze a run and return structured analysis."""
        history = self.get_metric_history(session.id)

        # Check for plateau
        plateau_detected = False
        if len(history) >= 3:  # Need at least 3 runs
            from revis.analyzer.detectors import detect_plateau
            result = detect_plateau(
                history + [current_eval.metrics.get(self.primary_metric, 0.0)],
                threshold=0.01,
                n_runs=3,
                minimize=self.minimize,
            )
            plateau_detected = result.triggered

        # Calculate metric delta
        metric_delta = None
        if previous_eval:
            prev_value = previous_eval.metrics.get(self.primary_metric)
            curr_value = current_eval.metrics.get(self.primary_metric)
            if prev_value is not None and curr_value is not None:
                metric_delta = curr_value - prev_value

        return Analysis(
            plateau_detected=plateau_detected,
            metric_delta=metric_delta,
            guardrail_violations=[],
        )

    def summarize_runs_for_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[RunSummary]:
        """Generate run summaries for LLM context."""
        runs = self.store.query_runs(session_id=session_id, limit=limit)
        summaries = []

        prev_metrics: dict[str, float] = {}

        for run in reversed(runs):  # Oldest first
            metrics_list = self.store.get_run_metrics(run.id)
            metrics = {m.name: m.value for m in metrics_list}

            # Generate change summary from decisions
            decisions = self.store.get_decisions(run.id)
            change_summary = decisions[0].rationale if decisions else "Initial run"

            # Generate result summary
            result_parts = []
            for name, value in metrics.items():
                if name in prev_metrics:
                    delta = value - prev_metrics[name]
                    pct = (delta / abs(prev_metrics[name]) * 100) if prev_metrics[name] != 0 else 0
                    sign = "+" if delta > 0 else ""
                    result_parts.append(f"{name}: {prev_metrics[name]:.4f} → {value:.4f} ({sign}{pct:.1f}%)")
                else:
                    result_parts.append(f"{name}: {value:.4f}")

            summaries.append(RunSummary(
                iteration=run.iteration_number,
                metrics=metrics,
                change_summary=change_summary,
                result_summary=", ".join(result_parts) if result_parts else "No metrics",
            ))

            prev_metrics = metrics

        return summaries


def format_run_history(summaries: list[RunSummary]) -> str:
    """Format run summaries for LLM context."""
    if not summaries:
        return "No previous runs."

    lines = []
    for s in summaries:
        lines.append(f"Run #{s.iteration}: {s.change_summary}")
        lines.append(f"  Result: {s.result_summary}")

    return "\n".join(lines)
