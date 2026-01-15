"""Guardrail detectors for catching training issues."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from revis.config import GuardrailsConfig
from revis.types import EvalResult, Run


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    triggered: bool
    guardrail: str
    message: str
    severity: str = "warning"  # warning, error


def detect_nan_inf(metrics: dict[str, float]) -> GuardrailResult:
    """Detect NaN or Inf values in metrics."""
    for name, value in metrics.items():
        if math.isnan(value):
            return GuardrailResult(
                triggered=True,
                guardrail="nan_detection",
                message=f"NaN detected in metric '{name}'",
                severity="error",
            )
        if math.isinf(value):
            return GuardrailResult(
                triggered=True,
                guardrail="nan_detection",
                message=f"Inf detected in metric '{name}'",
                severity="error",
            )

    return GuardrailResult(
        triggered=False,
        guardrail="nan_detection",
        message="No NaN/Inf values detected",
    )


def detect_divergence(
    current_value: float,
    initial_value: float,
    multiplier: float = 10.0,
) -> GuardrailResult:
    """Detect if loss has diverged (exceeded multiplier × initial)."""
    if initial_value == 0:
        return GuardrailResult(
            triggered=False,
            guardrail="divergence_detection",
            message="Initial value is zero, cannot detect divergence",
        )

    threshold = abs(initial_value) * multiplier

    if abs(current_value) > threshold:
        return GuardrailResult(
            triggered=True,
            guardrail="divergence_detection",
            message=f"Divergence detected: {current_value:.4f} > {threshold:.4f} ({multiplier}× initial)",
            severity="error",
        )

    return GuardrailResult(
        triggered=False,
        guardrail="divergence_detection",
        message=f"No divergence: {current_value:.4f} within {multiplier}× initial",
    )


def detect_plateau(
    metric_history: list[float],
    threshold: float = 0.01,
    n_runs: int = 3,
    minimize: bool = True,
) -> GuardrailResult:
    """Detect if metric has plateaued (no improvement for N runs)."""
    # Need n_runs + 1 to have something to compare against
    if len(metric_history) <= n_runs:
        return GuardrailResult(
            triggered=False,
            guardrail="plateau_detection",
            message=f"Not enough history ({len(metric_history)} <= {n_runs} runs)",
        )

    recent = metric_history[-n_runs:]
    best_before = min(metric_history[:-n_runs]) if minimize else max(metric_history[:-n_runs])
    best_recent = min(recent) if minimize else max(recent)

    if minimize:
        improvement = (best_before - best_recent) / abs(best_before) if best_before != 0 else 0
    else:
        improvement = (best_recent - best_before) / abs(best_before) if best_before != 0 else 0

    if improvement < threshold:
        return GuardrailResult(
            triggered=True,
            guardrail="plateau_detection",
            message=f"Plateau detected: {improvement:.2%} improvement over last {n_runs} runs (threshold: {threshold:.2%})",
            severity="warning",
        )

    return GuardrailResult(
        triggered=False,
        guardrail="plateau_detection",
        message=f"No plateau: {improvement:.2%} improvement",
    )


def detect_timeout(
    started_at: datetime,
    max_duration: timedelta,
) -> GuardrailResult:
    """Detect if run has exceeded maximum duration."""
    elapsed = datetime.now() - started_at

    if elapsed > max_duration:
        return GuardrailResult(
            triggered=True,
            guardrail="timeout_detection",
            message=f"Timeout: {elapsed} > {max_duration}",
            severity="error",
        )

    return GuardrailResult(
        triggered=False,
        guardrail="timeout_detection",
        message=f"Within time limit: {elapsed} / {max_duration}",
    )


class GuardrailChecker:
    """Checker that runs all enabled guardrails."""

    def __init__(self, config: GuardrailsConfig):
        self.config = config

    def check_eval_result(
        self,
        eval_result: EvalResult,
        primary_metric: str,
        initial_value: float | None,
        metric_history: list[float],
        minimize: bool = True,
    ) -> list[GuardrailResult]:
        """Run all guardrail checks on an eval result."""
        results = []

        # NaN/Inf detection
        if self.config.nan_detection_enabled:
            result = detect_nan_inf(eval_result.metrics)
            results.append(result)

        # Divergence detection
        if self.config.divergence_detection_enabled and initial_value is not None:
            current_value = eval_result.metrics.get(primary_metric)
            if current_value is not None:
                result = detect_divergence(
                    current_value,
                    initial_value,
                    self.config.divergence_multiplier,
                )
                results.append(result)

        # Plateau detection
        if self.config.plateau_detection_enabled:
            current_value = eval_result.metrics.get(primary_metric)
            if current_value is not None:
                full_history = metric_history + [current_value]
                result = detect_plateau(
                    full_history,
                    self.config.plateau_threshold,
                    self.config.plateau_runs,
                    minimize,
                )
                results.append(result)

        return results

    def check_run_duration(self, started_at: datetime, max_duration: timedelta) -> GuardrailResult:
        """Check if run has timed out."""
        if not self.config.timeout_enabled:
            return GuardrailResult(
                triggered=False,
                guardrail="timeout_detection",
                message="Timeout detection disabled",
            )
        return detect_timeout(started_at, max_duration)

    def has_critical_violation(self, results: list[GuardrailResult]) -> bool:
        """Check if any guardrail triggered with error severity."""
        return any(r.triggered and r.severity == "error" for r in results)

    def get_violations(self, results: list[GuardrailResult]) -> list[GuardrailResult]:
        """Get all triggered guardrails."""
        return [r for r in results if r.triggered]
