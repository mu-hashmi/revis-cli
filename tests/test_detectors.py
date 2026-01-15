"""Tests for guardrail detectors."""

import math
from datetime import datetime, timedelta

import pytest

from revis.analyzer.detectors import (
    detect_divergence,
    detect_nan_inf,
    detect_plateau,
    detect_timeout,
    GuardrailChecker,
)
from revis.config import GuardrailsConfig
from revis.types import EvalResult


class TestDetectNanInf:
    def test_clean_metrics(self):
        metrics = {"loss": 0.5, "accuracy": 0.95}
        result = detect_nan_inf(metrics)
        assert not result.triggered

    def test_nan_detected(self):
        metrics = {"loss": float("nan"), "accuracy": 0.95}
        result = detect_nan_inf(metrics)
        assert result.triggered
        assert "NaN" in result.message
        assert result.severity == "error"

    def test_inf_detected(self):
        metrics = {"loss": float("inf"), "accuracy": 0.95}
        result = detect_nan_inf(metrics)
        assert result.triggered
        assert "Inf" in result.message

    def test_negative_inf_detected(self):
        metrics = {"loss": float("-inf")}
        result = detect_nan_inf(metrics)
        assert result.triggered


class TestDetectDivergence:
    def test_no_divergence(self):
        result = detect_divergence(current_value=0.5, initial_value=0.4, multiplier=10.0)
        assert not result.triggered

    def test_divergence_detected(self):
        result = detect_divergence(current_value=50.0, initial_value=0.4, multiplier=10.0)
        assert result.triggered
        assert "Divergence" in result.message

    def test_zero_initial(self):
        result = detect_divergence(current_value=0.5, initial_value=0.0, multiplier=10.0)
        assert not result.triggered
        assert "zero" in result.message.lower()

    def test_custom_multiplier(self):
        result = detect_divergence(current_value=2.5, initial_value=1.0, multiplier=2.0)
        assert result.triggered  # 2.5 > 2.0 * 1.0


class TestDetectPlateau:
    def test_not_enough_history(self):
        history = [0.5, 0.4]
        result = detect_plateau(history, threshold=0.01, n_runs=3)
        assert not result.triggered
        assert "Not enough history" in result.message

    def test_no_plateau_improving(self):
        history = [0.5, 0.4, 0.3, 0.2, 0.1]
        result = detect_plateau(history, threshold=0.01, n_runs=3)
        assert not result.triggered

    def test_plateau_detected(self):
        # Last 3 runs show < 1% improvement from best before
        # best_before = 0.3, best_recent = 0.2995
        # improvement = (0.3 - 0.2995) / 0.3 = 0.17% < 1%
        history = [0.5, 0.4, 0.3, 0.2999, 0.2998, 0.2995]
        result = detect_plateau(history, threshold=0.01, n_runs=3, minimize=True)
        assert result.triggered
        assert "Plateau" in result.message

    def test_maximize_mode(self):
        history = [0.5, 0.6, 0.7, 0.701, 0.702, 0.703]
        result = detect_plateau(history, threshold=0.01, n_runs=3, minimize=False)
        assert result.triggered


class TestDetectTimeout:
    def test_within_limit(self):
        started = datetime.now() - timedelta(hours=1)
        max_duration = timedelta(hours=2)
        result = detect_timeout(started, max_duration)
        assert not result.triggered

    def test_exceeded_limit(self):
        started = datetime.now() - timedelta(hours=3)
        max_duration = timedelta(hours=2)
        result = detect_timeout(started, max_duration)
        assert result.triggered
        assert "Timeout" in result.message


class TestGuardrailChecker:
    def test_check_eval_result(self):
        config = GuardrailsConfig()
        checker = GuardrailChecker(config)

        eval_result = EvalResult(metrics={"loss": 0.5, "accuracy": 0.9})
        results = checker.check_eval_result(
            eval_result=eval_result,
            primary_metric="loss",
            initial_value=0.8,
            metric_history=[0.8, 0.7, 0.6],
            minimize=True,
        )

        # Should have nan check and divergence check at minimum
        assert len(results) >= 2
        assert all(not r.triggered for r in results)

    def test_disabled_guardrails(self):
        config = GuardrailsConfig(
            nan_detection_enabled=False,
            divergence_detection_enabled=False,
            plateau_detection_enabled=False,
        )
        checker = GuardrailChecker(config)

        eval_result = EvalResult(metrics={"loss": float("nan")})
        results = checker.check_eval_result(
            eval_result=eval_result,
            primary_metric="loss",
            initial_value=0.5,
            metric_history=[],
            minimize=True,
        )

        # No checks should run when all disabled
        assert len(results) == 0

    def test_has_critical_violation(self):
        config = GuardrailsConfig()
        checker = GuardrailChecker(config)

        eval_result = EvalResult(metrics={"loss": float("nan")})
        results = checker.check_eval_result(
            eval_result=eval_result,
            primary_metric="loss",
            initial_value=0.5,
            metric_history=[],
            minimize=True,
        )

        assert checker.has_critical_violation(results)
