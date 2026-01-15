"""Tests for evaluator harness."""

import json

import pytest

from revis.evaluator.harness import validate_eval_schema
from revis.types import EvalResult


class TestValidateEvalSchema:
    def test_valid_minimal(self):
        data = {"metrics": {"loss": 0.5}}
        errors = validate_eval_schema(data)
        assert errors == []

    def test_valid_full(self):
        data = {
            "metrics": {"loss": 0.5, "accuracy": 0.95},
            "slices": {
                "by_category": {
                    "cat_a": {"loss": 0.4},
                    "cat_b": {"loss": 0.6},
                }
            },
            "plots": ["loss_curve.png", "confusion_matrix.png"],
        }
        errors = validate_eval_schema(data)
        assert errors == []

    def test_missing_metrics(self):
        data = {"slices": {}}
        errors = validate_eval_schema(data)
        assert "Missing required field: 'metrics'" in errors

    def test_metrics_not_dict(self):
        data = {"metrics": [1, 2, 3]}
        errors = validate_eval_schema(data)
        assert "'metrics' must be a dictionary" in errors

    def test_non_numeric_metric(self):
        data = {"metrics": {"loss": "not a number"}}
        errors = validate_eval_schema(data)
        assert "Metric 'loss' must be numeric" in errors

    def test_slices_not_dict(self):
        data = {"metrics": {"loss": 0.5}, "slices": "invalid"}
        errors = validate_eval_schema(data)
        assert "'slices' must be a dictionary" in errors

    def test_plots_not_list(self):
        data = {"metrics": {"loss": 0.5}, "plots": "single.png"}
        errors = validate_eval_schema(data)
        assert "'plots' must be a list" in errors

    def test_plots_invalid_items(self):
        data = {"metrics": {"loss": 0.5}, "plots": [1, 2, 3]}
        errors = validate_eval_schema(data)
        assert "All items in 'plots' must be strings" in errors


class TestEvalResult:
    def test_from_dict(self):
        result = EvalResult(
            metrics={"loss": 0.5, "ade": 3.2},
            slices={},
            plots=["plot.png"],
        )
        assert result.metrics["loss"] == 0.5
        assert result.metrics["ade"] == 3.2
        assert len(result.plots) == 1

    def test_empty_optionals(self):
        result = EvalResult(metrics={"loss": 0.5})
        assert result.slices == {}
        assert result.plots == []
