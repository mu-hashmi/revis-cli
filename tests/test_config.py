"""Tests for configuration parsing."""

import tempfile
from pathlib import Path

import pytest

from revis.config import (
    RevisConfig,
    get_config_template,
    load_config,
    parse_duration,
)


class TestParseDuration:
    def test_seconds(self):
        assert parse_duration("30s") == 30
        assert parse_duration("1s") == 1

    def test_minutes(self):
        assert parse_duration("5m") == 300
        assert parse_duration("1m") == 60

    def test_hours(self):
        assert parse_duration("2h") == 7200
        assert parse_duration("24h") == 86400

    def test_days(self):
        assert parse_duration("1d") == 86400
        assert parse_duration("7d") == 604800

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration unit"):
            parse_duration("10x")

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid duration value"):
            parse_duration("abch")

    def test_empty(self):
        with pytest.raises(ValueError, match="Empty duration"):
            parse_duration("")


class TestConfigTemplate:
    def test_template_is_valid_yaml(self):
        import yaml

        template = get_config_template()
        data = yaml.safe_load(template)
        assert "executor" in data
        assert "entry" in data
        assert "metrics" in data

    def test_template_has_required_fields(self):
        import yaml

        template = get_config_template()
        data = yaml.safe_load(template)

        assert data["executor"]["type"] == "local"
        assert data["entry"]["train"] == "python train.py"
        assert data["metrics"]["primary"] == "loss"


class TestLoadConfig:
    def test_load_minimal_config(self):
        config_yaml = """
executor:
  type: local

entry:
  train: "python train.py"

metrics:
  primary: loss

context:
  include:
    - config.yaml
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            config = load_config(Path(f.name))

            assert config.executor.type == "local"
            assert config.executor.work_dir == "."  # default
            assert config.entry.train == "python train.py"
            assert config.metrics.primary == "loss"
            assert config.metrics.minimize is True  # default

    def test_load_full_config(self):
        config_yaml = """
executor:
  type: ssh
  host: gpu-server.example.com
  user: researcher
  port: 2222
  key_path: ~/.ssh/gpu_key
  work_dir: /data/experiments

entry:
  train: "python train.py --config config.yaml"
  eval: "python eval.py"

metrics:
  primary: val_loss
  minimize: true
  target: 0.05

guardrails:
  plateau_threshold: 0.005
  plateau_runs: 5
  max_run_duration: 12h
  retry_budget: 5
  divergence_multiplier: 5.0
  nan_detection_enabled: true
  divergence_detection_enabled: false

context:
  deny:
    - "*.lock"
  constraints:
    - "Learning rate between 1e-6 and 1e-2"
  history: 20
  log_tail_lines: 500
  max_agent_iterations: 15

llm:
  model: claude-opus-4-20250514
  fallback:
    - claude-sonnet-4-20250514
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()

            config = load_config(Path(f.name))

            assert config.executor.port == 2222
            assert config.executor.work_dir == "/data/experiments"
            assert config.entry.eval == "python eval.py"
            assert config.metrics.target == 0.05
            assert config.guardrails.plateau_runs == 5
            assert config.guardrails.divergence_detection_enabled is False
            assert config.context.max_agent_iterations == 15
            assert len(config.llm.fallback) == 1
            assert len(config.context.constraints) == 1
