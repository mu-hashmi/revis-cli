"""Configuration models for Revis."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class ExecutorConfig(BaseModel):
    """Configuration for the executor."""

    type: Literal["local", "ssh"] = "local"
    host: str | None = None
    user: str | None = None
    port: int = 22
    key_path: str | None = None
    work_dir: str = "."

    @field_validator("host", "user")
    @classmethod
    def validate_ssh_fields(cls, v, info):
        return v

    def model_post_init(self, __context):
        if self.type == "ssh":
            if not self.host:
                raise ValueError("executor.host is required when type is 'ssh'")
            if not self.user:
                raise ValueError("executor.user is required when type is 'ssh'")


class EntryConfig(BaseModel):
    """Entry point configuration.

    Revis injects REVIS_OUTPUT_DIR environment variable when running training.
    Your training script should write eval.json and any artifacts there.
    """

    train: str
    eval: str | None = None  # Optional separate eval command
    env: dict[str, str] = {}  # Additional env vars to set
    env_passthrough: list[str] = []  # Env var names to pass from current environment


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    primary: str
    minimize: bool = True
    target: float | None = None


class GuardrailsConfig(BaseModel):
    """Guardrails configuration."""

    plateau_threshold: float = 0.01
    plateau_runs: int = 3
    max_run_duration: str = "24h"
    retry_budget: int = 3
    divergence_multiplier: float = 10.0

    nan_detection_enabled: bool = True
    divergence_detection_enabled: bool = True
    plateau_detection_enabled: bool = True
    timeout_enabled: bool = True

    @field_validator("max_run_duration")
    @classmethod
    def validate_duration(cls, v: str) -> str:
        parse_duration(v)
        return v


class ContextConfig(BaseModel):
    """Context configuration for LLM agent."""

    deny: list[str] = []  # Patterns to block from reading AND writing
    constraints: list[str] = []  # Free-text constraints for the LLM
    history: int = 10
    log_tail_lines: int = 200
    max_agent_iterations: int = 20


class LLMConfig(BaseModel):
    """LLM configuration."""

    model: str = "claude-sonnet-4-20250514"
    api_base: str | None = None
    fallback: list[str] = []


class ArtifactsConfig(BaseModel):
    """Artifact storage configuration."""

    path: str = ".revis/artifacts"


class AutoMergeConfig(BaseModel):
    """Auto-merge configuration."""

    enabled: bool = False
    require_target_achieved: bool = True
    min_improvement_percent: float | None = None


class GitHubConfig(BaseModel):
    """GitHub configuration."""

    auto_merge: AutoMergeConfig = AutoMergeConfig()


class RevisConfig(BaseModel):
    """Main Revis configuration."""

    executor: ExecutorConfig
    entry: EntryConfig
    metrics: MetricsConfig
    guardrails: GuardrailsConfig = GuardrailsConfig()
    context: ContextConfig = ContextConfig()
    llm: LLMConfig = LLMConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    github: GitHubConfig = GitHubConfig()


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds. Supports: 30s, 5m, 2h, 1d."""
    duration_str = duration_str.strip().lower()
    if not duration_str:
        raise ValueError("Empty duration string")

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    unit = duration_str[-1]

    if unit not in multipliers:
        raise ValueError(f"Invalid duration unit: {unit}. Use s, m, h, or d.")

    try:
        value = int(duration_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid duration value: {duration_str[:-1]}")

    return value * multipliers[unit]


def load_config(path: Path) -> RevisConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return RevisConfig(**data)


def get_config_template() -> str:
    """Get the default configuration template."""
    return """# Revis Configuration

executor:
  type: local  # 'local' (run here) or 'ssh' (sync to remote)
  work_dir: .  # Working directory for training

  # SSH settings (only needed if type: ssh)
  # host: your-gpu-server.example.com
  # user: your-username
  # port: 22
  # key_path: ~/.ssh/id_rsa

entry:
  train: "python train.py"
  # eval: "python eval.py"  # Optional separate eval command
  #
  # Environment: Common ML keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, WANDB_API_KEY,
  # HF_TOKEN, etc.) are auto-passed from your shell. A .env file in the project
  # root is also loaded if it exists. For custom vars, use env_passthrough.
  #
  # Revis sets REVIS_OUTPUT_DIR env var when running your script.
  # Write eval.json and any artifacts there:
  #   output_dir = os.environ.get("REVIS_OUTPUT_DIR", "outputs/")
  #   json.dump({"metrics": {...}}, open(f"{output_dir}/eval.json", "w"))

metrics:
  primary: loss  # Name of the primary metric in eval.json
  minimize: true
  # target: 0.1  # Optional early stopping target

guardrails:
  plateau_threshold: 0.01
  plateau_runs: 3
  max_run_duration: 24h
  retry_budget: 3
  divergence_multiplier: 10.0
  nan_detection_enabled: true
  divergence_detection_enabled: true
  plateau_detection_enabled: true
  timeout_enabled: true

# Agent can access entire repo by default.
# Use deny to block access to files,
# and constraints to set boundaries for changes.
context:
  deny:
    - ".git/**"
    - "**/__pycache__/**"
    - "*.pt"
    - "*.ckpt"
    - "*.safetensors"
    - "wandb/**"
    - ".revis/**"
    - "node_modules/**"
    - "revis.yaml"  # Agent cannot modify revis config; use NEXT_COMMAND to change CLI args
  constraints:
    # - "Learning rate must be between 1e-6 and 1e-2"
    # - "Batch size must be a power of 2"
  history: 10
  log_tail_lines: 200
  max_agent_iterations: 20

llm:
  # Uncomment one of the models below (requires corresponding API key in env)
  # See https://docs.litellm.ai/docs/providers for full list of supported models
  #
  # Anthropic (ANTHROPIC_API_KEY)
  # model: claude-sonnet-4-20250514
  # model: claude-opus-4-5-20251101
  # model: claude-haiku-4-5-20251001
  #
  # OpenAI (OPENAI_API_KEY)
  # model: gpt-5.2-2025-12-11
  # model: gpt-5-mini-2025-08-07
  # model: gpt-5-nano-2025-08-07
  #
  # Google (GEMINI_API_KEY)
  # model: gemini/gemini-3-pro-preview
  # model: gemini/gemini-3-flash-preview
  # model: gemini/gemini-2.5-flash
  #
  model: claude-sonnet-4-5-20250929  # default
  # api_base: null  # Optional custom API endpoint
  fallback: []  # Fallback models on API errors

artifacts:
  path: .revis/artifacts

github:
  auto_merge:
    enabled: false
    require_target_achieved: true
    # min_improvement_percent: 5.0
"""
