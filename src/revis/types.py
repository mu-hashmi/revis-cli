"""Core type definitions for Revis."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class TerminationReason(str, Enum):
    """Reasons for session termination."""

    TARGET_ACHIEVED = "target_achieved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLATEAU = "plateau"
    RETRY_EXHAUSTION = "retry_exhaustion"
    LLM_ESCALATION = "llm_escalation"
    USER_STOP = "user_stop"
    ERROR = "error"


class Budget(BaseModel):
    """Budget tracking for a session."""

    type: Literal["time", "runs"]
    value: int  # seconds for time, count for runs
    used: int = 0

    def exhausted(self) -> bool:
        return self.used >= self.value

    def remaining(self) -> int:
        return max(0, self.value - self.used)


class Session(BaseModel):
    """A Revis session."""

    id: str
    name: str
    branch: str
    base_sha: str
    baseline_run_id: str | None = None
    status: Literal["running", "completed", "failed", "stopped"]
    termination_reason: TerminationReason | None = None
    started_at: datetime
    ended_at: datetime | None = None
    budget: Budget
    iteration_count: int = 0
    pr_url: str | None = None
    llm_cost_usd: float = 0.0
    retry_budget: int = 3
    exported_at: datetime | None = None


class Run(BaseModel):
    """A single training run (iteration)."""

    id: str
    session_id: str
    iteration_number: int
    config_json: str
    git_sha: str | None = None
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None

    # Change tracking
    change_type: Literal["config", "cli_args", "code_handoff", "initial"] | None = None
    change_description: str | None = None
    change_diff: str | None = None

    # Reasoning
    hypothesis: str | None = None

    # Results
    metrics_json: str | None = None
    outcome: Literal["improved", "regressed", "plateau", "failed"] | None = None
    analysis: str | None = None


class Metric(BaseModel):
    """A logged metric."""

    name: str
    value: float
    step: int | None = None
    logged_at: datetime


class Artifact(BaseModel):
    """A logged artifact."""

    id: str
    run_id: str
    kind: str
    path: str
    size_bytes: int | None = None
    uploaded_at: datetime


class Decision(BaseModel):
    """A decision made by the LLM."""

    action_type: Literal["config", "cli_args", "code_handoff", "code_patch", "escalate"]
    rationale: str
    commit_sha: str | None = None


class Suggestion(BaseModel):
    """A suggestion from the LLM agent."""

    id: int | None = None
    session_id: str
    run_id: str | None = None
    created_at: datetime | None = None
    suggestion_type: Literal["config", "cli_args", "code", "architecture", "data"]
    content: str
    status: Literal["pending", "accepted", "rejected", "handed_off"] = "pending"
    handed_off_to: str | None = None


class EvalResult(BaseModel):
    """Result of an evaluation."""

    metrics: dict[str, float]
    slices: dict[str, dict[str, dict[str, float]]] = {}
    plots: list[str] = []


class Analysis(BaseModel):
    """Analysis of a run."""

    plateau_detected: bool = False
    metric_delta: float | None = None
    guardrail_violations: list[str] = []
