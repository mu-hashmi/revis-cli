"""RunStore protocol for session and run persistence."""

from typing import Protocol

from revis.types import Budget, Decision, Run, Session, TerminationReason


class RunStore(Protocol):
    """Protocol for session and run storage."""

    # Session management
    def create_session(
        self,
        name: str,
        branch: str,
        base_sha: str,
        budget: Budget,
        baseline_run_id: str | None = None,
    ) -> str:
        """Create a new session. Returns session ID."""
        ...

    def end_session(
        self,
        session_id: str,
        reason: TerminationReason,
        pr_url: str | None,
    ) -> None:
        """End a session with the given reason."""
        ...

    def get_session(self, session_id: str) -> Session:
        """Get session by ID."""
        ...

    def get_running_session(self) -> Session | None:
        """Get the currently running session, if any."""
        ...

    def get_session_by_name(self, name: str) -> Session | None:
        """Get session by name."""
        ...

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """List all sessions, optionally filtered by status."""
        ...

    def delete_session(self, session_id: str, force: bool = False) -> bool:
        """Delete a session and all related data. Returns True if deleted.

        Args:
            session_id: Session ID to delete
            force: If True, allow deleting running sessions
        """
        ...

    def mark_session_exported(self, session_id: str, pr_url: str | None) -> None:
        """Mark session as exported."""
        ...

    def session_name_exists(self, name: str) -> bool:
        """Check if a session name already exists."""
        ...

    def update_session_budget(self, session_id: str, budget_used: int) -> None:
        """Update budget usage for a session."""
        ...

    def update_session_cost(self, session_id: str, cost_usd: float) -> None:
        """Update LLM cost for a session."""
        ...

    # Run management
    def create_run(
        self,
        session_id: str,
        config_json: str,
        iteration: int,
    ) -> str:
        """Create a new run. Returns run ID."""
        ...

    def set_run_status(self, run_id: str, status: str) -> None:
        """Update run status."""
        ...

    def set_run_commit(self, run_id: str, sha: str) -> None:
        """Set the commit SHA for a run."""
        ...

    def log_params(self, run_id: str, params: dict) -> None:
        """Log parameters for a run."""
        ...

    def log_metrics(self, run_id: str, metrics: dict, step: int | None = None) -> None:
        """Log metrics for a run."""
        ...

    def log_artifact(self, run_id: str, kind: str, path: str) -> str:
        """Log an artifact. Returns artifact ID."""
        ...

    def query_runs(
        self,
        session_id: str | None = None,
        branch: str | None = None,
        limit: int = 10,
    ) -> list[Run]:
        """Query runs with optional filters."""
        ...

    def get_run(self, run_id: str) -> Run:
        """Get run by ID."""
        ...

    def get_baseline_run(self, session_id: str) -> Run | None:
        """Get the baseline run for a session."""
        ...

    # Decision tracking
    def attach_decision(self, run_id: str, decision: Decision) -> str:
        """Attach a decision to a run. Returns decision ID."""
        ...

    def update_decision_commit(self, decision_id: str, commit_sha: str) -> None:
        """Update the commit SHA for a decision."""
        ...
