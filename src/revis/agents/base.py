"""CodingAgent protocol and context for handoffs."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class HandoffContext:
    """Context passed to coding agents during handoff."""

    iteration_history: str
    latest_metrics: dict[str, float]
    suggestion: str
    relevant_files: list[str]
    constraints: list[str] | None = None


@dataclass
class HandoffResult:
    """Result from a coding agent handoff."""

    success: bool
    files_changed: list[str]
    error_message: str | None = None


class CodingAgent(Protocol):
    """Protocol for coding agent implementations."""

    def handoff(self, context: HandoffContext) -> HandoffResult:
        """
        Execute a handoff to the coding agent.

        Args:
            context: The handoff context with iteration history, metrics, etc.

        Returns:
            HandoffResult indicating success and what files changed
        """
        ...

    def is_available(self) -> bool:
        """Check if this agent is available on the system."""
        ...
