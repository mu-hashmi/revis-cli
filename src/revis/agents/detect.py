"""Coding agent detection and factory."""

import shutil

from revis.agents.base import CodingAgent
from revis.agents.claude_code import ClaudeCodeAgent


def detect_coding_agent() -> str | None:
    """Detect available coding agents."""
    if shutil.which("claude"):
        return "claude-code"
    return None


def get_coding_agent(agent_type: str) -> CodingAgent | None:
    """
    Get a coding agent instance by type.

    Args:
        agent_type: One of "auto", "claude-code", "none"

    Returns:
        CodingAgent instance or None if type is "none" or not available
    """
    if agent_type == "none":
        return None

    if agent_type == "auto":
        detected = detect_coding_agent()
        if detected is None:
            return None
        agent_type = detected

    if agent_type == "claude-code":
        agent = ClaudeCodeAgent()
        if agent.is_available():
            return agent
        return None

    return None
