"""Claude Code agent implementation."""

import json
import logging
import shutil
import subprocess

from revis.agents.base import HandoffContext, HandoffResult

logger = logging.getLogger(__name__)


class ClaudeCodeAgent:
    """Claude Code (claude CLI) agent for code changes."""

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def handoff(self, context: HandoffContext) -> HandoffResult:
        """Execute handoff to Claude Code."""
        if not self.is_available():
            return HandoffResult(
                success=False,
                files_changed=[],
                error_message="claude CLI not found",
            )

        prompt = self._build_prompt(context)

        try:
            result = subprocess.run(
                ["claude", "--print"],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=600,
            )

            if result.returncode != 0:
                return HandoffResult(
                    success=False,
                    files_changed=[],
                    error_message=f"claude exited with code {result.returncode}",
                )

            changed_files = self._detect_changed_files()
            return HandoffResult(
                success=True,
                files_changed=changed_files,
            )

        except subprocess.TimeoutExpired:
            return HandoffResult(
                success=False,
                files_changed=[],
                error_message="claude timed out after 10 minutes",
            )
        except Exception as e:
            return HandoffResult(
                success=False,
                files_changed=[],
                error_message=str(e),
            )

    def _build_prompt(self, context: HandoffContext) -> str:
        metrics_str = json.dumps(context.latest_metrics, indent=2)
        files_str = "\n".join(f"  - {f}" for f in context.relevant_files)

        constraints_section = ""
        if context.constraints:
            constraints_str = "\n".join(f"  - {c}" for c in context.constraints)
            constraints_section = f"""
Constraints:
{constraints_str}
"""

        return f"""This is an ML training codebase. Revis (an autonomous ML iteration tool) has
determined that code changes are needed to improve training metrics.

## Iteration History
{context.iteration_history}

## Current Metrics
```json
{metrics_str}
```

## Requested Change
{context.suggestion}

## Relevant Files
{files_str}
{constraints_section}
## Instructions
Make minimal, targeted changes to implement the requested change. Do not refactor
unrelated code. Focus on the specific change requested.

After making changes, verify they are syntactically correct.
"""

    def _detect_changed_files(self) -> list[str]:
        """Detect which files were changed by running git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return []
        except Exception:
            return []
