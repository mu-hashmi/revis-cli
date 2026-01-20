"""Aider agent implementation."""

import json
import logging
import shutil
import subprocess

from revis.agents.base import HandoffContext, HandoffResult

logger = logging.getLogger(__name__)


class AiderAgent:
    """Aider agent for code changes."""

    def is_available(self) -> bool:
        return shutil.which("aider") is not None

    def handoff(self, context: HandoffContext) -> HandoffResult:
        """Execute handoff to Aider."""
        if not self.is_available():
            return HandoffResult(
                success=False,
                files_changed=[],
                error_message="aider CLI not found",
            )

        prompt = self._build_prompt(context)

        try:
            cmd = ["aider", "--yes", "--no-git", "--message", prompt]
            cmd.extend(context.relevant_files)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                return HandoffResult(
                    success=False,
                    files_changed=[],
                    error_message=f"aider exited with code {result.returncode}",
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
                error_message="aider timed out after 10 minutes",
            )
        except Exception as e:
            return HandoffResult(
                success=False,
                files_changed=[],
                error_message=str(e),
            )

    def _build_prompt(self, context: HandoffContext) -> str:
        metrics_str = json.dumps(context.latest_metrics, indent=2)

        constraints_section = ""
        if context.constraints:
            constraints_str = ", ".join(context.constraints)
            constraints_section = f"\n\nConstraints: {constraints_str}"

        return f"""ML training optimization requested by Revis.

Iteration history:
{context.iteration_history}

Current metrics:
{metrics_str}

Requested change:
{context.suggestion}
{constraints_section}

Make minimal, targeted changes. Do not refactor unrelated code."""

    def _detect_changed_files(self) -> list[str]:
        """Detect which files were changed."""
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
