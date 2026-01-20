"""GitHub PR and branch management."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from github import Github
from github.GithubException import GithubException

from revis.types import Run, Session, TerminationReason


@dataclass
class GitConfig:
    """Git configuration."""

    repo_path: Path
    remote: str = "origin"


class GitManager:
    """Local git operations."""

    def __init__(self, config: GitConfig):
        self.config = config

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run git command."""
        cmd = ["git", "-C", str(self.config.repo_path)] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def get_current_branch(self) -> str:
        """Get current branch name."""
        result = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def get_head_sha(self) -> str:
        """Get HEAD commit SHA."""
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()

    def create_branch(self, branch_name: str, base_sha: str | None = None) -> None:
        """Create and checkout a new branch."""
        if base_sha:
            self._run("checkout", "-b", branch_name, base_sha)
        else:
            self._run("checkout", "-b", branch_name)

    def checkout(self, branch_name: str) -> None:
        """Checkout existing branch."""
        self._run("checkout", branch_name)

    def branch_exists(self, branch_name: str) -> bool:
        """Check if branch exists locally or remotely."""
        result = self._run("branch", "-a", "--list", f"*{branch_name}", check=False)
        return branch_name in result.stdout

    def commit(self, message: str, files: list[str] | None = None) -> str:
        """Stage files and commit. Returns commit SHA."""
        if files:
            for f in files:
                self._run("add", f)
        else:
            self._run("add", "-A")

        self._run("commit", "-m", message)
        return self.get_head_sha()

    def push(self, branch_name: str, force: bool = False) -> None:
        """Push branch to remote."""
        args = ["push", "-u", self.config.remote, branch_name]
        if force:
            args.insert(1, "--force")
        self._run(*args)

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        result = self._run("status", "--porcelain")
        return bool(result.stdout.strip())

    def get_remote_url(self) -> str:
        """Get remote URL."""
        result = self._run("remote", "get-url", self.config.remote)
        return result.stdout.strip()

    def get_repo_info(self) -> tuple[str, str]:
        """Get owner and repo name from remote URL."""
        url = self.get_remote_url()

        # Handle SSH format: git@github.com:owner/repo.git
        if url.startswith("git@"):
            path = url.split(":")[-1]
        # Handle HTTPS format: https://github.com/owner/repo.git
        else:
            path = url.split("github.com/")[-1]

        path = path.rstrip(".git")
        parts = path.split("/")
        return parts[0], parts[1]


class GitHubManager:
    """GitHub API operations."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not set")
        self.gh = Github(self.token)

    def create_pr(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> str:
        """Create a pull request. Returns PR URL."""
        repository = self.gh.get_repo(f"{owner}/{repo}")

        try:
            pr = repository.create_pull(
                title=title,
                body=body,
                head=head,
                base=base,
            )
            return pr.html_url
        except GithubException as e:
            if "already exists" in str(e):
                # PR already exists, find it
                prs = repository.get_pulls(head=f"{owner}:{head}", state="open")
                for pr in prs:
                    return pr.html_url
            raise

    def merge_pr(self, owner: str, repo: str, pr_url: str) -> bool:
        """Merge a PR. Returns success."""
        repository = self.gh.get_repo(f"{owner}/{repo}")

        # Extract PR number from URL
        pr_number = int(pr_url.rstrip("/").split("/")[-1])
        pr = repository.get_pull(pr_number)

        try:
            pr.merge()
            return True
        except GithubException:
            return False


def format_pr_body(
    session: Session,
    runs: list[Run],
    run_metrics: dict[str, dict[str, float]],
    decisions: dict[str, str],
    primary_metric: str,
    baseline_value: float | None,
    fallback_used: bool = False,
) -> str:
    """Format PR body with session summary."""
    # Determine result status
    result_map = {
        TerminationReason.TARGET_ACHIEVED: "SUCCESS",
        TerminationReason.BUDGET_EXHAUSTED: "PROGRESS",
        TerminationReason.PLATEAU: "PLATEAU",
        TerminationReason.RETRY_EXHAUSTION: "FAILURE",
        TerminationReason.LLM_ESCALATION: "ESCALATED",
        TerminationReason.USER_STOP: "STOPPED",
    }
    result = result_map.get(session.termination_reason, "UNKNOWN")

    # Calculate final metrics
    final_metrics = {}
    if runs:
        last_run_id = runs[-1].id
        if last_run_id in run_metrics:
            final_metrics = run_metrics[last_run_id]

    final_value = final_metrics.get(primary_metric)

    # Calculate improvement
    improvement_str = "N/A"
    if baseline_value is not None and final_value is not None:
        delta = final_value - baseline_value
        pct = (delta / abs(baseline_value) * 100) if baseline_value != 0 else 0
        sign = "+" if delta > 0 else ""
        improvement_str = f"{sign}{pct:.1f}%"

    # Format duration
    if session.ended_at and session.started_at:
        duration = session.ended_at - session.started_at
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        duration_str = f"{hours}h {minutes}m"
    else:
        duration_str = "N/A"

    # Build metrics table
    metrics_table = f"""| Metric | Baseline | Final | Δ |
|--------|----------|-------|---|
| {primary_metric} | {baseline_value or "N/A"} | {final_value or "N/A"} | {improvement_str} |"""

    # Build key iterations table
    key_iterations = []
    for run in runs:
        run_id = run.id
        metrics = run_metrics.get(run_id, {})
        metric_val = metrics.get(primary_metric, "N/A")
        rationale = decisions.get(run_id, "Initial" if run.iteration_number == 1 else "N/A")

        # Truncate rationale
        if len(rationale) > 50:
            rationale = rationale[:47] + "..."

        sha = run.git_sha[:7] if run.git_sha else "N/A"
        key_iterations.append(f"| {run.iteration_number} | {sha} | {metric_val} | {rationale} |")

    iterations_table = (
        """| # | Commit | """
        + primary_metric
        + """ | Rationale |
|---|--------|-----|-----------|
"""
        + "\n".join(key_iterations)
    )

    # Build body
    body = f"""## Session Summary

**Result**: {result}
**Branch**: `{session.branch}`
**Termination**: {session.termination_reason.value if session.termination_reason else "N/A"}

### Metrics

{metrics_table}

### Session Overview

- **Iterations**: {session.iteration_count}
- **Runtime**: {duration_str}
- **LLM Cost**: ${session.llm_cost_usd:.2f}
- **Retries Used**: {3 - session.retry_budget}/3

### Iterations

{iterations_table}

---
🤖 Generated by [Revis](https://github.com/revis-ai/revis)"""

    if fallback_used:
        body += "\n\n⚠️ *Fallback LLM model was used during this session due to API issues.*"

    return body
