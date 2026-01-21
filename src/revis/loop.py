"""Main orchestration loop for Revis."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from rich.console import Console

from revis.agents.base import HandoffContext
from revis.agents.detect import get_coding_agent
from revis.analyzer.compare import RunAnalyzer
from revis.analyzer.detectors import GuardrailChecker
from revis.config import RevisConfig, parse_duration
from revis.executor.base import Executor
from revis.executor.local import LocalConfig, LocalExecutor
from revis.executor.ssh import SSHConfig, SSHExecutor
from revis.github.pr import GitConfig, GitManager
from revis.llm.agent import run_agent
from revis.llm.client import LLMClient
from revis.llm.prompts import SYSTEM_PROMPT, build_iteration_context
from revis.llm.tools import ToolExecutor
from revis.llm.tracer import AgentTracer
from revis.metrics.eval_json import EvalJsonCollector
from revis.store.sqlite import SQLiteRunStore
from revis.types import Budget, Decision, EvalResult, Session, TerminationReason

logger = logging.getLogger(__name__)

REVIS_DIR = Path(".revis")
STOP_SIGNAL_FILE = REVIS_DIR / "stop_signal"

# Common ML API keys that are auto-passed if set in environment
COMMON_ML_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
]


def collect_training_env(config: RevisConfig, project_root: Path) -> dict[str, str]:
    """Collect environment variables for training.

    Order of precedence (later wins):
    1. Auto-pass common ML keys from current env
    2. Load .env if it exists in project root
    3. Explicit env from config
    4. Explicit env_passthrough from config
    """
    env = {}

    # 1. Auto-pass common ML keys if set
    for key in COMMON_ML_ENV_VARS:
        if key in os.environ:
            env[key] = os.environ[key]

    # 2. Load .env if exists
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        try:
            for line in dotenv_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key:
                        env[key] = value
        except Exception as e:
            logger.warning(f"Failed to load .env: {e}")

    # 3. Add explicit env vars from config
    env.update(config.entry.env)

    # 4. Add passthrough vars from current environment
    for var_name in config.entry.env_passthrough:
        if var_name in os.environ:
            env[var_name] = os.environ[var_name]

    return env


class RevisLoop:
    """Main Revis orchestration loop."""

    def __init__(
        self,
        config: RevisConfig,
        store: SQLiteRunStore,
        repo_path: Path,
        console: Console,
    ):
        self.config = config
        self.store = store
        self.repo_path = repo_path
        self.console = console

        self._active_process_id: str | None = None

        # Initialize executor based on type
        if config.executor.type == "local":
            self.executor: Executor = LocalExecutor(
                LocalConfig(
                    work_dir=config.executor.work_dir,
                )
            )
        else:
            self.executor = SSHExecutor(
                SSHConfig(
                    host=config.executor.host,
                    user=config.executor.user,
                    port=config.executor.port,
                    key_path=config.executor.key_path,
                    work_dir=config.executor.work_dir,
                )
            )

        self.llm = LLMClient(config.llm)
        self.git = GitManager(GitConfig(repo_path=repo_path))

        # Create metrics collector based on config
        self.metrics_collector = self._create_metrics_collector()

        self.analyzer = RunAnalyzer(
            store=store,
            primary_metric=config.metrics.primary,
            minimize=config.metrics.minimize,
        )
        self.guardrails = GuardrailChecker(config.guardrails)
        self.tool_executor = ToolExecutor(
            repo_root=repo_path,
            deny_patterns=config.context.deny,
        )

        # Initialize coding agent if configured
        self.coding_agent = get_coding_agent(config.coding_agent.type)

    def _create_tracer(self, run_id: str) -> AgentTracer:
        """Create an AgentTracer for the given run."""
        return AgentTracer(console=self.console, backend=self.store, run_id=run_id)

    def _create_metrics_collector(self):
        """Create a metrics collector based on config."""
        if self.config.metrics.source == "wandb":
            try:
                from revis.metrics.wandb import WandbCollector

                return WandbCollector(
                    project=self.config.metrics.project,
                    entity=self.config.metrics.entity,
                )
            except ImportError:
                logger.warning("W&B not installed, falling back to eval.json")
                return EvalJsonCollector(self.executor)
        return EvalJsonCollector(self.executor)

    def _cleanup_active_process(self) -> None:
        """Kill any active training process."""
        if self._active_process_id is not None:
            logger.info(f"Killing active training process: {self._active_process_id}")
            try:
                self.executor.kill(self._active_process_id)
            except Exception as e:
                logger.warning(f"Failed to kill process: {e}")
            self._active_process_id = None

    def run(
        self,
        name: str,
        budget: Budget,
        baseline_run_id: str | None = None,
    ) -> Session:
        """Run the main loop."""
        # Clear any stale stop signal
        if STOP_SIGNAL_FILE.exists():
            STOP_SIGNAL_FILE.unlink()

        # Create session with user-provided name
        base_sha = self.git.get_head_sha()
        base_branch = self.git.get_current_branch()
        branch_name = f"revis/{name}"

        session_id = self.store.create_session(
            name=name,
            branch=branch_name,
            base_sha=base_sha,
            budget=budget,
            baseline_run_id=baseline_run_id,
        )

        session = self.store.get_session(session_id)
        logger.info(f"Started session '{name}' (ID: {session_id}) on branch {session.branch}")

        # Create working branch
        if not self.git.branch_exists(session.branch):
            self.git.create_branch(session.branch)
        else:
            self.git.checkout(session.branch)

        start_time = time.time()

        try:
            return self._run_loop(session, budget, start_time, base_branch)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return self._terminate(session, TerminationReason.USER_STOP, base_branch)
        except Exception as e:
            logger.error(f"Loop failed with error: {e}")
            self._terminate(session, TerminationReason.ERROR, base_branch)
            raise
        finally:
            self._cleanup_active_process()
            self.executor.close()

    def _run_loop(
        self,
        session: Session,
        budget: Budget,
        start_time: float,
        base_branch: str,
    ) -> Session:
        """Inner loop logic."""
        iteration = session.iteration_count
        # Track current training command (can be overridden by agent via NEXT_COMMAND)
        current_train_cmd = self.config.entry.train

        while True:
            # Check stop signal
            if STOP_SIGNAL_FILE.exists():
                logger.info("Stop signal received")
                STOP_SIGNAL_FILE.unlink()
                return self._terminate(session, TerminationReason.USER_STOP, base_branch)

            # Check budget
            if budget.type == "time":
                elapsed = int(time.time() - start_time)
                self.store.update_session_budget(session.id, elapsed)
                if elapsed >= budget.value:
                    logger.info("Time budget exhausted")
                    return self._terminate(session, TerminationReason.BUDGET_EXHAUSTED, base_branch)
            elif budget.type == "runs":
                if iteration >= budget.value:
                    logger.info("Run budget exhausted")
                    return self._terminate(session, TerminationReason.BUDGET_EXHAUSTED, base_branch)

            iteration += 1
            self.store.increment_iteration(session.id)
            logger.info(f"Starting iteration {iteration}")

            # Sync code to remote
            logger.info("Syncing code to remote...")
            self.executor.sync_code(self.repo_path, self.config.executor.work_dir)

            # Create run record
            run_id = self.store.create_run(
                session_id=session.id,
                config_json=json.dumps({"iteration": iteration}),
                iteration=iteration,
            )

            # Set up run output directory
            run_output_dir = f".revis/runs/{run_id}"

            # Launch training with Revis environment variables
            session_name = f"revis-{session.id}"
            train_cmd = current_train_cmd

            # Build environment variables for training
            run_env = collect_training_env(self.config, self.repo_path)

            # Add Revis-specific env vars (these take precedence)
            run_env.update(
                {
                    "REVIS_OUTPUT_DIR": run_output_dir,
                    "REVIS_RUN_ID": run_id,
                    "REVIS_SESSION_ID": session.id,
                }
            )

            # Redirect to log file, pipefail preserves training exit code
            log_path = f"{run_output_dir}/train.log"
            train_cmd = (
                f"set -o pipefail; mkdir -p {run_output_dir} && {train_cmd} 2>&1 | tee {log_path}"
            )

            logger.info(f"Launching training: {train_cmd}")
            logger.info(f"Run output dir: {run_output_dir}")

            try:
                self.executor.launch(train_cmd, run_env, session_name)
                self._active_process_id = session_name

                # Wait for completion
                max_duration = parse_duration(self.config.guardrails.max_run_duration)
                result = self.executor.wait(session_name, timeout=max_duration)
                self._active_process_id = None

                self.store.set_run_exit_code(run_id, result.exit_code)

            except Exception as e:
                logger.error(f"Training launch failed: {e}")
                self.store.set_run_status(run_id, "failed")

                session = self.store.get_session(session.id)
                new_retry = session.retry_budget - 1
                self.store.update_session_retry_budget(session.id, new_retry)

                if new_retry <= 0:
                    return self._terminate(session, TerminationReason.RETRY_EXHAUSTION, base_branch)

                error_context = f"Training launch failed:\n{e}\n\nInvestigate and fix."
                agent_result = self._run_agent_for_fix(error_context, run_id)
                if agent_result and agent_result.files_modified:
                    sha = self.git.commit(f"Revis fix: {agent_result.rationale}")
                    self.store.attach_decision(
                        run_id,
                        Decision(
                            action_type="code_patch",
                            rationale=agent_result.rationale,
                            commit_sha=sha,
                        ),
                    )
                continue

            # Handle run failure
            if result.failed:
                logger.warning(f"Run failed with exit code {result.exit_code}")
                self.store.set_run_status(run_id, "failed")

                session = self.store.get_session(session.id)
                new_retry = session.retry_budget - 1
                self.store.update_session_retry_budget(session.id, new_retry)

                if new_retry <= 0:
                    return self._terminate(session, TerminationReason.RETRY_EXHAUSTION, base_branch)

                # Get error from logs and run agent to fix
                log_tail = self.executor.get_log_tail(
                    f"{run_output_dir}/train.log",
                    self.config.context.log_tail_lines,
                )
                error_context = f"Training failed (exit {result.exit_code}):\n{log_tail}"
                agent_result = self._run_agent_for_fix(error_context, run_id)
                if agent_result:
                    if agent_result.escalate:
                        return self._terminate(
                            session, TerminationReason.LLM_ESCALATION, base_branch
                        )
                    if agent_result.files_modified:
                        sha = self.git.commit(f"Revis fix: {agent_result.rationale}")
                        self.store.attach_decision(
                            run_id,
                            Decision(
                                action_type="code_patch",
                                rationale=agent_result.rationale,
                                commit_sha=sha,
                            ),
                        )
                continue

            self.store.set_run_status(run_id, "completed")

            # Collect metrics based on configured source
            logger.info("Collecting evaluation results...")
            if self.config.metrics.source == "wandb":
                # For W&B, parse run ID from training log and fetch metrics
                log_content = self.executor.get_log_tail(
                    f"{run_output_dir}/train.log",
                    lines=1000,  # Should be enough to capture W&B init output
                )
                metrics = self.metrics_collector.get_metrics_from_log(log_content)
                if metrics is None:
                    logger.error("Failed to get metrics from W&B")
                    self.store.set_run_status(run_id, "failed")
                    session = self.store.get_session(session.id)
                    new_retry = session.retry_budget - 1
                    self.store.update_session_retry_budget(session.id, new_retry)
                    if new_retry <= 0:
                        return self._terminate(
                            session, TerminationReason.RETRY_EXHAUSTION, base_branch
                        )
                    continue
                eval_result = EvalResult(metrics=metrics)
            else:
                # For eval.json, read from run output directory
                eval_path = f"{run_output_dir}/eval.json"
                try:
                    import json as json_mod

                    content = self.executor.read_file(eval_path)
                    data = json_mod.loads(content)
                    metrics = {
                        k: float(v)
                        for k, v in data.get("metrics", {}).items()
                        if isinstance(v, (int, float))
                    }
                    eval_result = EvalResult(metrics=metrics)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to read eval.json: {e}")
                    self.store.set_run_status(run_id, "failed")
                    session = self.store.get_session(session.id)
                    new_retry = session.retry_budget - 1
                    self.store.update_session_retry_budget(session.id, new_retry)
                    if new_retry <= 0:
                        return self._terminate(
                            session, TerminationReason.RETRY_EXHAUSTION, base_branch
                        )
                    continue

            # Log metrics
            self.store.log_metrics(run_id, eval_result.metrics)
            primary_value = eval_result.metrics.get(self.config.metrics.primary)
            logger.info(f"Metrics: {self.config.metrics.primary}={primary_value}")

            # Check target achievement
            if self.config.metrics.target is not None and primary_value is not None:
                if self.config.metrics.minimize:
                    achieved = primary_value <= self.config.metrics.target
                else:
                    achieved = primary_value >= self.config.metrics.target

                if achieved:
                    logger.info(f"Target achieved: {primary_value} vs {self.config.metrics.target}")
                    return self._terminate(session, TerminationReason.TARGET_ACHIEVED, base_branch)

            # Run guardrails
            metric_history = self.analyzer.get_metric_history(session.id)
            initial_value = metric_history[0] if metric_history else None

            guardrail_results = self.guardrails.check_eval_result(
                eval_result=eval_result,
                primary_metric=self.config.metrics.primary,
                initial_value=initial_value,
                metric_history=metric_history,
                minimize=self.config.metrics.minimize,
            )

            # Check for critical violations
            if self.guardrails.has_critical_violation(guardrail_results):
                violations = self.guardrails.get_violations(guardrail_results)
                logger.warning(f"Critical guardrail violations: {[v.message for v in violations]}")

            # Check plateau
            for gr in guardrail_results:
                if gr.guardrail == "plateau_detection" and gr.triggered:
                    logger.info("Plateau detected")
                    return self._terminate(session, TerminationReason.PLATEAU, base_branch)

            # Get previous eval for comparison
            prev_runs = self.store.query_runs(session_id=session.id, limit=2)
            prev_eval = None
            if len(prev_runs) > 1:
                prev_metrics = self.store.get_run_metrics(prev_runs[1].id)
                if prev_metrics:
                    prev_eval_dict = {m.name: m.value for m in prev_metrics}
                    prev_eval = EvalResult(metrics=prev_eval_dict)

            # Analyze run
            analysis = self.analyzer.analyze_run(session, eval_result, prev_eval)

            # Get run summaries for context
            run_summaries = self.analyzer.summarize_runs_for_context(
                session.id, self.config.context.history
            )

            # Get baseline value
            baseline_value = initial_value

            # Build iteration context and run agent
            logger.info("Running agent to propose improvements...")

            # Reset tool executor state for this iteration
            self.tool_executor.config_changes = []
            self.tool_executor.next_command = None
            self.tool_executor.code_change_request = None
            self.tool_executor.files_modified = []
            self.tool_executor._executor = self.executor
            self.tool_executor._run_output_dir = run_output_dir

            task = build_iteration_context(
                run_summaries=run_summaries,
                eval_result=eval_result,
                primary_metric=self.config.metrics.primary,
                baseline_value=baseline_value,
                guardrail_results=guardrail_results,
                metric_delta=analysis.metric_delta,
                constraints=self.config.context.constraints,
                target_value=self.config.metrics.target,
                minimize=self.config.metrics.minimize,
                train_command=current_train_cmd,
            )

            tracer = self._create_tracer(run_id)
            agent_result = run_agent(
                task=task,
                system_prompt=SYSTEM_PROMPT,
                executor=self.tool_executor,
                client=self.llm,
                max_iterations=self.config.context.max_agent_iterations,
                tracer=tracer,
            )
            tracer.on_iteration_complete(
                iteration, agent_result.rationale, agent_result.significant
            )
            self.store.update_session_cost(session.id, self.llm.total_cost)
            logger.info(f"Agent finished (cost so far: ${self.llm.total_cost:.2f})")

            # Handle escalation
            if agent_result.escalate:
                logger.info(f"Agent escalated: {agent_result.escalate_reason}")
                return self._terminate(session, TerminationReason.LLM_ESCALATION, base_branch)

            # Determine what changes were made
            has_config_changes = len(self.tool_executor.config_changes) > 0
            has_command_change = self.tool_executor.next_command is not None
            has_code_request = self.tool_executor.code_change_request is not None

            # Handle no changes (plateau)
            if not has_config_changes and not has_command_change and not has_code_request:
                logger.info("Agent proposed no changes - treating as plateau")
                return self._terminate(session, TerminationReason.PLATEAU, base_branch)

            # Determine change type and description
            change_type = "config" if has_config_changes else "cli_args"
            change_descriptions = []

            if has_config_changes:
                for change in self.tool_executor.config_changes:
                    change_descriptions.append(
                        f"{change['key']}: {change['old_value']} → {change['new_value']}"
                    )

            if has_command_change:
                change_type = "cli_args" if not has_config_changes else change_type
                change_descriptions.append(f"command: {self.tool_executor.next_command}")

            # Handle code change request (hand off to coding agent)
            if has_code_request:
                change_type = "code_handoff"
                request = self.tool_executor.code_change_request

                # Store the suggestion
                suggestion_id = self.store.create_suggestion(
                    session_id=session.id,
                    suggestion_type="code",
                    content=request["suggestion"],
                    run_id=run_id,
                )

                # Hand off to coding agent if available
                if self.coding_agent is not None:
                    logger.info("Handing off code change to coding agent...")
                    handoff_context = HandoffContext(
                        iteration_history=run_summaries,
                        latest_metrics=eval_result.metrics,
                        suggestion=request["suggestion"],
                        relevant_files=request.get("relevant_files", []),
                        constraints=self.config.context.constraints or None,
                    )
                    handoff_result = self.coding_agent.handoff(handoff_context)

                    if handoff_result.success:
                        change_descriptions.append(f"[code] {request['suggestion'][:50]}...")
                        self.store.update_suggestion_status(
                            suggestion_id, "accepted", self.config.coding_agent.type
                        )
                    else:
                        logger.warning(f"Coding agent failed: {handoff_result.error_message}")
                        self.store.update_suggestion_status(suggestion_id, "rejected")
                else:
                    logger.info("No coding agent available - pausing for manual intervention")
                    self.console.print(
                        "[yellow]Code change requested but no coding agent available.[/yellow]"
                    )
                    self.console.print(f"[yellow]Suggestion: {request['suggestion']}[/yellow]")

            change_description = "; ".join(change_descriptions) if change_descriptions else None

            # Update run with change info
            self.store.update_run_change(
                run_id=run_id,
                change_type=change_type,
                change_description=change_description,
                hypothesis=agent_result.rationale,
            )

            # Commit any config changes
            if has_config_changes or has_code_request:
                commit_msg = f"Revis iteration {iteration}: {agent_result.rationale}"
                sha = self.git.commit(commit_msg)
                self.store.set_run_commit(run_id, sha)
                logger.info(f"Committed {sha[:7]}: {agent_result.rationale}")

                # Record decision
                self.store.attach_decision(
                    run_id,
                    Decision(
                        action_type=change_type,
                        rationale=agent_result.rationale,
                        commit_sha=sha,
                    ),
                )

            # Update training command if agent specified one for next iteration
            if has_command_change:
                current_train_cmd = self.tool_executor.next_command
                logger.info(f"Next iteration will use command: {current_train_cmd}")

            # Update budget for runs
            if budget.type == "runs":
                self.store.update_session_budget(session.id, iteration)

            # Refresh session
            session = self.store.get_session(session.id)

    def _run_agent_for_fix(self, error_context: str, run_id: str):
        """Run agent to fix an error."""
        logger.info("Running agent to fix error...")

        # Reset tool executor state
        self.tool_executor.config_changes = []
        self.tool_executor.next_command = None
        self.tool_executor.code_change_request = None
        self.tool_executor.files_modified = []

        task = f"""The training run failed. Here's the error:

{error_context}

Please use the available tools to:
1. Read relevant files to understand the issue
2. If the issue is a configuration problem, use modify_config to fix it
3. If the issue is a CLI argument problem, use set_next_command to fix it
4. If the issue requires code changes, use request_code_change

When done, provide:
RATIONALE: <what you fixed and why>
"""

        tracer = self._create_tracer(run_id)
        agent_result = run_agent(
            task=task,
            system_prompt=SYSTEM_PROMPT,
            executor=self.tool_executor,
            client=self.llm,
            max_iterations=self.config.context.max_agent_iterations,
            tracer=tracer,
        )
        if run_id:
            run = self.store.get_run(run_id)
            if run:
                self.store.update_session_cost(run.session_id, self.llm.total_cost)

        # Check if any changes were made
        has_changes = (
            len(self.tool_executor.config_changes) > 0
            or self.tool_executor.next_command is not None
            or self.tool_executor.code_change_request is not None
        )
        agent_result.files_modified = ["config_changed"] if has_changes else []

        return agent_result

    def _terminate(
        self,
        session: Session,
        reason: TerminationReason,
        base_branch: str,
    ) -> Session:
        """Terminate session locally (no PR creation - use revis export)."""
        logger.info(f"Terminating session: {reason.value}")

        # End session - no PR URL since we're not creating one here
        self.store.end_session(session.id, reason, pr_url=None)

        # Checkout back to base branch, handling uncommitted changes
        try:
            self.git.checkout(base_branch)
        except subprocess.CalledProcessError:
            # Uncommitted changes from failed iteration - stash them and retry
            logger.warning("Stashing uncommitted changes before checkout")
            self.git.stash()
            self.git.checkout(base_branch)

        return self.store.get_session(session.id)

    def _resume(
        self,
        session: Session,
        budget: Budget,
    ) -> Session:
        """Resume an existing session."""
        # Clear any stale stop signal
        if STOP_SIGNAL_FILE.exists():
            STOP_SIGNAL_FILE.unlink()

        # Get base branch from the session's base_sha
        # We need to figure out what branch it was based on
        # For now, use 'main' as default, but ideally this would be stored
        base_branch = "main"

        start_time = time.time()

        logger.info(f"Resuming session '{session.name}' (ID: {session.id})")

        try:
            return self._run_loop(session, budget, start_time, base_branch)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return self._terminate(session, TerminationReason.USER_STOP, base_branch)
        except Exception as e:
            logger.error(f"Loop failed with error: {e}")
            self._terminate(session, TerminationReason.ERROR, base_branch)
            raise
        finally:
            self._cleanup_active_process()
            self.executor.close()


def run_loop(
    config: RevisConfig,
    name: str,
    budget: Budget,
    console: Console,
    baseline_run_id: str | None = None,
) -> Session:
    """Run the main Revis loop."""
    repo_path = Path.cwd()
    store = SQLiteRunStore(REVIS_DIR / "revis.db")

    loop = RevisLoop(config, store, repo_path, console)
    return loop.run(name, budget, baseline_run_id)


def resume_loop(
    config: RevisConfig,
    session: Session,
    console: Console,
) -> Session:
    """Resume a stopped session."""
    repo_path = Path.cwd()
    store = SQLiteRunStore(REVIS_DIR / "revis.db")

    # Calculate remaining budget
    remaining = session.budget.remaining()
    budget = Budget(type=session.budget.type, value=remaining)

    loop = RevisLoop(config, store, repo_path, console)

    # Checkout the session branch
    loop.git.checkout(session.branch)

    # Resume uses the existing session name
    return loop._resume(session, budget)
