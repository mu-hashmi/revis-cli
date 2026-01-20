"""Prompt construction for LLM agent."""

import logging

from revis.analyzer.compare import RunSummary, format_run_history
from revis.analyzer.detectors import GuardrailResult
from revis.types import EvalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Revis, an autonomous ML training optimizer. You analyze training \
results and propose changes to improve model metrics.

## What you receive

Each iteration, you'll be given:
- Run history with metrics from previous iterations
- Current run results (eval.json metrics) with target metric goal
- The training command being used
- Analysis of guardrail checks (NaN detection, divergence, plateau)

## What you MUST do

You MUST make at least one change every iteration. Explore the codebase with your tools, then modify config or code:

1. **Explore first**: Use list_directory('.') to see project structure, read_file to examine configs and code.

2. **Make a change**: Modify something to improve the metric. Common targets:
   - Training configs (YAML/JSON): learning rate, batch size, epochs, scheduler
   - Model code: architecture tweaks, regularization, loss functions
   - Data: augmentation, preprocessing settings

3. **Verify**: After writing, use run_command with `python -m py_compile <file>` to check syntax.

## Important

- You MUST propose a change each iteration—don't just observe
- Always read a file before modifying it
- Write complete file contents (not partial updates)
- Test one hypothesis per iteration
- Look for configs in common locations: config/, configs/, *.yaml, *.json
- You cannot modify revis.yaml (the Revis config file)

## When you're done

After making changes, respond with:

RATIONALE: <1-2 sentence explanation of what you changed and why>
SIGNIFICANT: <yes/no - is this a key decision point?>

Optionally, if you need to change CLI arguments for the next training run:

NEXT_COMMAND: <full training command with new arguments>

Use NEXT_COMMAND when hyperparameters are passed via CLI args rather than config files. The command you specify will be used for the next iteration. If not specified, the current command continues to be used.

## If truly stuck

Only if you've explored the codebase and genuinely cannot find anything to change:

ESCALATE: <explanation of why you cannot proceed>
"""


def build_history_section(summaries: list[RunSummary]) -> str:
    """Build the run history section."""
    if not summaries:
        return "<run_history>\nNo previous runs.\n</run_history>"
    return "<run_history>\n" + format_run_history(summaries) + "\n</run_history>"


def build_current_run_section(
    eval_result: EvalResult,
    primary_metric: str,
    baseline_value: float | None,
    target_value: float | None = None,
    minimize: bool = True,
) -> str:
    """Build the current run section."""
    lines = ["<current_run>"]

    current_value = eval_result.metrics.get(primary_metric)
    if current_value is not None:
        lines.append(f"Primary metric ({primary_metric}): {current_value:.6f}")
        if target_value is not None:
            gap = abs(current_value - target_value)
            direction = "above" if minimize else "below"
            lines.append(f"  Target: {target_value} (currently {gap:.3f} {direction} target)")
        if baseline_value is not None:
            delta = current_value - baseline_value
            pct = (delta / abs(baseline_value) * 100) if baseline_value != 0 else 0
            sign = "+" if delta > 0 else ""
            lines.append(f"  vs baseline: {sign}{delta:.6f} ({sign}{pct:.1f}%)")

    lines.append("\nAll metrics:")
    for name, value in eval_result.metrics.items():
        lines.append(f"  {name}: {value:.6f}")

    if eval_result.slices:
        lines.append("\nMetric slices:")
        for group_name, slices in eval_result.slices.items():
            lines.append(f"  {group_name}:")
            for slice_name, metrics in slices.items():
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                lines.append(f"    {slice_name}: {metrics_str}")

    lines.append("</current_run>")
    return "\n".join(lines)


def build_log_tail_section(log_tail: str, lines: int = 200) -> str:
    """Build the training log tail section."""
    log_lines = log_tail.strip().split("\n")
    if len(log_lines) > lines:
        log_lines = log_lines[-lines:]

    return (
        f'<training_log_tail lines="{len(log_lines)}">\n'
        + "\n".join(log_lines)
        + "\n</training_log_tail>"
    )


def build_analysis_section(
    guardrail_results: list[GuardrailResult],
    metric_delta: float | None,
    primary_metric: str,
) -> str:
    """Build the analysis section."""
    lines = ["<analysis>"]

    if metric_delta is not None:
        direction = "improved" if metric_delta < 0 else "worsened"
        lines.append(f"Metric change: {primary_metric} {direction} by {abs(metric_delta):.6f}")

    lines.append("\nGuardrail checks:")
    for result in guardrail_results:
        status = "TRIGGERED" if result.triggered else "OK"
        lines.append(f"  [{status}] {result.guardrail}: {result.message}")

    lines.append("</analysis>")
    return "\n".join(lines)


def build_constraints_section(constraints: list[str]) -> str:
    """Build the constraints section."""
    if not constraints:
        return ""
    lines = ["<constraints>"]
    for constraint in constraints:
        lines.append(f"  - {constraint}")
    lines.append("</constraints>")
    return "\n".join(lines)


def build_training_command_section(train_command: str) -> str:
    """Build the training command section."""
    return (
        "<training_command>\n"
        f"{train_command}\n"
        "(Use NEXT_COMMAND in your response to change CLI arguments for the next run)\n"
        "</training_command>"
    )


def build_iteration_context(
    run_summaries: list[RunSummary],
    eval_result: EvalResult,
    primary_metric: str,
    baseline_value: float | None,
    guardrail_results: list[GuardrailResult],
    metric_delta: float | None,
    constraints: list[str],
    target_value: float | None = None,
    minimize: bool = True,
    train_command: str | None = None,
) -> str:
    """Build the iteration context for the agent.

    This is passed as the user message to the agent. Unlike the old approach,
    we don't stuff file contents here—the agent reads files on demand via tools.
    Training logs are available via the get_training_logs tool.
    """
    history_section = build_history_section(run_summaries)
    current_run_section = build_current_run_section(
        eval_result, primary_metric, baseline_value, target_value, minimize
    )
    analysis_section = build_analysis_section(guardrail_results, metric_delta, primary_metric)

    logger.debug(
        f"Context section sizes: history={len(history_section)}, current_run={len(current_run_section)}, analysis={len(analysis_section)}"
    )

    sections = [history_section, current_run_section, analysis_section]

    if train_command:
        sections.append(build_training_command_section(train_command))

    if constraints:
        sections.append(build_constraints_section(constraints))

    sections.append(
        "\nUse the available tools to explore the codebase and make improvements. "
        "You MUST make at least one change. Start with list_directory('.') to see the project structure. "
        "Use get_training_logs if you need to see training output or debug issues."
    )

    result = "\n\n".join(sections)
    logger.info(f"Total iteration context size: {len(result)} chars")
    return result
