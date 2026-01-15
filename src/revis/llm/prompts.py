"""Prompt construction for LLM interactions."""

from pathlib import Path

from revis.analyzer.compare import RunSummary, format_run_history
from revis.analyzer.detectors import GuardrailResult
from revis.config import ActionBoundsConfig, ContextConfig, RevisConfig
from revis.types import EvalResult, Session


SYSTEM_PROMPT = """You are Revis, an autonomous ML training optimizer. Your job is to analyze training results and propose changes to improve the model.

You will receive:
- Repository context (config files, model code)
- Run history with metrics
- Current run results
- Training log tail
- Analysis of guardrail checks

Based on this information, you must propose a change to improve the primary metric. Changes can be:
1. Config changes (learning rate, batch size, etc.)
2. Code changes (model architecture, data processing, etc.)

IMPORTANT RULES:
- You MUST propose a change. If you truly cannot improve further, emit <escalate/>.
- Keep changes focused and minimal - one hypothesis at a time.
- Provide a concise rationale (1-2 sentences).
- Mark significant changes with <significant/> tag.

OUTPUT FORMAT:
Use <edit> blocks for changes:

<edit>
<path>relative/path/to/file</path>
<search>
exact text to find
</search>
<replace>
replacement text
</replace>
</edit>

For new files or full replacement, omit <search>:

<edit>
<path>path/to/new/file</path>
<replace>
full file content
</replace>
</edit>

End with:
<rationale>Your 1-2 sentence explanation</rationale>

If this is a key decision point, add:
<significant/>

If you cannot propose any improvement, emit:
<escalate/>
<rationale>Explanation of why you cannot proceed</rationale>
"""


def build_context_section(
    config: ContextConfig,
    repo_root: Path,
) -> str:
    """Build the repository context section."""
    sections = []

    for file_path in config.include:
        full_path = repo_root / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                sections.append(f"<file path=\"{file_path}\">\n{content}\n</file>")
            except Exception as e:
                sections.append(f"<file path=\"{file_path}\" error=\"{e}\"/>")
        else:
            sections.append(f"<file path=\"{file_path}\" error=\"not found\"/>")

    return "<repo_context>\n" + "\n\n".join(sections) + "\n</repo_context>"


def build_history_section(summaries: list[RunSummary]) -> str:
    """Build the run history section."""
    return "<run_history>\n" + format_run_history(summaries) + "\n</run_history>"


def build_current_run_section(
    eval_result: EvalResult,
    primary_metric: str,
    baseline_value: float | None,
) -> str:
    """Build the current run section."""
    lines = ["<current_run>"]

    # Primary metric with comparison
    current_value = eval_result.metrics.get(primary_metric)
    if current_value is not None:
        lines.append(f"Primary metric ({primary_metric}): {current_value:.6f}")
        if baseline_value is not None:
            delta = current_value - baseline_value
            pct = (delta / abs(baseline_value) * 100) if baseline_value != 0 else 0
            sign = "+" if delta > 0 else ""
            lines.append(f"  vs baseline: {sign}{delta:.6f} ({sign}{pct:.1f}%)")

    # Other metrics
    lines.append("\nAll metrics:")
    for name, value in eval_result.metrics.items():
        lines.append(f"  {name}: {value:.6f}")

    # Slices if present
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
    # Truncate to specified lines
    log_lines = log_tail.strip().split("\n")
    if len(log_lines) > lines:
        log_lines = log_lines[-lines:]

    return f"<training_log_tail lines=\"{len(log_lines)}\">\n" + "\n".join(log_lines) + "\n</training_log_tail>"


def build_analysis_section(
    guardrail_results: list[GuardrailResult],
    metric_delta: float | None,
    primary_metric: str,
) -> str:
    """Build the analysis section."""
    lines = ["<analysis>"]

    # Metric change
    if metric_delta is not None:
        direction = "improved" if metric_delta < 0 else "worsened"  # Assuming minimize
        lines.append(f"Metric change: {primary_metric} {direction} by {abs(metric_delta):.6f}")

    # Guardrail results
    lines.append("\nGuardrail checks:")
    for result in guardrail_results:
        status = "TRIGGERED" if result.triggered else "OK"
        lines.append(f"  [{status}] {result.guardrail}: {result.message}")

    lines.append("</analysis>")
    return "\n".join(lines)


def build_bounds_section(bounds: ActionBoundsConfig) -> str:
    """Build the action bounds section."""
    lines = ["<action_bounds>"]

    if bounds.allow:
        lines.append("Allowed file patterns:")
        for pattern in bounds.allow:
            lines.append(f"  - {pattern}")

    if bounds.deny:
        lines.append("Denied file patterns (cannot modify):")
        for pattern in bounds.deny:
            lines.append(f"  - {pattern}")

    if bounds.constraints:
        lines.append("Constraints:")
        for constraint in bounds.constraints:
            lines.append(f"  - {constraint}")

    lines.append("</action_bounds>")
    return "\n".join(lines)


def build_prompt(
    config: RevisConfig,
    repo_root: Path,
    run_summaries: list[RunSummary],
    eval_result: EvalResult,
    baseline_value: float | None,
    log_tail: str,
    guardrail_results: list[GuardrailResult],
    metric_delta: float | None,
) -> list[dict]:
    """Build the full prompt for the LLM."""
    user_content_parts = [
        build_context_section(config.context, repo_root),
        build_history_section(run_summaries),
        build_current_run_section(eval_result, config.metrics.primary, baseline_value),
        build_log_tail_section(log_tail, config.context.log_tail_lines),
        build_analysis_section(guardrail_results, metric_delta, config.metrics.primary),
        build_bounds_section(config.action_bounds),
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_content_parts)},
    ]


def build_fix_prompt(
    error_message: str,
    failed_action: str,
    config: RevisConfig,
    repo_root: Path,
) -> list[dict]:
    """Build prompt for fixing a failed action."""
    context = build_context_section(config.context, repo_root)

    user_content = f"""The previous change failed with an error. Please fix it.

{context}

<failed_action>
{failed_action}
</failed_action>

<error>
{error_message}
</error>

Please provide a corrected version of the change that fixes this error.
Use the same <edit> block format as before.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
