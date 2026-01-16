"""Agentic loop for tool-based code editing."""

import json
import logging
from dataclasses import dataclass, field

from revis.llm.client import LLMClient
from revis.llm.tools import TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from running the agent."""

    rationale: str
    significant: bool = False
    escalate: bool = False
    escalate_reason: str | None = None
    files_modified: list[str] = field(default_factory=list)
    tool_calls_count: int = 0


def run_agent(
    task: str,
    system_prompt: str,
    executor: ToolExecutor,
    client: LLMClient,
    max_iterations: int = 20,
) -> AgentResult:
    """Run the agentic loop until the LLM finishes or hits max iterations.

    Args:
        task: The task description (iteration context)
        system_prompt: System prompt for the agent
        executor: ToolExecutor instance with repo_root and deny patterns
        client: LLMClient instance for API calls
        max_iterations: Maximum number of LLM round-trips

    Returns:
        AgentResult with rationale, files modified, and status
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tool_calls_count = 0

    for iteration in range(max_iterations):
        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

        response = client.complete_with_tools(messages, TOOLS)

        if response.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })

            for tool_call in response.tool_calls:
                tool_calls_count += 1
                name = tool_call["name"]
                args = tool_call["arguments"]
                tool_id = tool_call["id"]

                # Log at INFO level so we can see what agent is doing
                args_summary = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
                logger.info(f"Tool call: {name}({args_summary})")
                result = executor.execute(name, args)
                result_summary = result[:150] + "..." if len(result) > 150 else result
                logger.debug(f"Tool result: {result_summary}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })
        else:
            messages.append({
                "role": "assistant",
                "content": response.content,
            })
            break

    final_content = messages[-1].get("content", "") if messages else ""
    logger.info(f"Agent final response: {final_content[:500]}...")
    result = parse_agent_response(final_content)
    result.files_modified = executor.files_modified
    result.tool_calls_count = tool_calls_count

    if tool_calls_count == 0:
        logger.warning("Agent made no tool calls - treating as plateau")

    logger.info(
        f"Agent finished: {len(result.files_modified)} files modified, "
        f"{tool_calls_count} tool calls"
    )

    return result


def parse_agent_response(text: str) -> AgentResult:
    """Parse the agent's final summary message.

    Looks for:
        RATIONALE: <1-2 sentence explanation>
        SIGNIFICANT: yes/no
        ESCALATE: <reason>
    """
    rationale = None
    significant = False
    escalate = False
    escalate_reason = None

    for line in text.strip().split("\n"):
        line_upper = line.upper()
        if line_upper.startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()
        elif line_upper.startswith("SIGNIFICANT:"):
            significant = "yes" in line.lower()
        elif line_upper.startswith("ESCALATE:"):
            escalate = True
            escalate_reason = line.split(":", 1)[1].strip()

    return AgentResult(
        rationale=rationale or "No rationale provided",
        significant=significant,
        escalate=escalate,
        escalate_reason=escalate_reason,
    )
