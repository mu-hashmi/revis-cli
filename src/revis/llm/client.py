"""LiteLLM client wrapper with auto-fallback and cost tracking."""

import logging
from dataclasses import dataclass

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)

from revis.config import LLMConfig

logger = logging.getLogger(__name__)

# Disable litellm's verbose logging
litellm.set_verbose = False


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    used_fallback: bool


class LLMClient:
    """LiteLLM client with auto-fallback and cost tracking."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_cost = 0.0
        self.total_tokens = 0
        self.fallback_used = False

        if config.api_base:
            litellm.api_base = config.api_base

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Make a completion request with auto-fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            LLMResponse with content and metadata
        """
        models_to_try = [self.config.model] + self.config.fallback
        last_error = None

        for i, model in enumerate(models_to_try):
            is_fallback = i > 0
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Extract usage info
                usage = response.usage
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = prompt_tokens + completion_tokens

                # Calculate cost using litellm's cost tracking
                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    # Fallback cost estimation if litellm can't calculate
                    cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

                # Update totals
                self.total_cost += cost
                self.total_tokens += total_tokens
                if is_fallback:
                    self.fallback_used = True

                content = response.choices[0].message.content or ""

                if is_fallback:
                    logger.info(f"Used fallback model {model} after primary failed")

                return LLMResponse(
                    content=content,
                    model_used=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    used_fallback=is_fallback,
                )

            except (RateLimitError, APIConnectionError, ServiceUnavailableError, APIError) as e:
                last_error = e
                logger.warning(f"Model {model} failed: {e}. Trying next fallback...")
                continue

        # All models failed
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost when litellm can't calculate."""
        # Rough estimates per 1M tokens
        cost_per_1m = {
            "claude-sonnet-4-20250514": (3.0, 15.0),  # (input, output)
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-3-5-sonnet-20241022": (3.0, 15.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
        }

        # Default to Sonnet pricing
        input_cost, output_cost = cost_per_1m.get(model, (3.0, 15.0))

        return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "fallback_used": self.fallback_used,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.fallback_used = False
