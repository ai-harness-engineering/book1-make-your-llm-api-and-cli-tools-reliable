# harness/latency.py
import time
import warnings
import anthropic

# Per-feature latency budgets in seconds
LATENCY_BUDGETS: dict[str, float] = {
    "search":     2.0,
    "classify":   3.0,
    "summarise": 10.0,
    "draft":     30.0,
}

# Default budget for features not in the table
DEFAULT_BUDGET = 15.0

# Alert when a call exceeds this fraction of its budget
ALERT_THRESHOLD = 0.80


class LatencyBudgetExceeded(Exception):
    """Raised when a call exceeds its feature latency budget."""

    def __init__(self, feature: str, budget: float, elapsed: float):
        self.feature = feature
        self.budget = budget
        self.elapsed = elapsed
        super().__init__(
            f"Latency budget exceeded for '{feature}': "
            f"{elapsed:.2f}s > {budget:.1f}s budget"
        )


class LatencyWarning(UserWarning):
    """Issued when a call uses more than ALERT_THRESHOLD of its budget."""
    pass


def _budget(feature: str) -> float:
    return LATENCY_BUDGETS.get(feature, DEFAULT_BUDGET)

import logging

logger = logging.getLogger(__name__)


def generate(
    prompt: str,
    feature: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    stream: bool = True,
) -> dict:
    """
    Call the LLM with a latency budget enforced via streaming timeout.

    Returns a dict with keys:
      result       - the text response
      ttft_ms      - time to first token in milliseconds
      total_ms     - total wall time in milliseconds
      feature      - the feature name
      model        - the model used
    """
    client = anthropic.Anthropic()
    budget = _budget(feature)
    start = time.monotonic()
    ttft: float | None = None
    chunks: list[str] = []

    try:
        if stream:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for text in s.text_stream:
                    now = time.monotonic()
                    elapsed = now - start

                    if ttft is None:
                        ttft = elapsed  # first token received

                    if elapsed > budget:
                        raise LatencyBudgetExceeded(feature, budget, elapsed)

                    chunks.append(text)

            result = "".join(chunks)
        else:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            ttft = time.monotonic() - start  # no streaming: TTFT = total
            result = response.content[0].text

    except LatencyBudgetExceeded:
        raise
    except Exception as e:
        raise

    total = time.monotonic() - start
    ttft_ms = round((ttft or total) * 1000)
    total_ms = round(total * 1000)

    logger.info(
        "latency feature=%s model=%s ttft_ms=%d total_ms=%d budget_ms=%d",
        feature, model, ttft_ms, total_ms, round(budget * 1000),
    )

    # Warn if approaching budget
    if total >= budget * ALERT_THRESHOLD:
        warnings.warn(
            f"Feature '{feature}' used {total:.2f}s of {budget:.1f}s budget "
            f"({total / budget * 100:.0f}%)",
            LatencyWarning,
            stacklevel=2,
        )

    return {
        "result":    result,
        "ttft_ms":   ttft_ms,
        "total_ms":  total_ms,
        "feature":   feature,
        "model":     model,
    }
