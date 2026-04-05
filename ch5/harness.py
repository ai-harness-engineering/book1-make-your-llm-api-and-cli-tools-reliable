# harness.py
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
import anthropic

client = anthropic.Anthropic()

# USD per million tokens — verify against anthropic.com/pricing
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.25, "output":  1.25},
}

DEFAULT_MAX_TOKENS = 1024          # hard cap — never leave this unset
USER_DAILY_BUDGET_USD = 1.00       # reject calls from users over this threshold
CALL_COST_ALERT_USD = 0.10         # warn when a single call exceeds this


class BudgetExceededError(Exception):
    pass


class CostAlertWarning(UserWarning):
    pass


@dataclass
class CallRecord:
    model: str
    feature: str
    user_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
