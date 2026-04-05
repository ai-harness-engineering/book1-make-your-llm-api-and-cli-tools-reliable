# harness_openai.py
from openai import OpenAI

client = OpenAI()

# USD per million tokens — verify against platform.openai.com/docs/pricing
PRICING = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output":  0.60},
}


def generate(prompt: str, *, user_id: str, feature: str,
             model: str = "gpt-4o-mini",
             max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    current_spend = get_user_spend(user_id)
    if current_spend >= USER_DAILY_BUDGET_USD:
        raise BudgetExceededError(f"User {user_id!r} over daily budget.")

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    record_spend(user_id, cost)

    if cost > CALL_COST_ALERT_USD:
        warnings.warn(
            f"High-cost call: ${cost:.4f} | model={model} | feature={feature!r}",
            CostAlertWarning, stacklevel=2,
        )

    return {
        "result": response.choices[0].message.content,
        "cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }
