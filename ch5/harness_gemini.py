# harness_gemini.py
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# USD per million tokens — verify against ai.google.dev/pricing
PRICING = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
}


def generate(prompt: str, *, user_id: str, feature: str,
             model_name: str = "gemini-1.5-flash",
             max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    current_spend = get_user_spend(user_id)
    if current_spend >= USER_DAILY_BUDGET_USD:
        raise BudgetExceededError(f"User {user_id!r} over daily budget.")

    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
    )
    response = model.generate_content(prompt)

    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    record_spend(user_id, cost)

    if cost > CALL_COST_ALERT_USD:
        warnings.warn(
            f"High-cost call: ${cost:.4f} | model={model_name} | feature={feature!r}",
            CostAlertWarning, stacklevel=2,
        )

    return {
        "result": response.text,
        "cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model_name,
    }
