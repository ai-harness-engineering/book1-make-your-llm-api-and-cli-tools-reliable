# harness/context_openai.py
import openai

MODEL_LIMITS_OAI = {
    "gpt-4o":       120_000,
    "gpt-4o-mini":  120_000,
    "gpt-4-turbo":  120_000,
}


def count_tokens_estimate_oai(messages: list[dict]) -> int:
    """Estimate token count: 1 token ≈ 4 characters."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def trim_history_oai(
    messages: list[dict],
    token_limit: int,
    protected_roles: set[str] | None = None,
) -> list[dict]:
    """
    Trim oldest non-system, non-pinned messages from an OpenAI message list.
    The first system message is always protected.
    """
    if protected_roles is None:
        protected_roles = {"system"}

    trimmed = list(messages)

    while count_tokens_estimate_oai(trimmed) > token_limit:
        # Find oldest non-protected message
        removable = [
            i for i, m in enumerate(trimmed)
            if m.get("role") not in protected_roles
        ]
        if not removable:
            break
        trimmed.pop(removable[0])

    return trimmed


def chat_with_context(
    history: list[dict],
    new_message: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """
    Send a message with managed context. Returns (response_text, updated_history).
    """
    budget = MODEL_LIMITS_OAI.get(model, 100_000) - max_tokens

    history = history + [{"role": "user", "content": new_message}]
    history = trim_history_oai(history, budget)

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=history,
    )

    reply = response.choices[0].message.content or ""
    history = history + [{"role": "assistant", "content": reply}]

    return reply, history


# Usage
history: list[dict] = [
    {"role": "system", "content": "You are a helpful coding assistant."}
]

reply, history = chat_with_context(history, "Write a function to sort a list of dicts by key.")
print(reply)
print(f"History length: {len(history)} messages")
