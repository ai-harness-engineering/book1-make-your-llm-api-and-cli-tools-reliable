# harness.py — Chapter 1: Your First LLM API Harness
import anthropic
import time
import json
from typing import Optional

# ── 1. PROMPT MANAGEMENT ──────────────────────────────────────────
# Store prompts as versioned constants, not inline strings.
# When behaviour changes, you know exactly what changed.
PROMPTS = {
    "summarise_v1": {
        "version": "1.0",
        "system": "You are a precise summariser. Always respond with valid JSON.",
        "user_template": "Summarise the following in exactly 3 bullet points.\n\nRespond ONLY with this JSON format:\n{{\"bullets\": [\"point1\", \"point2\", \"point3\"]}}\n\nText:\n{text}"
    }
}

# ── 2. INPUT VALIDATION ───────────────────────────────────────────
def validate_input(text: str) -> tuple[bool, str]:
    """Check inputs before sending to the model."""
    if not text or not text.strip():
        return False, "Input text is empty"
    if len(text) > 10000:
        return False, f"Input too long: {len(text)} chars (max 10000)"
    return True, ""

# ── 3. OUTPUT VALIDATION ──────────────────────────────────────────
def validate_output(response_text: str) -> tuple[bool, dict]:
    """Check the model's response is in the expected format."""
    try:
        data = json.loads(response_text)
        if "bullets" not in data:
            return False, {}
        if not isinstance(data["bullets"], list):
            return False, {}
        if len(data["bullets"]) != 3:
            return False, {}
        return True, data
    except json.JSONDecodeError:
        return False, {}

# ── 4. RETRY & FALLBACK ───────────────────────────────────────────
def call_with_retry(
    client: anthropic.Anthropic,
    messages: list,
    system: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0
) -> Optional[anthropic.types.Message]:
    """Retry on transient failures with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-opus-4-6",
                max_tokens=512,
                system=system,
                messages=messages
            )
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = backoff_seconds * (2 ** attempt)
                print(f"  Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("  Max retries reached. Returning fallback.")
                return None
        except anthropic.APITimeoutError:
            if attempt < max_retries - 1:
                print(f"  Timeout. Retrying... (attempt {attempt + 1})")
            else:
                return None
    return None

# ── 5. COST TRACKING ──────────────────────────────────────────────
cost_log = []

def log_cost(response: anthropic.types.Message, feature: str):
    """Track token usage per call."""
    entry = {
        "feature": feature,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "timestamp": time.time()
    }
    cost_log.append(entry)
    print(f"  Tokens used: {entry['total_tokens']} "
          f"(in: {entry['input_tokens']}, out: {entry['output_tokens']})")

# ── 6. LATENCY BUDGETING ──────────────────────────────────────────
LATENCY_BUDGET_SECONDS = 10.0

# ── 7. CONTEXT MANAGEMENT ────────────────────────────────────────
MAX_CONTEXT_MESSAGES = 10
conversation_history = []

def trim_context(messages: list, max_messages: int) -> list:
    """Keep context within limits. Always preserve the first message."""
    if len(messages) <= max_messages:
        return messages
    # Keep first message (often contains key instructions) + most recent
    return [messages[0]] + messages[-(max_messages - 1):]

# ── THE HARNESS ───────────────────────────────────────────────────
def summarise(text: str) -> dict:
    client = anthropic.Anthropic()
    prompt = PROMPTS["summarise_v1"]

    # Step 1+2: Validate input
    valid, error = validate_input(text)
    if not valid:
        return {"error": f"Invalid input: {error}"}

    # Step 7: Build and trim context
    conversation_history.append({
        "role": "user",
        "content": prompt["user_template"].format(text=text)
    })
    messages = trim_context(conversation_history, MAX_CONTEXT_MESSAGES)

    # Steps 4+6: Call with retry and latency budget
    start = time.time()
    response = call_with_retry(client, messages, prompt["system"])
    elapsed = time.time() - start

    if elapsed > LATENCY_BUDGET_SECONDS:
        print(f"  ⚠ Slow response: {elapsed:.1f}s (budget: {LATENCY_BUDGET_SECONDS}s)")

    # Fallback if all retries failed
    if response is None:
        return {"error": "Service unavailable. Please try again."}

    # Step 5: Track cost
    log_cost(response, "summarise")

    # Step 3: Validate output
    response_text = response.content[0].text
    conversation_history.append({"role": "assistant", "content": response_text})

    valid_output, data = validate_output(response_text)
    if not valid_output:
        return {"error": "Unexpected response format", "raw": response_text}

    return data


# ── RUN IT ────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = """
    The company reported record revenues of $4.2 billion in Q3,
    driven by strong performance in cloud services and AI products.
    Operating margins improved to 28%, up from 22% last year.
    The board approved a $500 million share buyback programme.
    Headcount increased by 12% to 45,000 employees globally.
    """

    print("Running harness...\n")
    result = summarise(text)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nSummary:")
        for i, bullet in enumerate(result["bullets"], 1):
            print(f"  {i}. {bullet}")
