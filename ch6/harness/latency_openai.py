# harness/latency_openai.py
import time
import openai

LATENCY_BUDGETS_OAI: dict[str, float] = {
    "search":     2.0,
    "classify":   3.0,
    "summarise": 10.0,
    "draft":     30.0,
}


def generate_oai(
    prompt: str,
    feature: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    stream: bool = True,
) -> dict:
    client = openai.OpenAI()
    budget = LATENCY_BUDGETS_OAI.get(feature, 15.0)
    start = time.monotonic()
    ttft: float | None = None
    chunks: list[str] = []

    if stream:
        with client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ) as s:
            for chunk in s:
                now = time.monotonic()
                if ttft is None:
                    ttft = now - start
                if now - start > budget:
                    raise TimeoutError(
                        f"Budget exceeded for '{feature}': "
                        f"{now - start:.1f}s > {budget:.1f}s"
                    )
                delta = chunk.choices[0].delta.content or ""
                chunks.append(delta)
        result = "".join(chunks)
    else:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        ttft = time.monotonic() - start
        result = response.choices[0].message.content or ""

    total = time.monotonic() - start
    return {
        "result":   result,
        "ttft_ms":  round((ttft or total) * 1000),
        "total_ms": round(total * 1000),
        "feature":  feature,
        "model":    model,
    }
