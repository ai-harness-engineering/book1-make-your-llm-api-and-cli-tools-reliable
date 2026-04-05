# harness/parallel.py
import asyncio
import anthropic


async def _call(client: anthropic.AsyncAnthropic, prompt: str, model: str, max_tokens: int) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def parallel_generate(prompts: list[str], model: str = "claude-haiku-4-5-20251001", max_tokens: int = 512) -> list[str]:
    """Run multiple prompts in parallel and return results in the same order."""
    client = anthropic.AsyncAnthropic()
    tasks = [_call(client, p, model, max_tokens) for p in prompts]
    return await asyncio.gather(*tasks)


# Usage
import asyncio

prompts = [
    "Summarise section 1 of this contract.",
    "Summarise section 2 of this contract.",
    "Summarise section 3 of this contract.",
]

# Sequential: ~9s total (3 × 3s per call)
# Parallel:   ~3s total (all three run at once)
summaries = asyncio.run(parallel_generate(prompts))
