# harness.py
import json
from pathlib import Path
import anthropic

client = anthropic.Anthropic()
PROMPTS_DIR = Path("prompts")


def load_prompt(name: str, version: str = "active") -> tuple[str, str]:
    """Load a prompt by name. Returns (prompt_text, resolved_version)."""
    if version == "active":
        active = json.loads((PROMPTS_DIR / "active.json").read_text())
        version = active[name]
    text = (PROMPTS_DIR / name / f"{version}.txt").read_text().strip()
    return text, version


def summarise(text: str, version: str = "active") -> dict:
    prompt, resolved = load_prompt("summarise", version)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
    )
    return {
        "result": response.content[0].text,
        "prompt_version": resolved,          # ← logged with every response
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def ab_test(text: str, version_a: str, version_b: str) -> dict:
    """Run the same input through two prompt versions for direct comparison."""
    return {
        version_a: summarise(text, version=version_a),
        version_b: summarise(text, version=version_b),
    }
