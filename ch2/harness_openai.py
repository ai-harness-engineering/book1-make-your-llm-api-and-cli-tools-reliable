# harness_openai.py
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()
PROMPTS_DIR = Path("prompts")


def load_prompt(name: str, version: str = "active") -> tuple[str, str]:
    if version == "active":
        active = json.loads((PROMPTS_DIR / "active.json").read_text())
        version = active[name]
    text = (PROMPTS_DIR / name / f"{version}.txt").read_text().strip()
    return text, version


def summarise(text: str, version: str = "active") -> dict:
    prompt, resolved = load_prompt("summarise", version)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=128,
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
    )
    return {
        "result": response.choices[0].message.content,
        "prompt_version": resolved,
    }
