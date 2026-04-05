# harness_gemini.py
import json
from pathlib import Path
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
PROMPTS_DIR = Path("prompts")


def load_prompt(name: str, version: str = "active") -> tuple[str, str]:
    if version == "active":
        active = json.loads((PROMPTS_DIR / "active.json").read_text())
        version = active[name]
    text = (PROMPTS_DIR / name / f"{version}.txt").read_text().strip()
    return text, version


def summarise(text: str, version: str = "active") -> dict:
    prompt, resolved = load_prompt("summarise", version)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{prompt}\n\n{text}")
    return {
        "result": response.text,
        "prompt_version": resolved,
    }
