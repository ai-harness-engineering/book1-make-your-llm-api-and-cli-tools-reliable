# harness.py
import json
import re
import anthropic
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal

client = anthropic.Anthropic()
PROMPTS_DIR = Path("prompts")
CONFIDENCE_THRESHOLD = 0.7


def extract_json(text: str) -> str:
    """Strip markdown code fences if the model wrapped the JSON in them."""
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    if matches:
        return matches[-1].strip()  # last block is the final answer
    return text.strip()
