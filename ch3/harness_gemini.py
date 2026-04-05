# harness_gemini.py
import json, os
import google.generativeai as genai
from pydantic import ValidationError

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "total_value": {"type": "number"},
        "effective_date": {"type": "string"},
        "payment_terms": {"type": "string"},
    },
    "required": ["parties", "total_value", "effective_date"],
}


def extract_contract(text: str) -> ContractFields:
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=CONTRACT_SCHEMA,
        ),
    )
    response = model.generate_content(
        f"Extract contract fields from this text:\n\n{text}"
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nRaw: {response.text!r}") from e

    try:
        return ContractFields(**data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e
