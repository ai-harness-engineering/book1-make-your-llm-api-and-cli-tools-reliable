# harness_openai.py
import json
from openai import OpenAI
from pydantic import ValidationError

client = OpenAI()


def extract_contract(text: str) -> ContractFields:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=256,
        response_format={"type": "json_object"},   # reduces markdown wrapping
        messages=[
            {"role": "system", "content": (
                "Extract contract fields as JSON: "
                '{"parties": [...], "total_value": float, '
                '"effective_date": "YYYY-MM-DD", "payment_terms": "string"}'
            )},
            {"role": "user", "content": text},
        ],
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nRaw: {raw!r}") from e

    try:
        return ContractFields(**data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e
