# harness_gemini.py
import os
import time
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,   # 429 — rate limit
    DeadlineExceeded,    # timeout
    ServiceUnavailable,  # 503 — server error
    InvalidArgument,     # 400 — client error, don't retry
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


def call_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return {"result": response.text, "fallback": False}

        except ResourceExhausted:
            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt))

        except DeadlineExceeded:
            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt, base=BASE_DELAY * 0.5, jitter=False))

        except ServiceUnavailable:
            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt, jitter=False))

        except InvalidArgument:
            raise   # client error — don't retry

    return FALLBACK
