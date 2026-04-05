# harness_openai.py
from openai import RateLimitError, APITimeoutError, APIStatusError
from openai import OpenAI

client = OpenAI()


def call_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=64,
                timeout=TIMEOUT,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"result": response.choices[0].message.content, "fallback": False}

        except RateLimitError:
            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt))

        except APITimeoutError:
            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt, base=BASE_DELAY * 0.5, jitter=False))

        except APIStatusError as e:
            if e.status_code >= 500:
                if attempt == max_retries:
                    break
                time.sleep(_backoff(attempt, jitter=False))
            else:
                raise

    return FALLBACK
