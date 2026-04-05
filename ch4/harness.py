# harness.py
import random
import time
import httpx
import anthropic
from anthropic import RateLimitError, APITimeoutError, APIStatusError

client = anthropic.Anthropic()

MAX_RETRIES = 3
BASE_DELAY = 1.0      # seconds
TIMEOUT = 30.0        # seconds per call

FALLBACK = {
    "result": None,
    "error": "Service temporarily unavailable — please try again shortly.",
    "fallback": True,
}
