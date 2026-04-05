# main.py
import warnings
from harness.latency import generate, LatencyBudgetExceeded, LatencyWarning

# Normal call with streaming
try:
    response = generate("Summarise this contract in three bullet points.", feature="summarise")
    print(response["result"])
    print(f"TTFT: {response['ttft_ms']}ms  Total: {response['total_ms']}ms")
except LatencyBudgetExceeded as e:
    print(f"Too slow: {e}")
    print("Fallback: showing cached summary or asking user to try again.")

# Capture warnings to detect calls approaching the budget
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    response = generate("Classify this email.", feature="classify")
    for warning in w:
        if issubclass(warning.category, LatencyWarning):
            print(f"Warning: {warning.message}")
