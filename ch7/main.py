# main.py
import warnings
from harness.context import ContextManager, ContextFillWarning

ctx = ContextManager(model="claude-haiku-4-5-20251001")

# Set the system prompt — marked important automatically
ctx.add_system(
    "You are a coding assistant for a Python project. "
    "Always use type hints. Never use global variables. "
    "Prefer dataclasses over plain dicts for structured data."
)

# Mark decisions as important so they survive trimming
ctx.add("user", "We've decided to use PostgreSQL, not SQLite.", important=True)
ctx.add("assistant", "Understood. I'll use PostgreSQL in all examples.", important=True)

# Regular conversation
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    response = ctx.send("Write a function to fetch a user by ID.")
    print(response["result"])
    print(f"Fill rate: {response['fill_rate'] * 100:.0f}%")

    for warning in w:
        if issubclass(warning.category, ContextFillWarning):
            print(f"Warning: {warning.message}")

# When one task is done, start fresh for the next
ctx.reset()
ctx.add("user", "Now let's work on the authentication module.", important=True)
