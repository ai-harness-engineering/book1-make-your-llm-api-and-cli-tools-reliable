# AI Harness Engineering — Code Samples

Source code for *AI Harness Engineering: Make Your LLM API and CLI Tools Reliable* by Yuen Kit Lai.

## Requirements

- Python 3.11+
- `anthropic` SDK: `pip install anthropic`
- `openai` SDK (for OpenAI variants): `pip install openai`
- Set your API key: `export ANTHROPIC_API_KEY="your-key"`

## Structure

```
ch1/
  harness.py              # Quick-start harness: all seven concerns in one file

ch2/
  harness.py              # Prompt management (Anthropic SDK)
  harness_openai.py       # Prompt management (OpenAI SDK)
  harness_gemini.py       # Prompt management (Gemini CLI)

ch3/
  harness.py              # Input/output validation (Anthropic SDK)
  harness_openai.py       # Validation (OpenAI SDK)
  harness_gemini.py       # Validation (Gemini CLI)

ch4/
  harness.py              # Retry & fallback (Anthropic SDK)
  harness_openai.py       # Retry (OpenAI SDK)
  harness_gemini.py       # Retry (Gemini CLI)

ch5/
  harness.py              # Cost tracking (Anthropic SDK)
  harness_openai.py       # Cost tracking (OpenAI SDK)
  harness_gemini.py       # Cost tracking (Gemini CLI)

ch6/
  main.py                 # Latency budgeting entry point
  harness/
    latency.py            # Per-feature latency budgets + streaming TTFT
    latency_tracker.py    # P95 latency tracker
    parallel.py           # Parallel calls with asyncio.gather
    latency_openai.py     # Latency budgeting (OpenAI SDK)

ch7/
  main.py                 # Context management entry point
  harness/
    context.py            # Trim, summarise, ContextManager
    context_openai.py     # Context management (OpenAI SDK)

skills/
  checking-drift/         # /checking-drift slash command for Claude Code
  saving-decisions/       # /saving-decisions slash command for Claude Code
```

## Usage

Each chapter's code is self-contained. Run the Anthropic SDK version:

```bash
cd ch1
python harness.py
```

## Installing the Skills

Copy the skills directory into your project's Claude Code skills folder:

```bash
cp -r skills/ ~/.claude/skills/
```

Then use `/checking-drift` and `/saving-decisions` inside any Claude Code session.

## License

Copyright © 2026 Yuen Kit Lai. All rights reserved.
