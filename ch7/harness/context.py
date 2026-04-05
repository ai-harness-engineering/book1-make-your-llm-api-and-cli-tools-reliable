# harness/context.py
from __future__ import annotations

import anthropic
from dataclasses import dataclass, field

# Token limits per model — leave headroom for the response
MODEL_TOKEN_LIMITS: dict[str, int] = {
    "claude-opus-4-6":           180_000,
    "claude-sonnet-4-6":         180_000,
    "claude-haiku-4-5-20251001":  90_000,
}

# Warn when context fill exceeds this fraction
FILL_ALERT_THRESHOLD = 0.80

# Default budget if model is not in the table
DEFAULT_LIMIT = 90_000


@dataclass
class Message:
    """A conversation message with an optional importance flag."""
    role: str        # "user" or "assistant"
    content: str
    important: bool = False   # protected messages are never trimmed

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

import logging

logger = logging.getLogger(__name__)


def count_tokens_exact(messages: list[Message], model: str) -> int:
    """
    Use the Anthropic token counting endpoint for an exact count.
    Costs one API call — use for critical paths or debugging.
    """
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=model,
        messages=[m.to_dict() for m in messages],
    )
    return response.input_tokens


def count_tokens_estimate(messages: list[Message]) -> int:
    """
    Estimate token count without an API call.
    Rule of thumb: 1 token ≈ 4 characters.
    Accurate to within ~10% for English prose.
    """
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


def trim_history(
    messages: list[Message],
    token_limit: int,
    exact: bool = False,
    model: str = "claude-haiku-4-5-20251001",
) -> list[Message]:
    """
    Remove oldest non-important messages until history fits within token_limit.
    Important messages are never removed.
    Returns a new list — does not modify the input.
    """
    trimmed = list(messages)

    while True:
        count = (
            count_tokens_exact(trimmed, model)
            if exact
            else count_tokens_estimate(trimmed)
        )
        if count <= token_limit:
            break

        # Find the oldest non-important message to remove
        removable = [i for i, m in enumerate(trimmed) if not m.important]
        if not removable:
            # All remaining messages are marked important — cannot trim further
            logger.warning(
                "context trim: all %d messages are marked important; "
                "cannot trim below %d tokens",
                len(trimmed), count,
            )
            break

        trimmed.pop(removable[0])
        logger.debug("context trim: removed oldest non-important message")

    return trimmed


def summarise_oldest(
    messages: list[Message],
    keep_recent: int = 10,
    summary_model: str = "claude-haiku-4-5-20251001",
) -> list[Message]:
    """
    Summarise the oldest messages into a single compact summary message.
    The most recent `keep_recent` messages are left intact.

    Uses a cheap/fast model for the summarisation call.
    Returns a new list starting with the summary, then the recent messages.
    """
    if len(messages) <= keep_recent:
        return list(messages)

    oldest = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in oldest
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=summary_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarise the following conversation history in 3–5 bullet points. "
                    "Focus on decisions made, constraints established, and any explicit "
                    "instructions given. Be concise — this summary replaces the full history.\n\n"
                    f"{history_text}"
                ),
            }
        ],
    )

    summary_text = response.content[0].text
    summary_message = Message(
        role="user",
        content=f"[Conversation summary — earlier history compressed]\n{summary_text}",
        important=True,   # protect summaries from future trimming
    )

    logger.info(
        "context summarise: compressed %d messages into summary (%d chars)",
        len(oldest), len(summary_text),
    )

    return [summary_message] + list(recent)

import warnings


class ContextFillWarning(UserWarning):
    """Issued when context fill rate exceeds FILL_ALERT_THRESHOLD."""
    pass


class ContextManager:
    """
    Manages a conversation history within a token budget.

    Usage:
        ctx = ContextManager(model="claude-haiku-4-5-20251001")
        ctx.add_system("You are a helpful assistant. Prefer concise answers.")
        ctx.add("user", "What is the capital of France?")
        response = ctx.send()
        ctx.add("assistant", response["result"])
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens_response: int = 1024,
        summarise_threshold: float = 0.70,   # summarise when fill exceeds 70%
        exact_counting: bool = False,
    ):
        self.model = model
        self.max_tokens_response = max_tokens_response
        self.exact_counting = exact_counting
        self.token_limit = MODEL_TOKEN_LIMITS.get(model, DEFAULT_LIMIT)
        # Reserve headroom for the response
        self.budget = self.token_limit - max_tokens_response
        self.summarise_threshold = summarise_threshold
        self._history: list[Message] = []
        self._system: str | None = None

    def add_system(self, text: str) -> None:
        """Set the system prompt. Replaces any existing system prompt."""
        self._system = text

    def add(self, role: str, content: str, important: bool = False) -> None:
        """Append a message to the conversation history."""
        self._history.append(Message(role=role, content=content, important=important))

    def _all_messages(self) -> list[Message]:
        """Return system message (if set) + history."""
        if self._system:
            return [Message(role="system", content=self._system, important=True)] + self._history
        return list(self._history)

    def token_count(self) -> int:
        msgs = self._all_messages()
        if self.exact_counting:
            return count_tokens_exact(msgs, self.model)
        return count_tokens_estimate(msgs)

    def fill_rate(self) -> float:
        return self.token_count() / self.budget

    def _prepare(self) -> list[Message]:
        """
        Prepare the message list for sending:
        1. Check fill rate — warn if high
        2. Summarise if approaching summarise_threshold
        3. Trim if still over budget
        """
        msgs = self._all_messages()
        count = count_tokens_estimate(msgs)
        fill = count / self.budget

        logger.info(
            "context fill: %d tokens / %d budget (%.0f%%)",
            count, self.budget, fill * 100,
        )

        if fill >= FILL_ALERT_THRESHOLD:
            warnings.warn(
                f"Context fill rate is {fill * 100:.0f}% of budget "
                f"({count} / {self.budget} tokens). "
                "History will be summarised or trimmed.",
                ContextFillWarning,
                stacklevel=3,
            )

        # Summarise if over threshold
        if fill >= self.summarise_threshold:
            # Summarise everything except system + last 10 messages
            history_only = [m for m in msgs if m.role != "system"]
            compressed = summarise_oldest(history_only, keep_recent=10)
            self._history = compressed
            msgs = self._all_messages()

        # Trim if still over budget
        msgs = trim_history(msgs, self.budget, exact=self.exact_counting, model=self.model)

        return msgs

    def send(self, prompt: str | None = None) -> dict:
        """
        Send the current context to the model and return the response.
        If `prompt` is provided, it is added as a user message before sending.
        """
        if prompt:
            self.add("user", prompt)

        msgs = self._prepare()
        client = anthropic.Anthropic()

        # Separate system from user/assistant messages
        system_msgs = [m for m in msgs if m.role == "system"]
        chat_msgs = [m for m in msgs if m.role != "system"]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens_response,
            "messages": [m.to_dict() for m in chat_msgs],
        }
        if system_msgs:
            kwargs["system"] = system_msgs[0].content

        response = client.messages.create(**kwargs)
        result = response.content[0].text

        # Add the assistant's reply to history
        self.add("assistant", result)

        return {
            "result": result,
            "tokens_used": self.token_count(),
            "fill_rate": self.fill_rate(),
        }

    def reset(self) -> None:
        """Start a fresh context. Keeps the system prompt, clears history."""
        self._history = []
        logger.info("context reset: history cleared")
