---
name: checking-drift
description: Silently checks whether the last 5 exchanges have drifted from the locked decisions in _meta/decisions.md. Reports drift clearly if found; says nothing if clean.
user-invocable: true
---

# Drift Check

Silently read `_meta/decisions.md`. Compare it against the last 5 exchanges in the conversation.

If any exchange contradicts, ignores, or moves away from a locked decision, notify the user:

> ⚠️ Drift detected: [what drifted] — conflicts with [which decision]

If no drift is found, say nothing. Do not confirm "no drift found" — silence means clean.
