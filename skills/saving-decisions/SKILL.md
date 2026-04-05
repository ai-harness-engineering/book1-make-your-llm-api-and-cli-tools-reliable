---
name: saving-decisions
description: End-of-session check. Reports any decisions not yet saved to decisions.md, unresolved open questions, unsaved content, and a one-line session summary.
user-invocable: true
---

# Session Wrap-Up

Run this before closing a session. Check and report:

1. **Unsaved decisions** — any decisions made this session not yet recorded in `_meta/decisions.md`. If found, record them immediately.
2. **Unresolved open questions** — any open items in `_meta/decisions.md` that were raised but not resolved this session.
3. **Unsaved content** — any new content (analogies, examples, tips, diagrams) created this session that exists only in the conversation and has not been written to a file.
4. **Session summary** — one sentence describing what was accomplished.
