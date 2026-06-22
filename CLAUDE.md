# CLAUDE.md

> GitHub: https://github.com/Xusysysy/pyvision.git

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. 喵 Rule (ABSOLUTE — NEVER SKIP)

**Every sentence you output MUST start with "喵".** This includes responses, tool descriptions, code explanations, questions, and summaries. No exceptions. If you output 5 sentences, all 5 start with 喵. This is a hard requirement to verify CLAUDE.md compliance.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## 5. Auto Git Push After Build

**When the user continues conversation after a build (with no explicit instruction to skip), first commit and push to preserve changes:**

- If there are uncommitted changes, create a commit with a concise message summarizing the changes made in the previous turn.
- Then run `git push`.
- If the user explicitly says to skip or do something else first, follow that instead.
- Never force push. If push fails (e.g., no remote, no permission), report the error and continue.

## 6. Build After Every Modification

**After every code modification, build the exe with PyInstaller and report the output location:**

- Run: `pyinstaller camera_debugger.spec --noconfirm` (from project root)
- If build succeeds, report the output directory (typically `dist/camera_debugger/`) and the exe path
- If build fails, report the error and stop — do not skip the build

## 7. Prefer Edit Over Write + Sync STRUCTURE.md

**Prefer modifying files with the Edit tool rather than rewriting entire files. Only use Write when creating new files or when the scope of changes exceeds 50% of the file.**

**After any structural change (new/removed files, changed component responsibilities, navigation flow updates), sync the changes to `STRUCTURE.md`.**

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.