# ChatGPT ↔ Codex Coordination Protocol

## Roles

### ChatGPT — research lead
- Define scientific hypotheses, method design, ablations, acceptance criteria, and next tasks.
- Review Codex reports, commits, tests, and experimental evidence.
- Own `PROJECT_STATE.md` and `CHATGPT_TO_CODEX.md`.

### Codex — engineering lead
- Implement only well-scoped tasks from `CHATGPT_TO_CODEX.md`.
- Prefer minimal, testable implementations before large-scale experiments.
- Report results, failures, assumptions, and commit SHAs in `CODEX_TO_CHATGPT.md`.
- Do not silently change the scientific objective.

## Mailbox ownership

- `coordination/CHATGPT_TO_CODEX.md`: ChatGPT writes; Codex reads.
- `coordination/CODEX_TO_CHATGPT.md`: Codex writes; ChatGPT reads.
- `coordination/PROJECT_STATE.md`: ChatGPT writes; Codex reads and proposes changes only through its report.

This avoids concurrent edits to the same file.

## Task format

Every task from ChatGPT should contain:
1. Task ID (`T001`, `T002`, ...)
2. Objective
3. Scientific hypothesis being tested
4. Scope / non-goals
5. Required implementation
6. Acceptance tests
7. Deliverables
8. Expected report format

## Codex report format

For every completed or blocked task, append a section with:
- Task ID and status: `DONE`, `PARTIAL`, or `BLOCKED`
- Commit SHA / branch
- Files changed
- Commands executed
- Tests and exact outcomes
- Quantitative results when available
- Deviations from the requested design
- Failures / blockers
- Recommended next step

## Git workflow

- Coordination files live on `main`.
- Engineering work should preferably use `codex/Txxx-short-name` branches and PRs to `main`.
- Each task should end in a self-contained commit or PR that can be reviewed independently.
- Do not mix unrelated refactors with task implementation.

## Scientific discipline

- Separate **proof of mechanism** from **full benchmark performance**.
- Never use test labels for test-time adaptation. Test labels may be used only offline for evaluation/analysis.
- Always include a no-adaptation baseline.
- Any test-time parameter update must be auditable: which parameters, which loss, how many steps, learning rate, and reset policy.
- If a proposed method is unstable, report the failure rather than hiding it with ad-hoc tuning.

## Update convention

Use UTC timestamps in coordination entries and include the latest commit SHA whenever possible.
