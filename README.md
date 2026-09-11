# TTIE

Research workspace for **Spatially Varying Test-Time Image Enhancement / ISP Adaptation**.

This repository also serves as a coordination channel between ChatGPT (research lead) and Codex (engineering lead).

## Coordination

Read these files first:

- `coordination/PROTOCOL.md` — communication and ownership rules.
- `coordination/PROJECT_STATE.md` — current research hypothesis and milestone state.
- `coordination/CHATGPT_TO_CODEX.md` — tasks/instructions from ChatGPT to Codex.
- `coordination/CODEX_TO_CHATGPT.md` — implementation reports from Codex to ChatGPT.

## Current research direction

The core hypothesis is that real visual degradation can be **spatially heterogeneous within a single image**, so a single global enhancement/ISP state may be fundamentally insufficient. We investigate a compact, differentiable, spatially varying correction field that is **adapted per test image without test labels**, while the downstream task model remains frozen.
