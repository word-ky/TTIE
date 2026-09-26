# T075-A user decisions, 2026-09-26 midday (+08)

## 1. Darkness criterion amendment (option C)

The user approved relaxing the sealed near-black criterion's second leg from "≥80% of images Y ≤ 0.15" to **"≥60% of images Y ≤ 0.15"** (median ≤ 0.10 unchanged). Disclosure: this amendment was made **after** the low-image darkness statistics of SMID (65.3%) and LSRW (72.0%) were measured, but **before any method ran on them** and without any method output or GT; it admits SMID and LSRW. SID-sRGB is added if it passes when its download succeeds. SDSD-indoor already passed.

## 2. Exploratory "Ours ranks first" reference — NOT FOR THE PAPER

The user asked for an exploratory reference showing what it would take for Ours to rank first (e.g. oracle per-image step selection with GT, wider GT-selected settings). All such runs live under `research_log/T075A/exploratory_oracle/`, are labelled `EXPLORATORY_ORACLE_NOT_FOR_PAPER`, and are never merged into main-table rows.

## 3. SSIM improvement plan (method change)

(a) Diagnose the SSIM gap on SDSD-indoor with existing frozen outputs (brightness-aligned SSIM/PSNR, output noise estimates) — analysis only. (b) Add a per-image zero-shot denoising component after the TTT rendering (only the current low image, no paired data), optionally a noise-suppressing regularizer in the TTT objective and SSIM in the tuning selection. The new method is a new row ("Ours v2"); the frozen T070-A rows stay as they are. Development may use target test GT per `research_log/T074A/ours_target_tuning_decision.md` (disclosed), with an ablation of each new component.

## 4. Main-table datasets

Run the 8 non-tuned rows on LSRW and SMID with the same frozen protocol (outputs frozen and verified before GT), then gate → metrics. SID when available.
