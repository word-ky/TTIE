# T060-D-R2 — DONE

**Classification: T014-selection mismatch is not a sufficient explanation for the T060-C-R1 near-miss**

This is a fixed source-only trajectory/selector diagnostic. No deployable, target-domain or SOTA claim follows from source oracle states. Exact60 T060-B anchors/order and state-0 degraded inputs; no clean/reference gradient or outcome enters inference/selection.

Authorization a53d9901b2c48aef7f6c0877928817d4ede1829b; tested source 2f83e7bda8b7be64f1a520d4ad3afc144ad503be; branch codex/T060DR2-selector-audit. Initial release 20260919-ttie-t060dr2-selector, run 20260919-180757-ttie-t060dr2-selector; storage-only resume release 20260919-ttie-t060dr2-selector-resume, run 20260919-181949-ttie-t060dr2-resume, source e7f71d43ed72e8c5d6f04eb497373db40dc5ad66; physical NVIDIA RTX A6000GPU1. All273 bound source files match published Git bytes. Preflight source committed before execution; trajectory source committed/pushed before the initial attempt; recovery wrapper committed/pushed before continuation.

## Fixed implementation and reuse

A calls the literal accepted ttie.common_gain_ttt.trajectory. B calls the accepted research_log.T060CR1.core.trajectory, which changes only four gain-gradient coordinates. Both start from raw-low/identity, optimize all12 raw coordinates using fresh Adam lr0.03 for40 steps with unchanged CommonBox, and select minimum T014 scalar energy with earliest tie. A uses T014 for all gradients; B uses T014 EV/gamma and fresh current-image J_gain-transpose-q_E for gain. E scalar never selects a state. Source frozen receipt/calibration is reused for both; no target-domain gate/input is loaded. All41 rendered outputs and states, gradients, feature/scores, decisions and selected output are persisted for each trajectory.

The R2 preflight checks exact input/model/config/head bindings, exact active/winner, primitive score max error<=1e-5, and fresh-vs-frozen pre-reference E gain cosine>=0.999/relativeL2<=0.01. Evidence is reconstructed from unchanged calibration and recorded descriptively. Preflight statistics:

```json
{
  "rows": 60,
  "max_score_error": 1.6316771507263184e-06,
  "max_evidence_drift": 2.820755302046507e-05,
  "min_gain_cosine": 0.9999999997461764,
  "max_gain_relative_l2": 2.2650691345235727e-05,
  "zero_vectors": 0
}
```

## Frozen scientific verdict

```json
{
  "metrics": {
    "selected_delta": {
      "mean": 0.41350123567445796,
      "median": 0.22087623057871308,
      "p10": -1.8752552242182292,
      "p90": 3.986388067519122,
      "min": -5.732418453265733,
      "max": 5.958918050419182
    },
    "oracle_delta": {
      "mean": 0.5054509680637912,
      "median": 0.0027965773199136734,
      "p10": -0.9176126863909554,
      "p90": 2.6959808385174857,
      "min": -5.355919613505069,
      "max": 7.071636317727865
    },
    "regret_delta": {
      "mean": 0.09194973238933339,
      "median": 0.03368526546339368,
      "p10": -2.4114192881421594,
      "p90": 2.9166389551639598,
      "min": -9.462672659585017,
      "max": 9.095472442817607
    }
  },
  "counts": {
    "selected_delta": {
      "win": 37,
      "equal": 0,
      "loss": 23
    },
    "oracle_delta": {
      "win": 35,
      "equal": 1,
      "loss": 24
    },
    "regret_delta": {
      "win": 31,
      "equal": 0,
      "loss": 29
    }
  },
  "gates": {
    "oracle_mean": true,
    "oracle_median": true,
    "oracle_wins": false,
    "regret_mean": false
  },
  "absolute_means": {
    "A": {
      "selected_psnr": 14.921295312495477,
      "oracle_psnr": 21.55865906260983,
      "regret": 6.6373637501143525
    },
    "B": {
      "selected_psnr": 15.334796548169935,
      "oracle_psnr": 22.06411003067362,
      "regret": 6.729313482503686
    },
    "no_adaptation_identity": {
      "psnr": 15.179924832196827
    }
  }
}
```

Original four gates unchanged: mean B-minus-A oracle PSNR>=0.15dB, median>0, at least36/60 oracle wins, and mean B-minus-A selection regret>=0.10dB. Exact-zero paired differences count as ties. Oracle-best states use maximum frozen-state PSNR and earliest tie, only in the separate source-reference diagnostic; neither inference nor the T014 selector sees these values.

## Freeze, verification and outputs

Inference start 2026-09-19T10:08:04+00:00; all120 trajectories freeze 2026-09-19T10:24:38.693047+00:00; first source-clean stage 2026-09-19T10:24:51.473283+00:00; diagnostic ends 2026-09-19T10:25:08.267325+00:00; independent verifier ends 2026-09-19T10:25:50.617265+00:00. The retained120trajectories contain4800 optimizer updates. One additional failed-save B trajectory was computed before ENOSPC, preserved partially and recomputed on resume; no completed trajectory was rerun. All computation precedes the global freeze. Beforefreeze source-clean/reference-gradient/metric/target-domain/official-test reads0. Source-clean images opened afterfreeze: 16. Head and bound input/source hashes unchanged.

Tests: 7 passed4.97s; prior7 passed5.04s; preflight focused1 passed1.39s. The fixture compares literal A states exactly, checks the authorized B coordinate replacement and T014-only selection, and independently replays both Adam/CommonBox updates. Existing B direct-gradient/firewall and baseline tests pass. End-to-end independent verifier PASS on60anchors/120trajectories: every selected step/state/output, all41 fresh rendered images, T014 scalar scores, full NumPy Adam/CommonBox replays, B J-transpose-q and analytical normalized MLP q; fresh direct T014/E autograd at indices0/15/30/45/59 and steps0/20/39; independent full-RGB NumPy MSE/PSNR for all4920states, oracle ties/steps, regrets, distributions/counts and final gates. Max errors: Adam 3.26154594332273e-07; direct E gain 7.748603820800781e-06; PSNR 6.572520305780927e-13dB. Replay tolerances were bound before experiments.

Commands: python -m pytest -p no:cacheprovider --import-mode=importlib research_log/T060DR2/test_preflight.py research_log/T060DR2/test_trajectory.py research_log/T060CR1/test_contract.py tests/test_common_gain_ttt.py -q; then separate python -m research_log.T060DR2.infer, diagnose, verify processes with --out $AUTODL_ARTIFACTS_DIR/T060DR2. Full run.sh/meta/train.log retained in recovery archive. Environment CUDA_VISIBLE_DEVICES=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, seed7, TF32off, OMP/MKL/OPENBLAS threads1. E feature-gradient keeps its accepted CPU path; image/CLIP/trajectory and direct replay run on GPU.

Preparation-only Windows SSH argument-length error was fixed by reusing the exact already-remote prediction-freeze manifest. The initial inference hit home ENOSPC after34complete pairs and the next A trajectory;350files were copied to F and verified before replacing the home run path with a symlink. The incomplete B artifact is preserved. A narrow committed resume reuses all69complete trajectories and computes only the missing51; no clean reference was opened before recovery. The resumed global-start field records the original run-wrapper start timestamp; resume_started.json separately records restart time. The reused extra A elapsed time was unavailable, recorded as0 with an explicit note. No scientific setting changed. No tuning or alternative method was introduced.

Result SHA f246802e7086257b9e40586509e55db09b9ffcf30ec1a3fcad19dec49805d45e. Raw/recovery archives include exact source and preflight prediction records. Due to observed home disk exhaustion, the large raw archive physically resides on F with a home symlink (one physical copy); compact recovery has verified physical copies on home and F:

```json
{
  "raw": {
    "home": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t060dr2/T060DR2_raw.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t060dr2/T060DR2_raw.tar.gz",
    "bytes": 6073677963,
    "sha256": "a1d99dfa11205b28188e4631c502fba2e7663ef2dc92e479ff30d97dbb1aa015",
    "home_is_symlink": true,
    "physical_copies": 1
  },
  "recovery": {
    "home": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t060dr2/T060DR2_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t060dr2/T060DR2_recovery.tar.gz",
    "bytes": 6645681,
    "sha256": "0e25e8e8cdbf75201e3a3ab870f6729ca27d673b8076cccb5552a4a33f12b486",
    "home_is_symlink": false,
    "physical_copies": 2
  }
}
```

Next: research-lead review under the preregistered verdict. No selector fitting, target-development rerun, official-test access, threshold modification or self-merge.
