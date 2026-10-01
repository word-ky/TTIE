"""T075 target registry (LSRW, SMID) for the unchanged T074-C gate / metrics / tuning code.

research_log/T075A/plan_and_amendments.md (user decision, relaxed darkness leg) adds LSRW and SMID as targets.
``reference_gate.py`` is deliberately NOT edited: the committed SDSD-indoor gate receipt pins its SHA256
(``gate_code_sha256``) and ``validate_receipt`` requires it to match, so any edit would invalidate that receipt.
This module adds the new targets to ``reference_gate.TARGETS`` at run time and dispatches to the unchanged
entry points:

    python targets_t075.py gate    --target LSRW --low-receipt ... --opaque-manifest ... [--rows-dir] [--out]
    python targets_t075.py metrics --target LSRW ... (metrics.py arguments)
    python targets_t075.py tuning  --target LSRW ... (ours_tuning.py arguments; run from the Ours root)

``--rows-dir`` defaults to research_log/T075B/<target>. The gate receipt additionally records this registry's
SHA256. Receipt validation stays field-by-field (required rows, low receipt / opaque manifest SHAs, per-row
freeze / manifest / verification SHAs), so a registry edit after the gate also invalidates the receipt.

Per-image geometry: LSRW is native mixed 960x640 / 960x720. The unchanged code already checks every image
against its own receipt height/width (gate manifest rows, output tensors, GT decode), never a fixed shape.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_gate as rg  # noqa: E402

STAGE_SHA256 = "7f6ff8bc901eee6cbce987535f9197a6342442788883e6dbe1c7ae9e23c9b902"
REQUIRED_ROWS = ["retinexformer", "snr_aware", "promptir", "promptir_dctta",
                 "mr_illuminate", "quadprior", "ours_step0", "ours_ttt"]
T075_TARGETS = {
    "LSRW": {
        "display_name": "LSRW",
        "dataset": "lsrw",
        "geometry": "native",
        "expected_count": 50,
        "low_receipt_sha256": "6ec25b8a13146e0759b3c9b88f1d0e1dc0de19020a8094b487ef69ff62e0a907",
        "reference_opaque_manifest_sha256": "d2f06d729c618c8e189718be0dd7edfd22f38541daa1c6377ba8f4a4abd10eb7",
        "stage_script_sha256": STAGE_SHA256,
        "required_rows": list(REQUIRED_ROWS),
    },
    "SMID": {
        "display_name": "SMID",
        "dataset": "smid",
        "geometry": "snr960x512",
        "expected_count": 1470,
        "low_receipt_sha256": "bb727e446c5a79219461838033815db01d1a7d44b998a6510e1a2f285d982656",
        "reference_opaque_manifest_sha256": "3f04de92ddd90781247deaa7087578f06ed5532b5ab8a213404875ec20915c97",
        "stage_script_sha256": STAGE_SHA256,
        "required_rows": list(REQUIRED_ROWS),
    },
}


def register():
    for name, spec in T075_TARGETS.items():
        existing = rg.TARGETS.setdefault(name, spec)
        rg.require(existing == spec, f"target {name} already registered with a different spec")


def default_rows_dir(target):
    return HERE.parent / "T075B" / target


def _with_rows_dir(argv):
    if "--rows-dir" in argv or "--target" not in argv:
        return list(argv)
    target = argv[argv.index("--target") + 1]
    return list(argv) + (["--rows-dir", str(default_rows_dir(target))] if target in T075_TARGETS else [])


def gate(argv):
    p = argparse.ArgumentParser(description="T074-C reference gate for a T075 target")
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    out = a.out or a.rows_dir / "reference_gate_receipt.json"
    rg.require(not out.exists(), f"{out} exists; the gate receipt is immutable")
    fields = rg.evaluate(a.target, a.rows_dir, a.low_receipt, a.opaque_manifest)
    fields["target_registry"] = {"module": "research_log/T074C/targets_t075.py", "sha256": rg.sha256_file(__file__)}
    receipt = rg.write_receipt(fields, out)
    print(json.dumps({"receipt": str(out), "sha256": rg.sha256_file(out), "rows": receipt["required_rows"]}, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rg.require(argv and argv[0] in ("gate", "metrics", "tuning"), "usage: targets_t075.py {gate|metrics|tuning} ...")
    register()
    command, rest = argv[0], _with_rows_dir(argv[1:])
    if command == "gate":
        return gate(rest)
    module = __import__("metrics" if command == "metrics" else "ours_tuning")
    saved = sys.argv
    sys.argv = [f"{module.__name__}.py"] + rest
    try:
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
