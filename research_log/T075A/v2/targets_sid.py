"""SID registry for the T075-A v2 gate / metrics / ceiling modules (pinned T074-C code imported unchanged).

SID-sRGB (SNR-Aware sid_processed test split) was staged by T075-B with the pinned stager (7f6ff8bc…): 598 lows,
50 scene clusters, 960x512, darkness PASS (median 0.0953, fraction <= 0.15 = 0.856). This module pins its low receipt
and reference-opaque manifest and registers SID with the 8 preregistered + 10 v2 rows, then dispatches:

    python targets_sid.py gm      {gate|metrics} ...                (v2_gate_metrics.py arguments, --target SID)
    python targets_sid.py ceiling {tuning|kappa|metrics} ...        (ceiling.py arguments, --target SID)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_gate_metrics as gm  # noqa: E402

SID_SPEC = {
    "display_name": "SID",
    "dataset": "sid",
    "geometry": "snr960x512",
    "expected_count": 598,
    "low_receipt_sha256": "676e733421e1546a7300eef1fbbed6b0174451fe8e030b28d27ffaf3f9924787",
    "reference_opaque_manifest_sha256": "287c0b18190b2f873e698f88355488d6ca8cdb2d83b8fa1de883cc2e3de8de82",
    "stage_script_sha256": "7f6ff8bc901eee6cbce987535f9197a6342442788883e6dbe1c7ae9e23c9b902",
    "required_rows": list(gm.PREREG),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    gm.rg.require(len(argv) >= 2 and argv[0] in ("gm", "ceiling"), "usage: targets_sid.py {gm|ceiling} <command> ...")
    gm.rg.require("--target" in argv and argv[argv.index("--target") + 1] == "SID", "this wrapper is for --target SID only")
    gm.register("SID", SID_SPEC)
    if argv[0] == "gm":
        if "--rows-dir" not in argv:
            argv += ["--rows-dir", str(HERE.parents[1] / "T075B" / "SID")]
        return gm.main(argv[1:])
    import ceiling

    return ceiling.main(argv[1:])


if __name__ == "__main__":
    main()
