"""Derive dataset-generic runners from the frozen UHD-LL runners by declared text substitutions.

Every change to an official/frozen runner is listed here; nothing else differs.
``test_make_generic.py`` re-derives each generic file and requires byte equality,
so any undeclared edit fails. Substitutions only replace the UHD-LL constants
(count 150, geometry 2160x3840, pinned Step0 manifest) by receipt-derived values
and make the 4K memory schedules opt-in (off = official unscheduled forward).
"""

from pathlib import Path

import os

HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get("T074B_ORIGINALS_ROOT", HERE.parent))  # holds T073B/ and T073C/

PATH_HEADER = '''import os as _os
import sys as _sys

# T074-B: frozen UHD-LL helper modules (T073B/shared on the GPU host).
_sys.path.insert(0, _os.environ.get("T073B_SHARED", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "T073B")))
'''

OURS_HEADER = '''import os as _os
import sys as _sys

# T074-B: run from the frozen Ours source root (FinalOurs binds cwd-relative paths).
_sys.path.insert(0, _os.getcwd())
'''

CROP16 = ('    assert all(i["height"] % 16 == 0 and i["width"] % 16 == 0 for i in expected), '
          '"official PromptIR test loader would center-crop to multiples of 16"\n')
SHAPE_ROW ='"shape": [1, 3, item["height"], item["width"]],'
SCHEDULE_ARG = ('    parser.add_argument("--memory-schedule", action="store_true",\n'
                '                        help="UHD-LL 4K schedule; default is the official unscheduled forward")\n')
COUNT_ARG = '    parser.add_argument("--expected-count", type=int, required=True)\n'

SPECS = {
    "run_promptir_static_generic.py": ("T073B/run_static_native_batch.py", [
        ('"""Full native static PromptIR low-only outputs with the sealed memory schedule."""\n',
         '"""T074-B generic static PromptIR low-only outputs (derived from T073B; see make_generic.py)."""\n\n'
         + PATH_HEADER),
        ('    parser.add_argument("--smoke-one", action="store_true")\n',
         '    parser.add_argument("--smoke-one", action="store_true")\n' + COUNT_ARG + SCHEDULE_ARG),
        ('    assert len(expected) == 150\n', '    assert len(expected) == args.expected_count\n' + CROP16),
        ('    schedule_depthwise(model, rows=256, threshold=250_000_000)\n'
         '    schedule_feedforward(model, rows=64, threshold=250_000_000)\n'
         '    schedule_skip_spill(model, threshold=1_000_000)\n',
         '    if args.memory_schedule:\n'
         '        schedule_depthwise(model, rows=256, threshold=250_000_000)\n'
         '        schedule_feedforward(model, rows=64, threshold=250_000_000)\n'
         '        schedule_skip_spill(model, threshold=1_000_000)\n'),
        ('        assert low.shape == (3, 2160, 3840)\n',
         '        assert low.shape == (3, item["height"], item["width"])\n'),
        ('        assert output.shape == (1, 3, 2160, 3840)\n',
         '        assert output.shape == (1, 3, item["height"], item["width"])\n'),
        ('"shape": [1, 3, 2160, 3840],', SHAPE_ROW),
        ('"method": "PromptIR-static-five-task-native-schedule",',
         '"method": "PromptIR-static-five-task-generic",\n'
         '        "memory_schedule": bool(args.memory_schedule),'),
    ]),
    "run_dctta_generic.py": ("T073B/run_dctta_native_batch.py", [
        ('No reference path\nis accepted.\n"""\n',
         'No reference path\nis accepted.\n\nT074-B generic variant derived from T073B; see make_generic.py.\n"""\n\n'
         + PATH_HEADER),
        ('    parser.add_argument("--num-workers", type=int, default=16)\n',
         '    parser.add_argument("--num-workers", type=int, default=16)\n' + COUNT_ARG + SCHEDULE_ARG),
        ('    assert len(receipt) == EXPECTED_COUNT\n',
         '    assert len(receipt) == args.expected_count\n' + CROP16.replace("in expected", "in receipt")),
        ('    scheduled_depthwise = schedule_depthwise(model, rows=256, threshold=250_000_000)\n'
         '    scheduled_feedforward = schedule_feedforward(model, rows=64, threshold=250_000_000)\n'
         '    schedule_skip_spill(model, threshold=1_000_000)\n',
         '    scheduled_depthwise, scheduled_feedforward = [], []\n'
         '    if args.memory_schedule:\n'
         '        scheduled_depthwise = schedule_depthwise(model, rows=256, threshold=250_000_000)\n'
         '        scheduled_feedforward = schedule_feedforward(model, rows=64, threshold=250_000_000)\n'
         '        schedule_skip_spill(model, threshold=1_000_000)\n'),
        ('        assert low.shape == (3, 2160, 3840)\n',
         '        assert low.shape == (3, item["height"], item["width"])\n'),
        ('        assert output.shape == (1, 3, 2160, 3840)\n',
         '        assert output.shape == (1, 3, item["height"], item["width"])\n'),
        ('"shape": [1, 3, 2160, 3840],', SHAPE_ROW),
        ('method="PromptIR+DCTTA-five-task-domain-level-native-schedule"',
         'memory_schedule=bool(args.memory_schedule),\n'
         '        method="PromptIR+DCTTA-five-task-domain-level-generic"'),
    ]),
    "run_ours_step0_generic.py": ("T073C/run_step0_batch.py", [
        ('"""Run frozen zero-update Ours on the preregistered UHD-LL low-only cohort."""\n',
         '"""T074-B generic frozen zero-update Ours-Step0 (derived from T073C; see make_generic.py)."""\n\n'
         + OURS_HEADER),
        ('    parser.add_argument("--smoke-one", action="store_true")\n',
         '    parser.add_argument("--smoke-one", action="store_true")\n' + COUNT_ARG),
        ('    assert len(expected) == 150\n', '    assert len(expected) == args.expected_count\n'),
        ('        rows.append(record)\n',
         '        assert record["shape"] == [1, 3, item["height"], item["width"]]\n'
         '        rows.append(record)\n'),
        ('        "rows": rows,\n',
         '        "reference_reads": 0,\n        "metrics": 0,\n        "rows": rows,\n'),
    ]),
    "run_ours_ttt_abstain_generic.py": ("T073C/run_ours_ttt_abstain_batch.py", [
        ('"""Complete-case Ours-TTT with the prospectively sealed no-active rule."""\n',
         '"""T074-B generic Ours-TTT with the sealed no-active rule (derived from T073C; see make_generic.py)."""\n\n'
         + OURS_HEADER),
        ('    assert list(step0_tensor.shape) == [1, 3, 2160, 3840]\n',
         '    assert list(step0_tensor.shape) == [1, 3, expected["height"], expected["width"]]\n'),
        ('    parser.add_argument("--smoke-count", type=int, default=0)\n',
         '    parser.add_argument("--smoke-count", type=int, default=0)\n'
         '    parser.add_argument("--expected-step0-manifest-sha256", required=True)\n'),
        ('    assert len(expected) == frozen_step0["count"] == 150\n',
         '    assert len(expected) == frozen_step0["count"] == len(frozen_step0["rows"])\n'),
        ('    assert step0_manifest_hash == "76a1ee7b12f41991499802024811321dcce3b0af0bae2c06ab41dba719180cce"\n',
         '    assert step0_manifest_hash == args.expected_step0_manifest_sha256\n'),
    ]),
}


def derive(name):
    source, substitutions = SPECS[name]
    text = (LOG / source).read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in substitutions:
        assert text.count(old) == 1, f"{name}: substitution anchor not unique: {old[:60]!r}"
        text = text.replace(old, new)
    return text


def main():
    for name in SPECS:
        (HERE / name).write_bytes(derive(name).encode("utf-8"))
        print("wrote", name)


if __name__ == "__main__":
    main()
