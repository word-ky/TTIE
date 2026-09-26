"""Synthetic tests for the T075-A v2 producers, verifier, freeze maker and gate/metrics extension (no real data)."""

import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
T074C = HERE.parents[1] / "T074C"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(T074C))
import denoise_d as dd  # noqa: E402
import verify_v2_rows as vv  # noqa: E402
import make_freeze_v2 as mf  # noqa: E402
import v2_gate_metrics as gm  # noqa: E402
import reference_gate as rg  # noqa: E402
import metrics as mt  # noqa: E402
import test_t074c as tt  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tsha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def noisy(seed=0, h=40, w=56, sigma=0.03):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.2, 0.7, w)[None, None, None, :].repeat(h, 2).repeat(3, 1)
    return torch.from_numpy((base + sigma * rng.standard_normal((1, 3, h, w))).astype(np.float32))


# ---------------------------------------------------------------- D


def test_d_quantised_deterministic_and_rule():
    x = noisy(0)
    y1, f1 = dd.apply_d(x)
    y2, f2 = dd.apply_d(x.clone())
    assert torch.equal(y1, y2) and f1 == f2
    assert y1.shape == x.shape and y1.dtype == torch.float32
    levels = torch.round(y1 * 255)
    assert torch.equal(levels / 255, y1) or torch.equal(levels.numpy().astype(np.float32) / np.float32(255), y1.numpy())
    assert f1["h"] == 6.0 * 255.0 * f1["sigma_hat"]
    assert float((y1 - x.clamp(0, 1)).abs().mean()) > 0  # it filters
    # denoising reduces the residual against the clean ramp
    clean = noisy(0, sigma=0.0).clamp(0, 1)
    assert float((y1 - clean).square().mean()) < float((x.clamp(0, 1) - clean).square().mean())


def test_d_thread_count_does_not_change_output():
    import cv2

    x = noisy(1)
    cv2.setNumThreads(8)
    y_many = dd.apply_d(x)[0]  # apply_d pins one thread itself
    assert torch.equal(y_many, dd.apply_d(x)[0])
    x8 = np.round(np.clip(x[0].permute(1, 2, 0).numpy().astype(np.float64), 0, 1)[..., ::-1] * 255).astype(np.uint8)
    h = dd.apply_d(x)[1]["h"]
    cv2.setNumThreads(8)
    a = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(x8), None, h, h, 7, 21)
    cv2.setNumThreads(1)
    b = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(x8), None, h, h, 7, 21)
    assert np.array_equal(a, b)


def test_immerkaer_estimates_white_noise_and_matches_independent_version():
    rng = np.random.default_rng(3)
    x = np.clip(0.5 + 0.02 * rng.standard_normal((200, 300, 3)), 0, 1)
    s = dd.immerkaer_sigma(x)
    assert abs(s - 0.02) < 0.001
    assert abs(vv.sigma_by_slicing(x) - s) <= 1e-9 * s


def test_d_clips_and_matches_prototype_formula():
    x = noisy(2) * 1.4 - 0.2  # values outside [0, 1]
    y, facts = dd.apply_d(x)
    xc = np.clip(x[0].permute(1, 2, 0).numpy().astype(np.float64), 0, 1)
    assert facts["sigma_hat"] == dd.immerkaer_sigma(xc)
    assert float(y.min()) >= 0 and float(y.max()) <= 1


# ---------------------------------------------------------------- producer + verifier on a synthetic target


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2")
    t = tt.build_target(tmp, name="SYNTH_V2", rows=gm.PREREG, kind="lsrw")
    return t


def produce_plus(t, source_row, method_id, source_dir=None, source_freeze=None):
    out = t.runs / method_id
    dd.produce(source_dir or t.runs / source_row, t.low_receipt, len(t.files), method_id, source_row, out, workers=2,
               source_freeze=source_freeze or t.rows_dir / source_row / "freeze_receipt.json")
    return out


def verify_and_freeze(t, row, out, source_dir, source_row, source_freeze, kind="plus_d", frozen=None):
    result = vv.verify(kind, row, t.low_receipt, out, len(t.files), source_dir, source_row, frozen)
    Path(str(out) + ".verify.json").write_text(json.dumps(result, indent=2))
    return mf.make(t.name, row, out, source_freeze, t.rows_dir / row, f"rows/{row}")


def test_producer_verifier_roundtrip_and_tamper(target, tmp_path):
    t = target
    out = produce_plus(t, "promptir", "promptir_plus_D_probe")
    man = json.loads((out / "output_manifest.json").read_bytes())
    assert man["reference_reads"] == 0 and man["metrics"] == 0 and man["count"] == len(t.files)
    assert man["source_output_manifest_sha256"] == sha(t.runs / "promptir" / "output_manifest.json")
    shapes = {tuple(r["shape"]) for r in man["rows"]}
    assert len(shapes) == 2  # mixed LSRW-like geometry preserved per image
    ok = vv.verify("plus_d", "promptir_plus_D_probe", t.low_receipt, out, len(t.files), t.runs / "promptir", "promptir")
    assert ok["classification"] == "T075A_PLUS_D_OUTPUTS_VERIFIED" and ok["recomputed_images"] == len(t.files)
    # tamper: replace one output with a different (still quantised) tensor, keep hashes -> file SHA fails
    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    stem = Path(t.files[0]["name"]).stem
    tensor, _ = dd.load_tensor(copy / stem / "output.pt.gz")
    dd.save_tensor(copy / stem / "output.pt.gz", torch.flip(tensor, [-1]))
    with pytest.raises(AssertionError, match="output file SHA"):
        vv.verify("plus_d", "promptir_plus_D_probe", t.low_receipt, copy, len(t.files), t.runs / "promptir", "promptir")
    # tamper consistently (row hashes updated): only the D re-execution catches it
    rows = json.loads((copy / "output_manifest.json").read_bytes())
    row = rows["rows"][0]
    row["output_file_sha256"] = sha(copy / stem / "output.pt.gz")
    row["output_tensor_sha256"] = tsha(torch.flip(tensor, [-1]))
    (copy / stem / "decision.json").write_text(json.dumps(row, indent=2))
    (copy / "output_manifest.json").write_text(json.dumps(rows, indent=2))
    with pytest.raises(AssertionError, match="re-execution"):
        vv.verify("plus_d", "promptir_plus_D_probe", t.low_receipt, copy, len(t.files), t.runs / "promptir", "promptir")


def test_producer_refuses_changed_source(target, tmp_path):
    t = target
    src = tmp_path / "src"
    shutil.copytree(t.runs / "retinexformer", src)
    stem = Path(t.files[1]["name"]).stem
    tensor, _ = dd.load_tensor(src / stem / "output.pt.gz")
    dd.save_tensor(src / stem / "output.pt.gz", tensor * 0.5)
    with pytest.raises(RuntimeError, match="source output file SHA"):
        dd.produce(src, t.low_receipt, len(t.files), "x_plus_D", "retinexformer", tmp_path / "o", workers=1)
    with pytest.raises(RuntimeError, match="differs from its freeze receipt"):
        dd.produce(t.runs / "promptir", t.low_receipt, len(t.files), "x", "retinexformer", tmp_path / "o2", workers=1,
                   source_freeze=t.rows_dir / "retinexformer" / "freeze_receipt.json")


def test_smoke_manifest_is_not_promotable(target, tmp_path):
    t = target
    out = tmp_path / "smoke"
    dd.produce(t.runs / "promptir", t.low_receipt, len(t.files), "promptir_plus_D", "promptir", out, workers=1, smoke_count=1)
    with pytest.raises(AssertionError):
        vv.verify("plus_d", "promptir_plus_D", t.low_receipt, out, len(t.files), t.runs / "promptir", "promptir")


def fake_knobs_row(t, out, frozen_manifest_path, abstain_first=False, execution="e" * 64):
    """Knobs-schema row (as run_ours_knobs.py writes it) with synthetic tensors (GPU TTT not run in tests)."""
    files = t.files
    rows = []
    for i, item in enumerate(files):
        tensor = noisy(10 + i, item["height"], item["width"]).clamp(0, 1)
        d = out / Path(item["name"]).stem
        d.mkdir(parents=True)
        dd.save_tensor(d / "output.pt.gz", tensor)
        abstain = abstain_first and i == 0
        row = {"method": "Ours-TTT (SDSD-selected knobs)", "low_name": item["name"], "low_sha256": item["sha256"],
               "execution_manifest_sha256": execution, "setting_id": vv.KNOBS_SETTING_ID,
               "decision_status": "TTT_ABSTAIN_NO_ACTIVE_GATE" if abstain else "TTT_EXECUTED",
               "selected_step": 0 if abstain else 20, "active": [False] * 4 if abstain else [True, False, True, True],
               "shape": [1, 3, item["height"], item["width"]], "dtype": "torch.float32",
               "output_tensor_sha256": tsha(tensor), "output_file_sha256": sha(d / "output.pt.gz"),
               "whole_run_seconds": 0.1, "peak_gpu_memory_bytes": 1}
        (d / "decision.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
    manifest = {"method": "Ours-TTT (SDSD-selected knobs)", "method_id": "ours_ttt_sdsd_knobs", "count": len(rows),
                "low_receipt_sha256": sha(t.low_receipt), "execution_manifest_sha256": execution,
                "knobs": {**vv.KNOB_OVERRIDES, "lr": 0.03}, "setting_id": vv.KNOBS_SETTING_ID,
                "knob_overrides": vv.KNOB_OVERRIDES, "frozen_ours_ttt_manifest_sha256": sha(frozen_manifest_path),
                "default_reproduction_images": 1, "no_active_abstentions": int(abstain_first),
                "producer_sha256": "p" * 64, "promotable": True, "reference_reads": 0, "metrics": 0, "rows": rows}
    (out / "output_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def test_knobs_verifier(target, tmp_path):
    t = target
    frozen = tmp_path / "frozen_ttt"
    frozen.mkdir()
    (frozen / "output_manifest.json").write_text(json.dumps({"execution_manifest_sha256": "e" * 64}))
    out = tmp_path / "knobs"
    fake_knobs_row(t, out, frozen / "output_manifest.json", abstain_first=True)
    r = vv.verify("knobs", "ours_ttt_sdsd_knobs", t.low_receipt, out, len(t.files), frozen_ours_ttt=frozen)
    assert r["no_active_abstentions"] == 1 and r["classification"] == "T075A_KNOBS_OUTPUTS_VERIFIED"
    (frozen / "output_manifest.json").write_text(json.dumps({"execution_manifest_sha256": "f" * 64}))
    with pytest.raises(AssertionError):
        vv.verify("knobs", "ours_ttt_sdsd_knobs", t.low_receipt, out, len(t.files), frozen_ours_ttt=frozen)


# ---------------------------------------------------------------- gate + metrics extension


@pytest.fixture(scope="module")
def v2_target(target):
    t = target
    fr = lambda r: t.rows_dir / r / "freeze_receipt.json"  # noqa: E731
    knobs_out = t.runs / "ours_ttt_sdsd_knobs"
    # the build_target ours_ttt manifest has no execution manifest field; the knobs row binds to ours_ttt by
    # manifest SHA (checked by the freeze maker and the gate), the verifier's execution-manifest check is covered above
    fake_knobs_row(t, knobs_out, t.runs / "ours_ttt" / "output_manifest.json")
    ver = {"classification": "T075A_KNOBS_OUTPUTS_VERIFIED", "method_id": "ours_ttt_sdsd_knobs", "count": len(t.files),
           "output_manifest_sha256": sha(knobs_out / "output_manifest.json"), "low_receipt_sha256": sha(t.low_receipt),
           "compressed_output_bytes": 1, "no_active_abstentions": 0, "ttt_executed_count": len(t.files),
           "verifier_sha256": sha(vv.__file__), "reference_reads": 0, "metrics": 0}
    Path(str(knobs_out) + ".verify.json").write_text(json.dumps(ver, indent=2))
    mf.make(t.name, "ours_ttt_sdsd_knobs", knobs_out, fr("ours_ttt"), t.rows_dir / "ours_ttt_sdsd_knobs", "rows/k")
    for row, source in gm.SOURCES.items():
        if row == "ours_ttt_sdsd_knobs":
            continue
        out = produce_plus(t, source, row, t.runs / source, fr(source))
        verify_and_freeze(t, row, out, t.runs / source, source, fr(source))
    base = rg.TARGETS.pop(t.name)
    gm.register(t.name, base)
    return t


def common(t):
    return ["--target", t.name, "--low-receipt", str(t.low_receipt), "--opaque-manifest", str(t.opaque),
            "--rows-dir", str(t.rows_dir)]


def test_gate_binds_all_v2_rows_and_metrics_run(v2_target, tmp_path):
    t = v2_target
    receipt = gm.main(["gate"] + common(t))
    assert receipt["required_rows"] == gm.PREREG + gm.EXTRA_ROWS and len(receipt["required_rows"]) == 18
    assert receipt["target_registry"]["v2_sha256"] == sha(gm.__file__)
    with pytest.raises(rg.GateError, match="immutable"):
        gm.main(["gate"] + common(t))
    result = gm.main(["metrics"] + common(t) + ["--gate-receipt", str(t.receipt), "--stage-target", str(tt.STAGE_PATH),
                                                "--out", str(tmp_path / "m")])
    assert set(result["rows"]) == set(gm.PREREG + gm.EXTRA_ROWS) and result["development_after_gt"] is False
    dirs = {c["direction"] for c in result["comparisons"]}
    for v in gm.V2_ROWS:
        assert all(f"{v} - {b}" in dirs and f"{v} - {b}_plus_D" in dirs for b in gm.BASELINES)
        assert f"{v} - ours_ttt" in dirs
    assert len(result["comparisons"]) == 2 * 13 + 2
    # per-image value equals a direct frozen call on the stored v2 tensor
    core = mt.import_frozen_metric()
    item = t.files[0]
    man = json.loads((t.runs / "ours_v2" / "output_manifest.json").read_bytes())
    out = mt.load_output(t.runs / "ours_v2" / Path(item["name"]).stem / "output.pt.gz", man["rows"][0], item)
    ctx = mt.Context(t.name, t.low_receipt, t.opaque, t.receipt, t.rows_dir, tt.STAGE_PATH)
    direct = mt.score(out, ctx.reference(0), core)
    assert result["per_image"]["ours_v2"][0]["psnr"] == direct["psnr"]
    assert result["per_image"]["ours_v2"][0]["rgb_ssim"] == direct["rgb_ssim"]


def test_gate_refuses_misbound_derived_row(v2_target, tmp_path):
    t = v2_target
    rows = tmp_path / "rows"
    shutil.copytree(t.rows_dir, rows)
    (rows / "reference_gate_receipt.json").unlink(missing_ok=True)
    freeze_path = rows / "quadprior_plus_D" / "freeze_receipt.json"
    freeze = json.loads(freeze_path.read_bytes())
    freeze["source_output_manifest_sha256"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze, indent=2))
    with pytest.raises(rg.GateError, match="quadprior_plus_D: freeze not bound"):
        gm.main(["gate", "--target", t.name, "--low-receipt", str(t.low_receipt), "--opaque-manifest", str(t.opaque),
                 "--rows-dir", str(rows)])


def test_gate_refuses_missing_v2_row(v2_target, tmp_path):
    t = v2_target
    rows = tmp_path / "rows"
    shutil.copytree(t.rows_dir, rows)
    (rows / "reference_gate_receipt.json").unlink(missing_ok=True)
    shutil.rmtree(rows / "snr_aware_plus_D")
    with pytest.raises(rg.GateError, match="snr_aware_plus_D: required row not frozen"):
        gm.main(["gate", "--target", t.name, "--low-receipt", str(t.low_receipt), "--opaque-manifest", str(t.opaque),
                 "--rows-dir", str(rows)])


def test_registry_refuses_conflicting_spec():
    with pytest.raises(rg.GateError):
        gm.register("LSRW", dict(rg.TARGETS.get("SDSD_indoor"), required_rows=["x"]))


# ---------------------------------------------------------------- declared knobs-rejection fallback


def fake_frozen_ttt(t, root):
    rows = []
    for i, item in enumerate(t.files):
        tensor = noisy(50 + i, item["height"], item["width"]).clamp(0, 1)
        d = root / Path(item["name"]).stem
        d.mkdir(parents=True)
        dd.save_tensor(d / "output.pt.gz", tensor)
        rows.append({"low_name": item["name"], "low_sha256": item["sha256"], "shape": [1, 3, item["height"], item["width"]],
                     "dtype": "torch.float32", "decision_status": "TTT_EXECUTED", "decision": {"selected_step": 19},
                     "execution_manifest_sha256": "e" * 64, "output_tensor_sha256": tsha(tensor),
                     "output_file_sha256": sha(d / "output.pt.gz")})
    (root / "output_manifest.json").write_text(json.dumps({"execution_manifest_sha256": "e" * 64, "rows": rows}))
    return rows


def test_knobs_rejection_fallback_is_bitwise_frozen_output(target, tmp_path):
    import run_ours_knobs as rk

    t = target
    frozen_dir = tmp_path / "frozen_ttt"
    frozen_rows = fake_frozen_ttt(t, frozen_dir)
    out = tmp_path / "knobs"
    manifest = fake_knobs_row(t, out, frozen_dir / "output_manifest.json")
    item, d = t.files[0], out / Path(t.files[0]["name"]).stem
    (d / "output.pt.gz").unlink()
    image, extra = rk.fallback_copy(frozen_rows[0], frozen_dir, item, d)
    assert tsha(image) == frozen_rows[0]["output_tensor_sha256"]
    row = dict(manifest["rows"][0], decision_status=rk.FALLBACK, selected_step=19, rejection_error="AssertionError()",
               output_tensor_sha256=tsha(image), output_file_sha256=sha(d / "output.pt.gz"), **extra)
    manifest["rows"][0] = row
    manifest["knobs_rejected_fallback_count"] = 1
    (d / "decision.json").write_text(json.dumps(row, indent=2))
    (out / "output_manifest.json").write_text(json.dumps(manifest, indent=2))
    r = vv.verify("knobs", "ours_ttt_sdsd_knobs", t.low_receipt, out, len(t.files), frozen_ours_ttt=frozen_dir)
    assert r["knobs_rejected_fallback_count"] == 1 and r["ttt_executed_count"] == len(t.files) - 1
    # a fallback row that is not the frozen output fails, even with self-consistent hashes
    other = noisy(99, item["height"], item["width"]).clamp(0, 1)
    dd.save_tensor(d / "output.pt.gz", other)
    row = dict(row, output_tensor_sha256=tsha(other), output_file_sha256=sha(d / "output.pt.gz"))
    manifest["rows"][0] = row
    (d / "decision.json").write_text(json.dumps(row, indent=2))
    (out / "output_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pytest.raises(AssertionError, match="fallback row not the frozen output"):
        vv.verify("knobs", "ours_ttt_sdsd_knobs", t.low_receipt, out, len(t.files), frozen_ours_ttt=frozen_dir)
    # undeclared count fails
    manifest["knobs_rejected_fallback_count"] = 0
    (out / "output_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pytest.raises(AssertionError):
        vv.verify("knobs", "ours_ttt_sdsd_knobs", t.low_receipt, out, len(t.files), frozen_ours_ttt=frozen_dir)
    # the producer refuses a changed frozen source
    dd.save_tensor(frozen_dir / Path(t.files[1]["name"]).stem / "output.pt.gz", other)
    (tmp_path / "x").mkdir()
    with pytest.raises(RuntimeError, match="frozen ours_ttt file changed"):
        rk.fallback_copy(frozen_rows[1], frozen_dir, t.files[1], tmp_path / "x")
