"""Synthetic tests for the T074-C reference gate, metrics and phase-4 harness (no real target data)."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
OURS_SRC = Path(os.environ.get("OURS_SRC", "/root/autodl-tmp/TTIE/T073C/recovery/source"))
if (OURS_SRC / "ttie" / "common_gain.py").is_file():
    # As in production (ours_tuning runs from the Ours root): the Ours ttie package must win over the
    # vendored metric-only ttie copy, whose ssim_transfer.py / __init__.py bytes are identical.
    sys.path.insert(0, str(OURS_SRC))
sys.path.insert(0, str(HERE))
import reference_gate as rg  # noqa: E402
import metrics as mt  # noqa: E402

STAGE_PATH = HERE.parent / "T074B" / "stage_target.py"
STAGE_SHA = rg.TARGETS["SDSD_indoor"]["stage_script_sha256"]
ROWS = ["retinexformer", "promptir", "promptir_dctta", "ours_step0", "ours_ttt"]
T073A_INDICES_SHA256 = "f3348c731c348b52eed32160d6b4b2b904101e59a542e1c5dc22850e812da904"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stage():
    spec = importlib.util.spec_from_file_location("test_stage_target", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
    return sha(path)


def save_output(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)


KINDS = {"sdsd": ("sdsd_in", "snr960x512"), "lsrw": ("lsrw", "native"), "smid": ("smid", "snr960x512")}
LSRW_SIZES = {"Huawei": (48, 72), "Nikon": (64, 72)}  # mixed native geometry, like 960x720 / 960x640


def make_data(stage, kind, root, frames, rng):
    """Synthetic source layout for the real stager; returns (count, smid test list or None)."""
    if kind == "sdsd":
        for video in stage.SDSD_TEST_DIRS["sdsd_in"]:
            for index in range(frames):
                low = rng.integers(0, 40, (48, 90, 3), dtype=np.uint8)
                gt = rng.integers(0, 256, (48, 90, 3), dtype=np.uint8)
                (root / "input" / video).mkdir(parents=True, exist_ok=True)
                (root / "GT" / video).mkdir(parents=True, exist_ok=True)
                np.save(root / "input" / video / f"{index:04d}.npy", low)
                np.save(root / "GT" / video / f"{index + 7:04d}.npy", gt)
        return 6 * frames, None
    if kind == "lsrw":
        from PIL import Image

        count = 0
        for camera, (h, w) in LSRW_SIZES.items():
            for side in ("low", "high"):
                (root / camera / side).mkdir(parents=True, exist_ok=True)
            for index in range(3 if camera == "Huawei" else 2):
                Image.fromarray(rng.integers(0, 40, (h, w, 3), dtype=np.uint8)).save(root / camera / "low" / f"{index:03d}.jpg")
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(root / camera / "high" / f"{index:03d}.jpg")
                count += 1
        return count, None
    sequences = ["0003", "0011", "0027"]
    for seq in sequences:
        (root / "SMID_LQ_np" / seq).mkdir(parents=True)
        (root / "SMID_Long_np" / seq).mkdir(parents=True)
        for index in range(frames):
            np.save(root / "SMID_LQ_np" / seq / f"{index + 1:04d}.npy", rng.integers(0, 40, (40, 60, 3), dtype=np.uint8))
        np.save(root / "SMID_Long_np" / seq / "0001_5.npy", rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
    test_list = root.parent / "test_list.txt"
    test_list.write_text("".join(f"{seq}\n" for seq in sequences))
    return len(sequences) * frames, test_list


def build_target(tmp, name="SYNTH", rows=ROWS, frames=2, seed=0, kind="sdsd"):
    """Synthetic target staged with the real stager (SDSD-, LSRW- or SMID-like), plus frozen-looking rows."""
    stage = load_stage()
    rng = np.random.default_rng(seed)
    root = tmp / "data"
    dataset, geometry = KINDS[kind]
    count, test_list = make_data(stage, kind, root, frames, rng)
    staged = tmp / "staged"
    stage.stage(argparse.Namespace(dataset=dataset, root=root, out=staged, geometry=geometry,
                                   frames_per_sequence=30 if kind == "smid" else 0, smid_test_list=test_list,
                                   expected_count=count, list_only=False))
    low_receipt, opaque = staged / "low_receipt.json", staged / "reference_opaque_manifest.json"
    files = json.loads(low_receipt.read_bytes())["files"]
    low_sha = sha(low_receipt)
    rows_dir, runs = tmp / "rows", tmp / "runs"
    manifest_shas = {}
    for row_id in rows:
        remote = runs / row_id
        manifest = {"method": row_id, "count": len(files), "low_receipt_sha256": low_sha,
                    "reference_reads": 0, "metrics": 0, "rows": []}
        for item in files:
            tile = rng.uniform(-0.05, 1.05, (1, 3, 16, 24)).astype(np.float32)  # tiled: fast gzip, still varied
            h, w = item["height"], item["width"]
            tensor = torch.from_numpy(np.ascontiguousarray(np.tile(tile, (1, 1, h // 16, w // 24))))
            path = remote / Path(item["name"]).stem / "output.pt.gz"
            save_output(path, tensor)
            manifest["rows"].append({"low_name": item["name"], "low_sha256": item["sha256"], "shape": [1, 3, h, w],
                                     "dtype": "torch.float32", "output_file_sha256": sha(path),
                                     "output_tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
        freeze = {"task": "T074-C", "method_id": row_id, "target": name, "classification": "FROZEN_OUTPUTS",
                  "output_count": len(files), "canonical_low_receipt_sha256": low_sha,
                  "verification_classification": f"T074B_{row_id.upper()}_OUTPUTS_VERIFIED",
                  "remote_output_root": str(remote), "target_reference_reads": 0, "target_metrics": 0,
                  "preregistration_changed": False, "quality_inspection_of_outputs": False}
        if row_id == "ours_ttt":
            manifest["step0_output_manifest_sha256"] = manifest_shas["ours_step0"]
            freeze["expected_step0_manifest_sha256"] = manifest_shas["ours_step0"]
            freeze["ttt_abstain_no_active_gate_count"] = 0
        if row_id == "promptir_dctta":
            order_sha = write_json(rows_dir / row_id / "adaptation_order.json", {"order": [f["name"] for f in files]})
            manifest["expected_order_sha256"] = freeze["sealed_adaptation_order_sha256"] = order_sha
        msha = write_json(remote / "output_manifest.json", manifest)
        (rows_dir / row_id).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / "output_manifest.json", rows_dir / row_id / "output_manifest.json")
        manifest_shas[row_id] = msha
        verification = {"classification": freeze["verification_classification"], "count": len(files),
                        "output_manifest_sha256": msha, "low_receipt_sha256": low_sha, "reference_reads": 0, "metrics": 0}
        vsha = write_json(Path(str(remote) + ".verify.json"), verification)
        shutil.copyfile(Path(str(remote) + ".verify.json"), rows_dir / row_id / "verification.json")
        write_json(rows_dir / row_id / "freeze_receipt.json",
                   dict(freeze, output_manifest_sha256=msha, independent_verification_sha256=vsha))
    rg.TARGETS[name] = {"display_name": name, "dataset": dataset, "geometry": geometry,
                        "expected_count": len(files), "low_receipt_sha256": low_sha,
                        "reference_opaque_manifest_sha256": sha(opaque), "stage_script_sha256": STAGE_SHA,
                        "required_rows": list(rows)}
    return SimpleNamespace(name=name, root=root, rows_dir=rows_dir, runs=runs, low_receipt=low_receipt, opaque=opaque,
                           files=files, receipt=rows_dir / "reference_gate_receipt.json", tmp=tmp)


def open_gate(t):
    return rg.write_receipt(rg.evaluate(t.name, t.rows_dir, t.low_receipt, t.opaque), t.receipt)


def context(t):
    return mt.Context(t.name, t.low_receipt, t.opaque, t.receipt, t.rows_dir, STAGE_PATH)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    t = build_target(tmp_path_factory.mktemp("t074c"), name="SYNTH_EVAL")
    open_gate(t)
    ctx = context(t)
    return t, ctx, mt.evaluate(ctx)


# ---------------------------------------------------------------- gate


def test_gate_opens_once_and_receipt_is_immutable(tmp_path):
    t = build_target(tmp_path, name="SYNTH_GATE")
    receipt = open_gate(t)
    assert receipt["classification"] == rg.CLASSIFICATION and receipt["required_rows"] == ROWS
    assert receipt["low_receipt_sha256"] == sha(t.low_receipt)
    assert receipt["reference_opaque_manifest_sha256"] == sha(t.opaque)
    assert receipt["rows"]["ours_ttt"]["output_manifest_sha256"] == sha(t.rows_dir / "ours_ttt" / "output_manifest.json")
    assert stat.S_IMODE(os.stat(t.receipt).st_mode) == 0o444
    with pytest.raises(FileExistsError):
        open_gate(t)
    rg.validate_receipt(t.receipt, t.name, t.rows_dir, t.low_receipt, t.opaque)


def test_gate_refuses_missing_required_row(tmp_path):
    t = build_target(tmp_path, name="SYNTH_MISSING")
    write_json(t.rows_dir / "mr_illuminate" / "status.json", {"classification": "PENDING_NOT_RUN"})
    rg.TARGETS[t.name]["required_rows"].insert(4, "mr_illuminate")
    with pytest.raises(rg.GateError, match="mr_illuminate: .*PENDING_NOT_RUN"):
        open_gate(t)
    assert not t.receipt.exists()


@pytest.mark.parametrize("victim", ["local_manifest", "local_verification", "remote_manifest", "remote_output",
                                    "freeze_classification", "opaque", "step0_binding"])
def test_gate_refuses_tampering(tmp_path, victim):
    t = build_target(tmp_path, name=f"SYNTH_TAMPER_{victim}")
    if victim == "local_manifest":
        path = t.rows_dir / "promptir" / "output_manifest.json"
        path.write_text(path.read_text().replace('"metrics": 0', '"metrics": 0 '))
    elif victim == "local_verification":
        path = t.rows_dir / "retinexformer" / "verification.json"
        path.write_text(path.read_text() + " ")
    elif victim == "remote_manifest":
        path = t.runs / "ours_step0" / "output_manifest.json"
        path.write_text(path.read_text() + "\n")
    elif victim == "remote_output":
        path = t.runs / "promptir_dctta" / Path(t.files[3]["name"]).stem / "output.pt.gz"
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(bytes(raw))
    elif victim == "freeze_classification":
        path = t.rows_dir / "ours_ttt" / "freeze_receipt.json"
        path.write_text(path.read_text().replace("FROZEN_OUTPUTS", "SMOKE_OUTPUTS"))
    elif victim == "opaque":
        t.opaque.write_text(t.opaque.read_text().replace('"reference_decodes": 0', '"reference_decodes": 1'))
    elif victim == "step0_binding":
        path = t.rows_dir / "ours_ttt" / "freeze_receipt.json"
        data = json.loads(path.read_bytes())
        data["expected_step0_manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    with pytest.raises(rg.GateError):
        open_gate(t)
    assert not t.receipt.exists()


def reseal_verification(t, row_id, mutate):
    """Rewrite a row's verification consistently (local + remote copy + freeze hashes/classification), so only
    the 'verification must have passed' checks can catch the mutation."""
    local = t.rows_dir / row_id / "verification.json"
    data = json.loads(local.read_bytes())
    mutate(data)
    digest = write_json(local, data)
    shutil.copyfile(local, Path(str(t.runs / row_id) + ".verify.json"))
    freeze_path = t.rows_dir / row_id / "freeze_receipt.json"
    freeze = json.loads(freeze_path.read_bytes())
    freeze.update(independent_verification_sha256=digest, verification_classification=data["classification"])
    write_json(freeze_path, freeze)


@pytest.mark.parametrize("mutation, message", [
    (lambda v: v.update(classification="T074B_PROMPTIR_OUTPUTS_FAILED"), "did not pass"),
    (lambda v: v.update(failures=[{"row": 3, "reason": "hash"}]), "lists failures"),
    (lambda v: v.update(pass_count=11, expected_count=12), "pass_count"),
    (lambda v: v.update(observed_row_count=11), "observed_row_count"),
    (lambda v: v.update(count=11), "count"),
])
def test_gate_requires_a_passing_verification(tmp_path, mutation, message):
    t = build_target(tmp_path, name="SYNTH_VERIFY")
    reseal_verification(t, "promptir", mutation)
    with pytest.raises(rg.GateError, match=message):
        open_gate(t)
    assert not t.receipt.exists()


def test_gate_accepts_quadprior_style_passing_verification(tmp_path):
    t = build_target(tmp_path, name="SYNTH_VERIFY_OK")
    reseal_verification(t, "promptir", lambda v: v.update(
        classification="QUADPRIOR_OUTPUTS_VERIFIED", failures=[], expected_count=12, observed_row_count=12, pass_count=12))
    open_gate(t)


def test_optimised_python_is_refused():
    import subprocess

    out = subprocess.run([sys.executable, "-O", "-c", "import reference_gate"], cwd=HERE, capture_output=True, text=True)
    assert out.returncode != 0 and "without -O" in out.stderr


# ---------------------------------------------------------------- metrics refusal


def test_metrics_refuse_without_gate_receipt(tmp_path, monkeypatch):
    t = build_target(tmp_path, name="SYNTH_NORECEIPT")
    decoded = []
    monkeypatch.setattr(np, "load", lambda *a, **k: decoded.append(a) or pytest.fail("GT decoded"))
    with pytest.raises(rg.GateError, match="no reference gate receipt"):
        context(t)
    assert not decoded


def test_metrics_refuse_if_bound_file_changes_after_gate(tmp_path):
    t = build_target(tmp_path, name="SYNTH_POSTGATE")
    open_gate(t)
    path = t.rows_dir / "ours_step0" / "verification.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(rg.GateError):
        context(t)


def test_metrics_refuse_forged_receipt(tmp_path):
    t = build_target(tmp_path, name="SYNTH_FORGED")
    receipt = rg.evaluate(t.name, t.rows_dir, t.low_receipt, t.opaque)
    receipt["rows"]["promptir"]["output_manifest_sha256"] = "f" * 64
    t.receipt.write_text(json.dumps(dict(receipt, created_utc="x", output_files_rehashed=True)))
    with pytest.raises(rg.GateError, match="rows"):
        context(t)


def test_metrics_refuse_changed_gt_bytes(tmp_path):
    t = build_target(tmp_path, name="SYNTH_GTBYTES")
    open_gate(t)
    ctx = context(t)
    gt = t.root / json.loads(t.opaque.read_bytes())["pairs"][0]["gt_relpath"]
    array = np.load(gt)
    array[0, 0, 0] ^= 1
    np.save(gt, array)
    with pytest.raises(rg.GateError, match="GT raw bytes changed"):
        ctx.reference(0)


# ---------------------------------------------------------------- metric fidelity


def independent_reference(t, index):
    """T073-A reference conversion on the official-loader uint8: uint8/255 in float64 -> float32 -> float64."""
    import cv2

    pair = json.loads(t.opaque.read_bytes())["pairs"][index]
    array = cv2.resize(np.load(t.root / pair["gt_relpath"]), (960, 512))[:, :, [2, 1, 0]]
    return (array.astype(np.float64) / 255).astype(np.float32).astype(np.float64)


def test_per_image_metrics_equal_direct_frozen_call(evaluated):
    t, ctx, result = evaluated
    core = mt.import_frozen_metric()
    for index, item in enumerate(t.files[:4]):
        y = independent_reference(t, index)
        assert np.array_equal(ctx.reference(index).astype(np.float64), y)
        for row_id in ("ours_ttt", "promptir"):
            manifest = json.loads((t.rows_dir / row_id / "output_manifest.json").read_bytes())
            with gzip.open(t.runs / row_id / Path(item["name"]).stem / "output.pt.gz", "rb") as stream:
                tensor = torch.load(stream, weights_only=True)
            # T071-B convention: C-contiguous HWC float64 (np.load'ed npy), then the preregistered clip
            x = np.clip(np.ascontiguousarray(tensor[0].permute(1, 2, 0).numpy(), dtype=np.float64), 0, 1)
            direct = core.metrics(x, y)
            got = result["per_image"][row_id][index]
            assert got["psnr"] == direct["psnr"] and got["rgb_ssim"] == direct["rgb_ssim"]
            assert got["clipped_below"] > 0 and got["clipped_above"] > 0
            assert manifest["rows"][index]["low_name"] == got["name"]


def test_result_structure_and_statistics(evaluated):
    t, ctx, result = evaluated
    assert result["n"] == 12 and result["clusters"]["G"] == 6 and result["clusters"]["low_power"]
    assert result["clusters"]["order"] == sorted({f["cluster"] for f in t.files})
    pairs = {(c["a"], c["b"]): c for c in result["comparisons"]}
    assert set(pairs) == {("ours_ttt", r) for r in ROWS if r != "ours_ttt"} | {("promptir_dctta", "promptir")}
    assert pairs[("ours_ttt", "ours_step0")]["families"] == ["headline", "adaptation_gain"]
    comp = pairs[("ours_ttt", "promptir")]
    d = np.asarray([a["psnr"] - b["psnr"] for a, b in zip(result["per_image"]["ours_ttt"], result["per_image"]["promptir"])])
    assert comp["psnr"]["mean_delta"] == float(d.mean()) and comp["psnr"]["win_fraction"] == float((d > 0).sum() / 12)
    for r in ROWS:
        ps = [x["psnr"] for x in result["per_image"][r]]
        assert result["rows"][r]["mean_psnr"] == float(np.mean(ps)) and result["rows"][r]["median_psnr"] == float(np.median(ps))
    assert result["rows"]["ours_ttt"]["no_active_abstentions"] == 0
    idx = mt.bootstrap_indices(6)
    assert result["bootstrap"]["indices_sha256"] == hashlib.sha256(idx.tobytes()).hexdigest()
    json.dumps(result, allow_nan=False)
    assert "ours_ttt - promptir" in mt.markdown(result)


def test_output_clip_is_applied():
    core = mt.import_frozen_metric()
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 1, (40, 56, 3)).astype(np.float32)
    x = rng.uniform(-0.2, 1.2, (40, 56, 3)).astype(np.float32)
    got = mt.score(x, y, core)
    clipped = core.metrics(np.clip(x.astype(np.float64), 0, 1), y.astype(np.float64))
    raw = core.metrics(x.astype(np.float64), y.astype(np.float64))
    assert got["psnr"] == clipped["psnr"] and got["rgb_ssim"] == clipped["rgb_ssim"]
    assert got["psnr"] != raw["psnr"] and got["rgb_ssim"] != raw["rgb_ssim"]
    # clipping after widening == widening after clipping (0 and 1 are exact in both dtypes)
    assert np.array_equal(np.clip(x.astype(np.float64), 0, 1), np.clip(x, 0, 1).astype(np.float64))


def test_score_fixes_c_contiguous_layout():
    """The frozen PSNR (np.mean) is layout-sensitive at one ulp; score() must not depend on the caller's strides."""
    core = mt.import_frozen_metric()
    rng = np.random.default_rng(8)
    chw = rng.uniform(-0.1, 1.1, (3, 64, 96)).astype(np.float32)
    y = np.ascontiguousarray(rng.uniform(0, 1, (64, 96, 3)).astype(np.float32).astype(np.float64))
    strided = chw.transpose(1, 2, 0)
    contiguous = np.ascontiguousarray(strided)
    assert not strided.flags["C_CONTIGUOUS"]
    got_strided, got_contiguous = mt.score(strided, y, core), mt.score(contiguous, y, core)
    assert got_strided == got_contiguous
    direct = core.metrics(np.clip(contiguous.astype(np.float64), 0, 1), y)
    assert got_contiguous["psnr"] == direct["psnr"] and got_contiguous["rgb_ssim"] == direct["rgb_ssim"]


def test_psnr_uses_unit_data_range():
    core = mt.import_frozen_metric()
    rng = np.random.default_rng(4)
    y = rng.uniform(0, 1, (32, 48, 3))
    x = np.clip(y + rng.normal(0, 0.05, y.shape), 0, 1)
    assert abs(mt.score(x, y, core)["psnr"] - 10 * np.log10(1.0 / np.mean((x - y) ** 2))) < 1e-10
    assert core.rgb_ssim(y, y) == 1.0
    with pytest.raises(rg.GateError, match="nonfinite"):  # identical images: PSNR inf fails closed
        mt.score(y, y, core)


def test_official_loader_unit_conversion_equals_t073a_conversion_exhaustively():
    stage = load_stage()
    values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    official = stage.to_unit(values)
    t073a = (values.astype(np.float64) / 255).astype(np.float32)
    assert official.dtype == np.float32 and np.array_equal(official, t073a)
    assert np.array_equal(official.view(np.uint32), t073a.view(np.uint32))


def test_frozen_metric_pins_match_t073a_plan():
    plan_path = HERE.parent / "T073A" / "metric_plan.json"
    if not plan_path.exists():
        pytest.skip("metric_plan.json not staged")
    plan = json.loads(plan_path.read_bytes())["metric_source"]["source_sha256"]
    assert plan["research_log/T071A/core.py"] == mt.FROZEN_METRIC_SHA256["research_log/T071A/core.py"]
    assert plan["ttie/ssim_transfer.py"] == mt.FROZEN_METRIC_SHA256["ttie/ssim_transfer.py"]
    core = mt.import_frozen_metric()
    assert sha(core.__file__) == plan["research_log/T071A/core.py"]


# ---------------------------------------------------------------- bootstrap


def test_rng_convention_reproduces_t073a_index_matrix_hash():
    indices = mt.bootstrap_indices(150)
    assert indices.shape == (10000, 150) and indices.dtype == np.int64 and indices.flags["C_CONTIGUOUS"]
    assert hashlib.sha256(indices.tobytes()).hexdigest() == T073A_INDICES_SHA256


@pytest.mark.parametrize("n", [1, 7, 150])
def test_singleton_clusters_reduce_exactly_to_t073a_image_bootstrap(n):
    rng = np.random.default_rng(n)
    deltas = rng.normal(0.3, 1.0, n)
    names = [f"img{i:04d}.png" for i in range(n)]
    order, position, sizes = mt.cluster_layout(names, names)
    boot, stats = mt.cluster_bootstrap(deltas, position, sizes, mt.bootstrap_indices(len(order)))
    image_idx = np.random.Generator(np.random.PCG64(20260922)).integers(0, n, size=(10000, n), dtype=np.int64)
    image_stats = deltas[image_idx].mean(axis=1)
    assert np.array_equal(stats, image_stats)
    assert boot["ci95"] == np.quantile(image_stats, [0.025, 0.975], method="linear").tolist()


def test_cluster_ratio_statistic_matches_explicit_resampling():
    rng = np.random.default_rng(5)
    clusters = ["b"] * 3 + ["a"] * 5 + ["c"] * 1 + ["d"] * 4
    deltas = rng.normal(0, 1, len(clusters))
    order, position, sizes = mt.cluster_layout([f"{i}" for i in range(len(clusters))], clusters)
    assert order == ["a", "b", "c", "d"] and sizes.tolist() == [5, 3, 1, 4]
    indices = mt.bootstrap_indices(4)
    _, stats = mt.cluster_bootstrap(deltas, position, sizes, indices)
    for b in range(0, 10000, 997):
        chosen = [deltas[i] for c in indices[b] for i in range(len(clusters)) if clusters[i] == order[c]]
        assert stats[b] == pytest.approx(np.mean(chosen), rel=0, abs=1e-12)


def test_singleton_order_guard():
    with pytest.raises(rg.GateError):
        mt.cluster_layout(["pair11__a.png", "pair1__b.png"], ["pair11", "pair1"])


# ---------------------------------------------------------------- phase-4 harness


def ours_available():
    return (OURS_SRC / "research_log" / "T070A" / "infer.py").is_file()


needs_ours = pytest.mark.skipif(not ours_available(), reason="frozen T070-A source not present")


@pytest.fixture(scope="module")
def ours():
    if str(OURS_SRC) not in sys.path:
        sys.path.insert(0, str(OURS_SRC))
    cwd = os.getcwd()
    os.chdir(OURS_SRC)
    threads = torch.get_num_threads()
    # FinalOurs.__init__ pins one thread; with many CPU threads even two frozen trajectory() calls differ.
    torch.set_num_threads(1)
    import ours_tuning

    binding = json.loads((OURS_SRC / "research_log" / "T070A" / "inference_binding.json").read_bytes())
    yield ours_tuning, binding
    torch.set_num_threads(threads)
    os.chdir(cwd)


def synthetic_low(seed=0, h=64, w=96):
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(0.02, 0.12, w).repeat(h, 1)
    return (base[None, None].repeat(1, 3, 1, 1) * (0.8 + 0.4 * torch.rand(1, 3, h, w, generator=g))).clamp(0, 1)


def fake_gate():
    return SimpleNamespace(active=torch.tensor([True, True, False, True]), winner=torch.tensor([0, 0, 1, 0]))


@needs_ours
def test_derivation_substitutions_and_default_equivalence(ours):
    tuning, binding = ours
    from research_log.T062CR2.core import trajectory
    from research_log.T070A.infer import select_trajectory

    derived, _ = tuning.derive(tuning.DEFAULTS, binding)
    model = json.loads((OURS_SRC / "research_log" / "T066A" / "evidence" / "model.json").read_bytes())
    compared = 0
    for seed in range(3):
        x = synthetic_low(seed)
        frozen = trajectory(x, fake_gate())
        mine = derived["research_log.T062CR2.core.trajectory"](x, fake_gate())
        for key in ("images", "states", "components", "gradients", "pre_box"):
            assert torch.equal(frozen[key], mine[key]), key
        assert frozen["values"] == mine["values"] and frozen["selected_step"] == mine["selected_step"] == 27
        try:
            expected = select_trajectory(x, frozen, model)[0]
        except AssertionError as exc:
            with pytest.raises(AssertionError, match=re.escape(str(exc)) if str(exc) else None):
                derived["research_log.T070A.infer.select_trajectory"](x, mine, model)
            continue
        assert derived["research_log.T070A.infer.select_trajectory"](x, mine, model)[0] == expected
        compared += 1
    assert compared > 0


@needs_ours
def test_knob_overrides_take_effect(ours):
    tuning, binding = ours
    x = synthetic_low(1)
    base = tuning.derive(tuning.canonical_knobs({}), binding)[0]["research_log.T062CR2.core.trajectory"](x, fake_gate())
    for override, check in (({"updates": 5}, lambda tr: len(tr["images"]) == 6 and tr["selected_step"] == 5),
                            ({"lr": 0.06}, lambda tr: not torch.equal(tr["states"][1], base["states"][1])),
                            ({"loss_weights": [1, 0, 5]}, lambda tr: tr["values"][0] != base["values"][0]),
                            ({"exposure_target": 0.5}, lambda tr: not torch.equal(tr["components"][0], base["components"][0]))):
        derived = tuning.derive(tuning.canonical_knobs(override), binding)[0]
        assert check(derived["research_log.T062CR2.core.trajectory"](x, fake_gate())), override


class FakeScorer:
    """Stands in for the CLIP scorer: fixed per-region (dark, bright) scores."""

    def __init__(self, scores):
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __call__(self, image):
        return self.scores.to(image.device)


def frozen_gate_json():
    return {"q_joint": 1.053775168916056,
            "calibration": {"tau": [0.027419920079410076, 0.001673370413482167],
                            "scale": [0.07507099353490992, 0.057845398696933635]}}


@needs_ours
def test_run_group_defaults_match_frozen_pipeline_and_abstention(ours):
    tuning, binding = ours
    from ttie.semantic_ttt import FixedObjective
    from research_log.T062CR2.core import trajectory
    from research_log.T070A.infer import select_trajectory

    model = json.loads((OURS_SRC / "research_log" / "T066A" / "evidence" / "model.json").read_bytes())
    scorer = FakeScorer([[0.2, 0.0], [0.3, 0.0], [0.0, 0.0], [0.25, 0.0]])
    gate = frozen_gate_json()
    compared = 0
    for seed in range(4):
        x = synthetic_low(seed)
        knobs = tuning.canonical_knobs({})
        variants = [{"id": "d", "knobs": knobs, "derived": tuning.derive(knobs, binding)[0]}]
        image, record = tuning.run_group(scorer, gate, model, x, variants, "cpu")["d"]
        objective = FixedObjective(scorer, x, gate)
        assert record["active"] == objective.active.tolist() and any(record["active"])
        try:
            trace = trajectory(x, objective)
            expected = trace["images"][select_trajectory(x, trace, model)[0]["selected_step"]]
        except AssertionError:
            assert record["status"] == "REJECTED"
            continue
        assert record["status"] == "TTT_EXECUTED" and torch.equal(image, expected)
        compared += 1
    assert compared > 0
    knobs = tuning.canonical_knobs({"q_joint": 1e9})
    variants = [{"id": "q", "knobs": knobs, "derived": tuning.derive(knobs, binding)[0]}]
    image, record = tuning.run_group(scorer, gate, model, synthetic_low(0), variants, "cpu")["q"]
    assert record["status"] == "TTT_ABSTAIN_NO_ACTIVE_GATE" and torch.equal(image, synthetic_low(0))


def test_grid_expansion_and_knob_validation():
    import ours_tuning as tuning

    settings = tuning.expand_grid({"product": {"lambda_value": [0.75, 0.875], "updates": [18, 27]}})
    assert settings[0] == tuning.canonical_knobs({}) and len(settings) == 4  # defaults deduplicated
    for bad in ({"lambda_value": 0.8}, {"unknown": 1}, {"updates": 2.0}, {"tau": [1.0]}, {"exposure_target": 1.5},
                {"updates": 28}, {"tau": [0.03, 0.001673370413482167]}):
        with pytest.raises(rg.GateError):
            tuning.canonical_knobs(bad)
    with pytest.raises(rg.GateError, match="duplicate"):
        tuning.expand_grid({"settings": [{"lr": 0.05}, {"lr": 0.05}]})
    assert set(tuning.KNOB_LOCATIONS) == set(tuning.DEFAULTS)


def test_selection_rule():
    import ours_tuning as tuning

    settings = tuning.expand_grid({"settings": [{"lr": 0.05}, {"lr": 0.09}, {"updates": 18}, {"lambda_value": 0.5}]})
    ids = [tuning.setting_id(s) for s in settings]

    def rec(i, psnr, ssim, status="COMPLETE"):
        return {"setting_id": ids[i], "knobs": settings[i], "status": status,
                "summary": {"mean_psnr": psnr, "mean_rgb_ssim": ssim}}

    records = [rec(0, 20.0, 0.70), rec(1, 20.995, 0.80), rec(2, 21.0, 0.80), rec(3, 21.004, 0.60), rec(4, 30.0, 0.9, "FAILED")]
    chosen, window = tuning.select(records, settings)
    assert set(window) == {ids[1], ids[2], ids[3]}
    assert chosen["setting_id"] == ids[1]  # SSIM tie with ids[2] (lr 0.09); lr 0.05 is nearer the defaults
    assert tuning.distance(settings[1], settings) == pytest.approx(0.02 / 0.06)
    assert tuning.distance(settings[2], settings) == pytest.approx(1.0)


def test_log_chain_detects_tampering(tmp_path):
    import ours_tuning as tuning

    path, records = tmp_path / "log.jsonl", []
    for i in range(3):
        tuning.append_log(path, records, {"setting_id": str(i), "status": "COMPLETE", "summary": {"mean_psnr": float(i)}})
    assert [r["setting_id"] for r in tuning.read_log(path)] == ["0", "1", "2"]
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[1].replace('"mean_psnr":1.0', '"mean_psnr":9.0'), lines[2]]) + "\n")
    with pytest.raises(rg.GateError):
        tuning.read_log(path)
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(rg.GateError, match="chain"):
        tuning.read_log(path)


def test_harness_refuses_to_start_without_gate_receipt(tmp_path, monkeypatch):
    import ours_tuning as tuning

    t = build_target(tmp_path, name="SYNTH_TUNE_NORECEIPT")
    (tmp_path / "grid.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["ours_tuning.py", "--target", t.name, "--low-receipt", str(t.low_receipt),
                                      "--low-dir", str(tmp_path / "staged" / "low"), "--opaque-manifest", str(t.opaque),
                                      "--gate-receipt", str(t.receipt), "--rows-dir", str(t.rows_dir),
                                      "--stage-target", str(STAGE_PATH), "--manifest", str(tmp_path / "none.json"),
                                      "--grid", str(tmp_path / "grid.json"), "--work", str(tmp_path / "work")])
    with pytest.raises(rg.GateError, match="no reference gate receipt"):
        tuning.main()
    assert not (tmp_path / "work").exists()


def test_tuned_row_materialises_and_enters_metrics(evaluated, tmp_path):
    import ours_tuning as tuning

    t, ctx, _ = evaluated
    settings = tuning.expand_grid({"settings": [{"lr": 0.05}]})
    sid = tuning.setting_id(settings[1])
    rng = np.random.default_rng(9)
    images, per_image = [], []
    for item in t.files:
        image = torch.from_numpy(rng.uniform(0, 1, (1, 3, 512, 960)).astype(np.float32))
        images.append((item["name"], image))
        per_image.append({"name": item["name"], "status": "TTT_EXECUTED", "selected_step": 3, "active": [True] * 4,
                          "shape": [1, 3, 512, 960], "dtype": "torch.float32",
                          "output_tensor_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest()})
    work = tmp_path / "work"
    (work / "candidates").mkdir(parents=True)
    tuning.write_outputs(work / "candidates" / sid, images, per_image)
    records = []
    record = tuning.append_log(work / "log.jsonl", records, {"setting_id": sid, "knobs": settings[1], "status": "COMPLETE",
                                                             "per_image": per_image,
                                                             "summary": {"mean_psnr": 1.0, "mean_rgb_ssim": 0.5, "abstentions": 0}})
    manifest = tuning.materialize(work, record, [sid], ctx, records, settings, "e" * 64, work / "log.jsonl")
    assert manifest["method_id"] == rg.TUNED_ROW and manifest["gate_receipt_sha256"] == ctx.receipt_sha256
    with pytest.raises(FileExistsError):
        tuning.materialize(work, record, [sid], ctx, records, settings, "e" * 64, work / "log.jsonl")
    result = mt.evaluate(ctx, work / rg.TUNED_ROW)
    families = {(c["a"], c["b"]): c["families"] for c in result["comparisons"]}
    assert families[(rg.TUNED_ROW, "ours_ttt")] == ["tuned_headline"]
    assert result["rows"][rg.TUNED_ROW]["no_active_abstentions"] == 0
    assert result["rows"][rg.TUNED_ROW]["tuned_on_test_gt"] is True and "optimistic" in result["rows"][rg.TUNED_ROW]["note"]
    assert all(c.get("tuned_on_test_gt") is (True if rg.TUNED_ROW in (c["a"], c["b"]) else None) for c in result["comparisons"])
    assert "optimistic" in mt.markdown(result) and "(tuned on test GT)" in mt.markdown(result)
    assert "tuned_on_test_gt" not in result["rows"]["ours_ttt"]


# ---------------------------------------------------------------- multi-round search (B1, M1)


def test_round1_grid_file_matches_decisions():
    import ours_tuning as tuning

    spec = json.loads((HERE / "tuning_grid_round1.json").read_bytes())
    settings = tuning.expand_grid(spec)
    defaults = tuning.canonical_knobs({})
    assert settings[0] == defaults and len(settings) == 23  # 2+2+8+2+2+2+4 one-factor settings + defaults
    assert spec["one_factor"]["lambda_value"] == list(tuning.LAMBDA_GRID)
    changed = [[k for k in tuning.DEFAULTS if s[k] != defaults[k]] for s in settings[1:]]
    assert all(len(c) == 1 for c in changed) and "tau" not in {c[0] for c in changed}
    assert max(s["updates"] for s in settings) == 27


def r1_record(tuning, knobs, psnr, ssim, status="COMPLETE", round_no=1):
    knobs = tuning.canonical_knobs(knobs)
    return {"setting_id": tuning.setting_id(knobs), "knobs": knobs, "status": status, "round": round_no,
            "summary": None if status != "COMPLETE" else {"mean_psnr": psnr, "mean_rgb_ssim": ssim}}


def test_round2_rule():
    import ours_tuning as tuning

    records = [r1_record(tuning, {}, 20.0, 0.70),
               r1_record(tuning, {"lr": 0.015}, 20.5, 0.71), r1_record(tuning, {"lr": 0.04}, 20.5, 0.71),
               r1_record(tuning, {"lr": 0.06}, 19.0, 0.70),
               r1_record(tuning, {"updates": 9}, 20.2, 0.72), r1_record(tuning, {"updates": 18}, 20.3, 0.60),
               r1_record(tuning, {"exposure_target": 0.5}, 20.3, 0.75), r1_record(tuning, {"exposure_target": 0.7}, 20.3, 0.74),
               r1_record(tuning, {"lambda_value": 0.5}, 20.1, 0.70),
               r1_record(tuning, {"q_joint": 0.35266535990213066}, None, None, "FAILED"),
               r1_record(tuning, {"loss_weights": [1, 20, 5]}, 20.3, 0.70),
               r1_record(tuning, {"probability_threshold": 0.3}, 19.9, 0.90),
               r1_record(tuning, {"lr": 0.05}, 99.0, 0.99, round_no=2)]  # round-2 records never feed the rule
    round1 = [r["knobs"] for r in records if r["round"] == 1]
    settings, why = tuning.round2_grid(records, round1)
    assert "q_joint" not in why["gains"] and why["gains"]["probability_threshold"] == 0.0
    # gains: lr 0.5; exposure/loss_weights/updates 0.3 each -> SSIM of best setting 0.75 > 0.70 > 0.60
    assert why["selected_knobs"] == ["lr", "exposure_target", "loss_weights"]
    assert why["values"]["lr"] == [0.04, 0.015]  # PSNR and SSIM tie -> nearer the default first
    assert why["values"]["exposure_target"] == [0.5, 0.7]  # PSNR tie -> higher SSIM
    assert why["values"]["loss_weights"] == [[1.0, 20.0, 5.0], [1.0, 10.0, 5.0]]  # vector is one value
    assert len(settings) == 8 and len({tuning.setting_id(s) for s in settings}) == 8
    assert all(s["updates"] == 27 and s["q_joint"] == tuning.DEFAULTS["q_joint"] for s in settings)
    flat_records = [r1_record(tuning, {}, 20.0, 0.7), r1_record(tuning, {"lr": 0.06}, 19.0, 0.8)]
    assert tuning.round2_grid(flat_records, [r["knobs"] for r in flat_records])[0] == []


def test_round2_takes_at_most_three_knobs_and_breaks_gain_ties_by_name():
    import ours_tuning as tuning

    records = [r1_record(tuning, {}, 20.0, 0.7)] + [r1_record(tuning, {k: v}, 20.5, 0.7) for k, v in
                                                     (("updates", 18), ("lr", 0.06), ("exposure_target", 0.5), ("probability_threshold", 0.3))]
    _, why = tuning.round2_grid(records, [r["knobs"] for r in records])
    assert why["selected_knobs"] == ["exposure_target", "lr", "probability_threshold"]


def test_round_files_superset_and_immutability(tmp_path):
    import ours_tuning as tuning

    small = tuning.expand_grid({"one_factor": {"lr": [0.015]}})
    big = tuning.expand_grid({"one_factor": {"lr": [0.015, 0.06]}})
    tuning.store_round(tmp_path, 1, small, "a")
    tuning.store_round(tmp_path, 1, big, "b")  # superset accepted
    with pytest.raises(rg.GateError, match="superset"):
        tuning.store_round(tmp_path, 1, small[1:], "c")
    tuning.store_round(tmp_path, 2, [big[1]], "r1")
    with pytest.raises(rg.GateError, match="after round 2"):
        tuning.store_round(tmp_path, 1, big, "b")
    with pytest.raises(rg.GateError, match="round-2 grid differs"):
        tuning.store_round(tmp_path, 2, [big[2]], "r1")


def test_materialize_global_over_union_of_rounds(evaluated, tmp_path):
    import ours_tuning as tuning

    t, ctx, _ = evaluated
    work = tmp_path / "work"
    (work / "candidates").mkdir(parents=True)
    context = dict(tuning.static_context(ctx), execution_manifest_sha256="e" * 64, derivation_sha256="d" * 64)
    round1 = tuning.expand_grid({"one_factor": {"lr": [0.05], "updates": [18]}})
    tuning.store_round(work, 1, round1, "g1")
    rng = np.random.default_rng(11)
    records = []
    for knobs, psnr, round_no in ((round1[0], 20.0, 1), (round1[1], 21.0, 1), (round1[2], 20.5, 1)):
        sid = tuning.setting_id(knobs)
        images, per_image = [], []
        for item in t.files:
            image = torch.from_numpy(rng.uniform(0, 1, (1, 3, 512, 960)).astype(np.float32))
            images.append((item["name"], image))
            per_image.append({"name": item["name"], "status": "TTT_EXECUTED", "output_tensor_sha256":
                              hashlib.sha256(image.numpy().tobytes()).hexdigest()})
        tuning.write_outputs(work / "candidates" / sid, images, per_image)
        tuning.append_log(work / "tuning_log.jsonl", records, {
            "setting_id": sid, "knobs": knobs, "status": "COMPLETE", "round": round_no, "grid_sha256": "g1",
            "context": context, "per_image": per_image,
            "summary": {"mean_psnr": psnr, "mean_rgb_ssim": 0.5, "abstentions": 0}})
    with pytest.raises(rg.GateError, match="round 2 not run"):
        tuning.materialize_global(work, ctx, records, work / "tuning_log.jsonl")
    settings2, _ = tuning.round2_grid(records, round1)
    assert len(settings2) == 4  # lr {0.05, 0.03} x updates {18, 27}
    tuning.store_round(work, 2, settings2, "r1")
    with pytest.raises(rg.GateError, match="round 2 incomplete"):
        tuning.materialize_global(work, ctx, records, work / "tuning_log.jsonl")
    combo = tuning.canonical_knobs({"lr": 0.05, "updates": 18})
    sid = tuning.setting_id(combo)
    images = [(item["name"], torch.zeros(1, 3, 512, 960)) for item in t.files]
    per_image = [{"name": n, "status": "TTT_EXECUTED", "output_tensor_sha256": hashlib.sha256(i.numpy().tobytes()).hexdigest()}
                 for n, i in images]
    tuning.write_outputs(work / "candidates" / sid, images, per_image)
    tuning.append_log(work / "tuning_log.jsonl", records, {
        "setting_id": sid, "knobs": combo, "status": "COMPLETE", "round": 2, "grid_sha256": "g2", "context": context,
        "per_image": per_image, "summary": {"mean_psnr": 21.5, "mean_rgb_ssim": 0.5, "abstentions": 0}})
    manifest = tuning.materialize_global(work, ctx, records, work / "tuning_log.jsonl")
    assert manifest["setting_id"] == sid and manifest["selection"]["settings_declared"] == 4
    assert manifest["selection"]["tie_window"] == [sid]
    with pytest.raises(FileExistsError):  # materialize only once
        tuning.materialize_global(work, ctx, records, work / "tuning_log.jsonl")


# ---------------------------------------------------------------- T075 targets: LSRW (mixed native geometry), SMID


import targets_t075 as t075  # noqa: E402


def t073a_reference_png_jpeg(path):
    """T071-B / T073-A reference conversion for image files: PIL RGB uint8 / 255 in float64 -> float32 -> float64."""
    from PIL import Image

    with Image.open(path) as image:
        return (np.asarray(image.convert("RGB"), dtype=np.float64) / 255).astype(np.float32).astype(np.float64)


def test_t075_registry_pins_and_sdsd_untouched():
    sdsd_before = json.dumps(rg.TARGETS["SDSD_indoor"], sort_keys=True)
    t075.register()
    t075.register()  # idempotent
    assert json.dumps(rg.TARGETS["SDSD_indoor"], sort_keys=True) == sdsd_before
    for name, spec in t075.T075_TARGETS.items():
        assert rg.TARGETS[name] == spec and spec["required_rows"] == rg.TARGETS["SDSD_indoor"]["required_rows"]
        assert t075.default_rows_dir(name) == HERE.parent / "T075B" / name
        folder = HERE.parent / "T074B" / "targets" / name
        if not (folder / "reference_opaque_manifest.json").exists():
            pytest.skip("target receipts not staged")
        low, low_sha, opaque, opaque_sha = rg.check_low_and_opaque(spec, folder / "low_receipt.json",
                                                                  folder / "reference_opaque_manifest.json")
        assert low_sha == spec["low_receipt_sha256"] and opaque_sha == spec["reference_opaque_manifest_sha256"]
    clusters = [f["cluster"] for f in json.loads((HERE.parent / "T074B/targets/LSRW/low_receipt.json").read_bytes())["files"]]
    assert len(set(clusters)) == len(clusters) == 50
    mt.cluster_layout(clusters, clusters)  # singleton guard holds: exact T073-A reduction for LSRW
    smid = json.loads((HERE.parent / "T074B/targets/SMID/low_receipt.json").read_bytes())["files"]
    assert len({f["cluster"] for f in smid}) == 49
    with pytest.raises(rg.GateError, match="different spec"):
        rg.TARGETS["LSRW"] = dict(rg.TARGETS["LSRW"], expected_count=51)
        t075.register()
    rg.TARGETS["LSRW"] = t075.T075_TARGETS["LSRW"]


def test_t075_default_rows_dir_injection():
    assert t075._with_rows_dir(["--target", "LSRW", "--x", "1"])[-2:] == ["--rows-dir", str(HERE.parent / "T075B" / "LSRW")]
    assert t075._with_rows_dir(["--target", "SMID", "--rows-dir", "r"]) == ["--target", "SMID", "--rows-dir", "r"]
    assert t075._with_rows_dir(["--target", "SDSD_indoor"]) == ["--target", "SDSD_indoor"]


@pytest.fixture(scope="module")
def lsrw(tmp_path_factory):
    t = build_target(tmp_path_factory.mktemp("lsrw"), name="SYNTH_LSRW", kind="lsrw")
    open_gate(t)
    ctx = context(t)
    return t, ctx, mt.evaluate(ctx)


def test_lsrw_mixed_geometry_metrics_equal_direct_frozen_call(lsrw):
    t, ctx, result = lsrw
    shapes = {(f["height"], f["width"]) for f in t.files}
    assert shapes == set(LSRW_SIZES.values())  # really mixed
    core = mt.import_frozen_metric()
    opaque = json.loads(t.opaque.read_bytes())
    for index, item in enumerate(t.files):
        y = t073a_reference_png_jpeg(t.root / opaque["pairs"][index]["gt_relpath"])
        assert y.shape == (item["height"], item["width"], 3)
        assert np.array_equal(ctx.reference(index).astype(np.float64), y)
        for row_id in ("ours_ttt", "promptir"):
            with gzip.open(t.runs / row_id / Path(item["name"]).stem / "output.pt.gz", "rb") as stream:
                tensor = torch.load(stream, weights_only=True)
            x = np.clip(np.ascontiguousarray(tensor[0].permute(1, 2, 0).numpy(), dtype=np.float64), 0, 1)
            direct = core.metrics(x, y)
            got = result["per_image"][row_id][index]
            assert got["psnr"] == direct["psnr"] and got["rgb_ssim"] == direct["rgb_ssim"]


def test_lsrw_singleton_clusters_give_the_t073a_image_bootstrap(lsrw):
    t, ctx, result = lsrw
    n = len(t.files)
    assert result["clusters"]["G"] == n and result["clusters"]["sizes"] == [1] * n
    comp = next(c for c in result["comparisons"] if (c["a"], c["b"]) == ("ours_ttt", "promptir"))
    d = np.asarray([a["psnr"] - b["psnr"] for a, b in zip(result["per_image"]["ours_ttt"], result["per_image"]["promptir"])])
    idx = np.random.Generator(np.random.PCG64(20260922)).integers(0, n, size=(10000, n), dtype=np.int64)
    assert comp["psnr"]["ci95"] == np.quantile(d[idx].mean(axis=1), [0.025, 0.975], method="linear").tolist()


def test_lsrw_geometry_checks_are_per_image(lsrw):
    t, ctx, _ = lsrw
    tall = next(i for i, f in enumerate(t.files) if f["height"] == 64)
    short = next(i for i, f in enumerate(t.files) if f["height"] == 48)
    manifest = json.loads((t.rows_dir / "promptir" / "output_manifest.json").read_bytes())
    manifest["rows"][tall]["shape"] = manifest["rows"][short]["shape"]
    with pytest.raises(rg.GateError, match="geometry"):
        rg._check_manifest_rows("promptir", manifest, t.files, sha(t.low_receipt))
    row = json.loads((t.rows_dir / "promptir" / "output_manifest.json").read_bytes())["rows"][tall]
    path = t.runs / "promptir" / Path(t.files[tall]["name"]).stem / "output.pt.gz"
    assert mt.load_output(path, row, t.files[tall]).shape == (64, 72, 3)
    wrong_item = dict(t.files[tall], height=48)
    with pytest.raises(rg.GateError, match="geometry"):
        mt.load_output(path, row, wrong_item)
    with pytest.raises(rg.GateError, match="shape mismatch"):
        mt.score(np.zeros((48, 72, 3)), ctx.reference(tall), ctx.core)


def test_smid_shared_sequence_gt_and_clusters(tmp_path):
    t = build_target(tmp_path, name="SYNTH_SMID", kind="smid", rows=["promptir", "ours_step0", "ours_ttt"])
    opaque = json.loads(t.opaque.read_bytes())
    assert opaque["count"] == 6 and opaque["unique_gt_count"] == 3
    open_gate(t)
    ctx = context(t)
    refs = [ctx.reference(i) for i in range(len(t.files))]
    for i, j in ((0, 1), (2, 3), (4, 5)):
        assert t.files[i]["cluster"] == t.files[j]["cluster"] and np.array_equal(refs[i], refs[j])
        assert refs[i].shape == (512, 960, 3)
    result = mt.evaluate(ctx)
    assert result["clusters"]["G"] == 3 and result["clusters"]["sizes"] == [2, 2, 2]
    assert result["clusters"]["order"] == ["0003", "0011", "0027"]


def test_t075_wrapper_gate_and_metrics(tmp_path):
    t = build_target(tmp_path, name="SYNTH_WRAP", kind="lsrw", rows=["promptir", "ours_step0", "ours_ttt"])
    common = ["--target", t.name, "--low-receipt", str(t.low_receipt), "--opaque-manifest", str(t.opaque),
              "--rows-dir", str(t.rows_dir)]
    t075.main(["gate"] + common)
    receipt = json.loads(t.receipt.read_bytes())
    assert receipt["target_registry"]["sha256"] == sha(HERE / "targets_t075.py")
    with pytest.raises(rg.GateError, match="immutable"):
        t075.main(["gate"] + common)
    t075.main(["metrics"] + common + ["--gate-receipt", str(t.receipt), "--stage-target", str(STAGE_PATH),
                                      "--out", str(tmp_path / "metrics")])
    result = json.loads((tmp_path / "metrics" / "metrics_result.json").read_bytes())
    assert result["n"] == 5 and set(result["rows"]) == {"promptir", "ours_step0", "ours_ttt"}
    with pytest.raises(rg.GateError, match="usage"):
        t075.main(["bogus"])
