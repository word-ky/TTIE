"""CPU tests for run_quadprior_batch.py and verify_quadprior_full.py.

A fake source tree mirrors the official layout the runner hooks
(`test.py` run as __main__, `ldm.models.diffusion.ddpm.LatentDiffusion`,
`ldm.models.diffusion.dpm_solver.DPMSolverSampler`, cwd-relative weights and
`empty_embedding.pkl`) and reproduces test.py's resize/quantize/save code on
tiny synthetic images. No real low image, weight or reference is touched.
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

import run_quadprior_batch as runner
import verify_quadprior_full as verifier

SIZES = [(20, 36), (24, 24)]

FAKE_TEST_PY = textwrap.dedent('''
    import argparse, glob, os, pickle
    import cv2, einops
    import numpy as np
    import torch
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from annotator.util import resize_image, HWC3

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/COCO-final.ckpt", type=str)
    parser.add_argument("--same_folder", default="output_QuadPrior", type=str)
    parser.add_argument("--input_folder", default="test_data", type=str)
    parser.add_argument("--use_float16", default=True, type=bool)
    parser.add_argument("--save_memory", default=False, type=bool)
    MODE = os.environ.get("FAKE_MODE", "ok")

    if __name__ == "__main__":
        args = parser.parse_args()
        for required in (args.checkpoint, "./models/cldm_v15.yaml", "./models/control_sd15_ini.ckpt",
                         "./checkpoints/main-epoch=00-step=7000.ckpt"):
            with open(required, "rb") as f:
                f.read(1)
        with open("empty_embedding.pkl", "rb") as f:
            pickle.load(f)
        model = LatentDiffusion().to(dtype=torch.float16)
        diffusion_sampler = DPMSolverSampler(model)
        for img_path in glob.glob(f"{args.input_folder}/*.*"):
            save_name = os.path.splitext(os.path.split(img_path)[1])[0] + ".png"
            save_path = os.path.join(args.same_folder, save_name)
            if os.path.exists(save_path):
                continue
            input_image = cv2.imread(img_path)
            if MODE == "read_other":
                cv2.imread("../low/stray.png")
            detected = resize_image(HWC3(input_image), 512)
            control = torch.from_numpy(detected.copy()).to(dtype=torch.float16) / 255.0
            control = einops.rearrange(control[None], "b h w c -> b c h w").clone()
            x_samples = model.decode_new_first_stage((control * 2 - 1) * 1.03, None)
            x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            output = x_samples[0]
            if MODE == "tamper":
                output = output.copy()
                output[0, 0, 0] ^= 1
            os.makedirs(args.same_folder, exist_ok=True)
            cv2.imwrite(save_path, output)
''')

FAKE_DDPM = textwrap.dedent('''
    import torch
    from my_vae.models import Encoder

    class FirstStage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(3, 4, 1)

    class LatentDiffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_stage_model = FirstStage()
            self.control_model = torch.nn.Linear(2, 2)

        @torch.no_grad()
        def decode_new_first_stage(self, z, hs, predict_cids=False, force_not_quantize=False):
            return z
''')

FAKE_SAMPLER = textwrap.dedent('''
    class DPMSolverSampler(object):
        def __init__(self, model, **kwargs):
            self.model = model
''')

FAKE_UTIL = textwrap.dedent('''
    import cv2
    import numpy as np

    def HWC3(x):
        return x

    def resize_image(input_image, resolution):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        return cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
''')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def purge_fake_modules():
    for key in list(sys.modules):
        if key.split(".")[0] in {"ldm", "my_vae", "annotator"}:
            del sys.modules[key]


def build(root, extra_file=False, xformers_available=False):
    src = root / "src"
    write(src / "test.py", FAKE_TEST_PY)
    write(src / "ldm" / "__init__.py", "")
    write(src / "ldm" / "models" / "__init__.py", "")
    write(src / "ldm" / "models" / "diffusion" / "__init__.py", "")
    write(src / "ldm" / "models" / "diffusion" / "ddpm.py", FAKE_DDPM)
    write(src / "ldm" / "models" / "diffusion" / "dpm_solver" / "__init__.py",
          "from .sampler import DPMSolverSampler\n")
    write(src / "ldm" / "models" / "diffusion" / "dpm_solver" / "sampler.py", FAKE_SAMPLER)
    write(src / "ldm" / "modules" / "__init__.py", "")
    write(src / "ldm" / "modules" / "attention.py", f"XFORMERS_IS_AVAILBLE = {xformers_available}\n")
    write(src / "ldm" / "modules" / "diffusionmodules" / "__init__.py", "")
    write(src / "ldm" / "modules" / "diffusionmodules" / "model.py",
          f"XFORMERS_IS_AVAILBLE = {xformers_available}\n")
    write(src / "my_vae" / "__init__.py", "")
    write(src / "my_vae" / "models.py",
          f"import torch\n\nXFORMERS_IS_AVAILBLE = {xformers_available}\n\n"
          "class Encoder(torch.nn.Conv2d):\n    pass\n")
    write(src / "annotator" / "__init__.py", "")
    write(src / "annotator" / "util.py", FAKE_UTIL)
    write(src / "models" / "cldm_v15.yaml", "model: {}\n")
    (src / "empty_embedding.pkl").write_bytes(pickle.dumps([0.0]))
    weights = {}
    for label in ("coco", "vae", "control"):
        weights[label] = root / f"{label}.ckpt"
        weights[label].write_bytes(label.encode() * 10)
    low = root / "low"
    low.mkdir()
    files = []
    for index, hw in enumerate(SIZES):
        y, x = np.indices(hw)
        image = np.stack(((x * 7 + index) % 256, (y * 11) % 256, (x + y) * 5 % 256), -1).astype(np.uint8)
        name = f"syn{index}_low.png"
        assert cv2.imwrite(str(low / name), image)
        files.append({"name": name, "sha256": runner.sha256_file(low / name)})
    if extra_file:
        (low / "stray.png").write_bytes(b"x")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"count": len(files), "files": files}))
    purge_fake_modules()
    return argparse.Namespace(
        source_root=src, coco_checkpoint=weights["coco"], vae_checkpoint=weights["vae"],
        control_checkpoint=weights["control"], low_dir=low, low_receipt=receipt, out=root / "out",
        workdir=root / "work", expected_count=2, smoke_one=False, expected_coco_sha256=None,
        expected_vae_sha256=None, expected_control_sha256=None, synthetic=True)


def run_fake(args, mode="ok"):
    os.environ["FAKE_MODE"] = mode
    try:
        return runner.run(args, require_cuda=False, pin_path=None)
    finally:
        os.environ.pop("FAKE_MODE", None)
        purge_fake_modules()


def verifier_args(run_args, root):
    return argparse.Namespace(
        low_receipt=run_args.low_receipt, low_dir=run_args.low_dir, outputs=run_args.out,
        out=root / "verify.json", expected_count=2,
        expected_coco_sha256=runner.sha256_file(run_args.coco_checkpoint),
        expected_vae_sha256=runner.sha256_file(run_args.vae_checkpoint),
        expected_control_sha256=runner.sha256_file(run_args.control_checkpoint),
        expected_output_manifest_sha256="", first_row_tensor_sha256="", require_promotable=False)


def rewrite_row(outputs, index, **changes):
    manifest_path = outputs / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["rows"][index].update(changes)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    row = manifest["rows"][index]
    (outputs / Path(row["low_name"]).stem / "decision.json").write_text(json.dumps(row, indent=2))


def load(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, weights_only=True)


class RunnerTests(unittest.TestCase):
    def test_happy_path_and_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build(root)
            manifest = run_fake(args)
            self.assertEqual(manifest["count"], 2)
            self.assertFalse(manifest["promotable"])
            self.assertTrue(manifest["model_state_unchanged"])
            first, second = manifest["rows"]
            self.assertEqual(first["shape"], [1, 3, *SIZES[0]])
            self.assertEqual(first["native_shape"], [1, 3, 512, 896])
            self.assertEqual(second["native_shape"], [1, 3, 512, 512])
            self.assertGreater(first["prequant_fraction_above_255"] + first["prequant_fraction_below_0"], 0)
            native = load(args.out / "syn0_low" / "native_output.pt.gz")
            output = load(args.out / "syn0_low" / "output.pt.gz")
            self.assertGreater(float(native.max()), 1.0)  # native stays unclipped
            self.assertLessEqual(float(output.max()), 1.0 + 1e-6)  # saturated before map-back
            # RGB order: the fake model is identity on the BGR control, so the native
            # float's channel 0 must track the low image's R channel.
            low = cv2.imread(str(args.low_dir / "syn0_low.png"))
            small = cv2.resize(low, (896, 512), interpolation=cv2.INTER_LANCZOS4)
            expected_r = np.clip(((small[..., 2].astype(np.float32) / 255) * 2 - 1) * 1.03 * 0.5 + 0.5, 0, 1)
            np.testing.assert_allclose(np.clip(native[0, 0].numpy(), 0, 1), expected_r, atol=2e-3)
            ok, receipt = verifier.verify(verifier_args(args, root))
            self.assertTrue(ok, receipt["failures"])

    def test_unplanned_read_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaises(PermissionError):
                run_fake(args, "read_other")

    def test_capture_must_equal_official_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaisesRegex(AssertionError, "official quantized"):
                run_fake(args, "tamper")

    def test_extra_file_in_low_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp), extra_file=True)
            with self.assertRaisesRegex(AssertionError, "exactly the receipt"):
                run_fake(args)

    def test_existing_output_dir_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            args.out.mkdir()
            with self.assertRaisesRegex(AssertionError, "must not exist"):
                run_fake(args)

    def test_promotable_requires_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            args.synthetic = False
            with self.assertRaisesRegex(AssertionError, "requires all three weight pins"):
                run_fake(args)

    def test_smoke_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            args.smoke_one = True
            manifest = run_fake(args)
            self.assertEqual([r["low_name"] for r in manifest["rows"]], ["syn0_low.png"])
            self.assertEqual(sorted(os.listdir(args.workdir / "input")), ["syn0_low.png"])

    def test_xformers_available_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp), xformers_available=True)
            with self.assertRaisesRegex(AssertionError, "XFORMERS_IS_AVAILBLE is True"):
                run_fake(args)


class VerifierMutationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = build(self.root)
        run_fake(self.args)
        self.vargs = verifier_args(self.args, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def assertFailsWith(self, text):
        ok, receipt = verifier.verify(self.vargs)
        self.assertFalse(ok)
        self.assertTrue(any(text in f for f in receipt["failures"]), receipt["failures"])

    def test_tensor_hash_mutation(self):
        rewrite_row(self.args.out, 0, output_tensor_sha256="2" * 64)
        self.assertFailsWith("output_tensor_sha256 mismatch")

    def test_map_back_mutation(self):
        path = self.args.out / "syn1_low" / "output.pt.gz"
        tensor = (load(path) * 0.99).contiguous()
        runner.save_gz(tensor, path)
        rewrite_row(self.args.out, 1, output_tensor_sha256=verifier.thash(tensor),
                    output_file_sha256=verifier.sha256_file(path))
        self.assertFailsWith("map-back recomputation differs")

    def test_native_outside_official_bin(self):
        path = self.args.out / "syn0_low" / "native_output.pt.gz"
        tensor = (load(path) + 2.0 / 255.0).contiguous()
        runner.save_gz(tensor, path)
        rewrite_row(self.args.out, 0, native_tensor_sha256=verifier.thash(tensor),
                    native_file_sha256=verifier.sha256_file(path))
        self.assertFailsWith("outside the official uint8 truncation bin")

    def test_low_image_mutation(self):
        path = self.args.low_dir / "syn1_low.png"
        path.write_bytes(path.read_bytes() + b"x")
        self.assertFailsWith("low sha256 does not match frozen receipt")

    def test_wrong_weight_pin(self):
        self.vargs.expected_control_sha256 = "0" * 64
        self.assertFailsWith("control_sd15_ini sha256 mismatch")

    def test_missing_official_png(self):
        (self.args.out / "official_output" / "syn1_low.png").unlink()
        self.assertFailsWith("official PNG missing")


if __name__ == "__main__":
    unittest.main()
