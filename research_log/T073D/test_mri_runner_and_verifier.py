"""CPU tests for run_mri_native_batch.py and verify_mri_full.py.

A stub stands in for the official `main` module: it mimics the official
decode/quantize code path (main.py:326-338, 505-537) on tiny synthetic images.
No real low image, model weight or reference is touched.
"""

import argparse
import gzip
import hashlib
import json
import logging
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import einops
import numpy as np
import PIL.Image
import torch

import run_mri_native_batch as runner
import verify_mri_full as verifier

SIZES = [(8, 16), (10, 18)]  # second image exercises the official /8 resize-back path


class XFormersAttnProcessor:
    pass


class Encoder(torch.nn.Conv2d):
    pass


Encoder.__module__ = "finetuned_vae.models"


class StubVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(3, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 3, 1)

    def decode(self, z, hs):
        return z


def make_official(mode="ok"):
    module = types.SimpleNamespace()
    module.DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    module.LOGGER = logging.getLogger("stub_main")

    class InferenceConfig:
        def __init__(self, checkpoint_path, img_dir_path, save_dir, scaled_save_dir):
            self.checkpoint_path = checkpoint_path
            self.img_dir_path = img_dir_path
            self.save_dir = save_dir
            self.scaled_save_dir = scaled_save_dir
            self.device = "cpu"
            for key, value in runner.OFFICIAL_DEFAULTS.items():
                setattr(self, key, value)

    class Enhancer(torch.nn.Module):  # nn.Module like the official DiffusionPriorEnhancer
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight_dtype = torch.float32
            self.vae = StubVAE()
            self.pipeline = types.SimpleNamespace(vae=self.vae)
            self.unet = torch.nn.Linear(2, 2)
            self.unet.attn1 = torch.nn.Linear(1, 1, bias=False)
            self.unet.register_buffer("steps", torch.zeros(1))
            self.unet.attn_processors = {"a": XFormersAttnProcessor()}
            self.text_encoder = torch.nn.Linear(2, 2)
            self.scheduler = types.SimpleNamespace(config={"num_train_timesteps": 1000})
            state = torch.load(config.checkpoint_path, map_location="cpu")["state_dict"]
            cleaned = {k.replace("my_vae.", "", 1): v for k, v in state.items() if k.startswith("my_vae.")}
            self.vae.load_state_dict(cleaned, strict=False)
            self.should_resize_output_back = False
            self.original_image_size = None

        def process(self, path):
            import cv2
            pil = PIL.Image.open(path).convert("RGB")
            if mode == "extra_open":
                PIL.Image.open(Path(path).with_name("not_planned.png"))
            self.original_image_size = pil.size
            # main_utils.py:109/195: the enhancer becomes a submodule of its own UNet (a cycle).
            setattr(self.unet.attn1, "our_pipeline", self)
            if mode == "replace_param":
                self.unet.weight = torch.nn.Parameter(self.unet.weight.detach().clone())
            if mode == "mutate_param":
                with torch.no_grad():
                    self.unet.weight.add_(1.0)
            rounded = tuple(-(-v // 8) * 8 for v in pil.size)
            self.should_resize_output_back = rounded != pil.size
            if self.should_resize_output_back:
                pil = pil.resize(rounded, PIL.Image.LANCZOS)
            image = np.array(pil)
            decoded_in = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 127.5 - 1.0
            decoded_in = decoded_in * 1.02 + 0.003  # some values leave [-1, 1]
            decoded = self.pipeline.vae.decode(decoded_in, None)
            decoded = einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5
            decoded = decoded.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            if mode == "tamper_output":
                decoded = decoded.copy()
                decoded[0, 0, 0, 0] ^= 1
            output = decoded[..., ::-1][0]
            if self.should_resize_output_back:
                output = cv2.resize(output, self.original_image_size)
            return output

    def find_image_paths(directory):
        return [p for p in sorted(Path(directory).iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    def save_output_image(output_dir, input_path, image):
        import cv2
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / Path(input_path).name
        assert cv2.imwrite(str(target), image)
        return target

    module.InferenceConfig = InferenceConfig
    module.DiffusionPriorEnhancer = Enhancer
    module.find_image_paths = find_image_paths
    module.save_output_image = save_output_image
    module.seed_everything = lambda seed: torch.manual_seed(seed)
    return module


def build(root, drop_key=False, extra_file=False):
    low = root / "low"
    low.mkdir()
    files = []
    for index, hw in enumerate(SIZES):
        y, x = np.indices(hw)
        image = np.stack(((x * 16 + index) % 256, (y * 30) % 256, (x + y) * 7 % 256), -1).astype(np.uint8)
        name = f"syn{index}_low.png"
        PIL.Image.fromarray(image).save(low / name)
        files.append({"name": name, "sha256": runner.sha256_file(low / name), "bytes": (low / name).stat().st_size})
    if extra_file:
        (low / "stray.txt").write_text("x")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"count": len(files), "files": files}))
    state = {"my_vae." + k: v for k, v in StubVAE().state_dict().items()}
    if drop_key:
        state.pop("my_vae.decoder.bias")
    state["other.weight"] = torch.zeros(1)
    ckpt = root / "vae.ckpt"
    torch.save({"state_dict": state, "epoch": 0}, ckpt)
    snap = root / "hub" / "models--sd--v15" / "snapshots" / "rev123"
    (snap / "unet").mkdir(parents=True)
    (snap / "model_index.json").write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "U"]}))
    (snap / "unet" / "config.json").write_text("{}")
    args = argparse.Namespace(
        source_root=root, vae_checkpoint=ckpt, sd_snapshot=snap, low_dir=low, low_receipt=receipt,
        out=root / "out", smoke_one=False, expected_vae_sha256=None, expected_sd_revision=None,
        synthetic=True, expected_count=2,
        # These runner-mechanics tests use a stub `official` module and a CPU torch build
        # that need not match the official pins checked by check_environment_versions();
        # only EnvironmentVersionTests below exercises that gate.
        accept_env_drift=True)
    return args


def patch_versions(torch_version, xformers_version, diffusers_version):
    """Deterministically control what check_environment_versions() sees, regardless of
    what is actually installed locally."""
    saved = {}
    for name, version in (("xformers", xformers_version), ("diffusers", diffusers_version)):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = types.SimpleNamespace(__version__=version)
    torch_patch = unittest.mock.patch.object(runner.torch, "__version__", torch_version)
    torch_patch.start()
    return torch_patch, saved


def unpatch_versions(torch_patch, saved):
    torch_patch.stop()
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def verifier_args(run_args, root):
    return argparse.Namespace(
        low_receipt=run_args.low_receipt, low_dir=run_args.low_dir, outputs=run_args.out,
        out=root / "verify.json", expected_count=2,
        expected_vae_sha256=runner.sha256_file(run_args.vae_checkpoint), expected_sd_revision="rev123",
        expected_output_manifest_sha256="", first_row_tensor_sha256="", require_promotable=False)


def rewrite_row(outputs, index, **changes):
    manifest_path = outputs / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["rows"][index].update(changes)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    row = manifest["rows"][index]
    (outputs / Path(row["low_name"]).stem / "decision.json").write_text(json.dumps(row, indent=2))


class RunnerTests(unittest.TestCase):
    def test_happy_path_and_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build(root)
            manifest = runner.run(args, official=make_official(), require_cuda=False)
            self.assertEqual(manifest["count"], 2)
            self.assertFalse(manifest["promotable"])
            self.assertEqual(manifest["vae_key_report"]["unexpected_keys"], [])
            self.assertEqual(manifest["reference_reads"], 0)
            row = manifest["rows"][0]
            self.assertTrue(row["official_uint8_match"])
            self.assertGreater(row["prequant_fraction_above_255"] + row["prequant_fraction_below_0"], 0)
            with gzip.open(args.out / "syn0_low" / "output.pt.gz", "rb") as stream:
                tensor = torch.load(stream, weights_only=True)
            self.assertEqual(list(tensor.shape), [1, 3, *SIZES[0]])
            self.assertFalse(row["official_resize_back"])
            second = manifest["rows"][1]
            self.assertTrue(second["official_resize_back"])
            self.assertEqual(second["shape"], [1, 3, *SIZES[1]])
            self.assertEqual(second["native_shape"], [1, 3, 16, 24])
            low = np.array(PIL.Image.open(args.low_dir / "syn0_low.png").convert("RGB")).astype(np.float32)
            expected = (((low / 127.5 - 1.0) * 1.02 + 0.003) * 127.5 + 127.5) / 255.0
            np.testing.assert_allclose(tensor[0].numpy().transpose(1, 2, 0), expected, atol=1e-5)
            ok, receipt = verifier.verify(verifier_args(args, root))
            self.assertTrue(ok, receipt["failures"])

    def test_extra_file_in_low_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp), extra_file=True)
            with self.assertRaisesRegex(AssertionError, "exactly the receipt"):
                runner.run(args, official=make_official(), require_cuda=False)

    def test_wrong_low_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            receipt = json.loads(args.low_receipt.read_bytes())
            receipt["files"][1]["sha256"] = "0" * 64
            args.low_receipt.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError):
                runner.run(args, official=make_official(), require_cuda=False)

    def test_missing_vae_key_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp), drop_key=True)
            with self.assertRaisesRegex(AssertionError, "missing keys"):
                runner.run(args, official=make_official(), require_cuda=False)

    def test_capture_must_equal_official_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaisesRegex(AssertionError, "official quantized"):
                runner.run(args, official=make_official("tamper_output"), require_cuda=False)

    def test_unplanned_decode_is_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaises(PermissionError):
                runner.run(args, official=make_official("extra_open"), require_cuda=False)

    def test_full_canonical_requires_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            args.synthetic = False
            args.expected_count = None
            with self.assertRaisesRegex(AssertionError, "requires the published"):
                runner.run(args, official=make_official(), require_cuda=False)

    def test_official_module_cycle_is_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            official = make_official()
            manifest = runner.run(args, official=official, require_cuda=False)
            self.assertTrue(manifest["model_state_unchanged"])
            # The stub reproduces the official self-reference, so the old post-run
            # state_dict() hash recurses.
            config = official.InferenceConfig(str(args.vae_checkpoint), str(args.low_dir), "x", "y")
            model = official.DiffusionPriorEnhancer(config)
            model.process(args.low_dir / "syn0_low.png")
            with self.assertRaises(RecursionError):
                runner.state_sha256(model.unet)
            self.assertIsInstance(runner.tensor_identity(model)["parameters"], frozenset)

    def test_in_place_parameter_change_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaisesRegex(AssertionError, "model state changed"):
                runner.run(args, official=make_official("mutate_param"), require_cuda=False)

    def test_parameter_replacement_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            with self.assertRaisesRegex(AssertionError, "replaced during inference"):
                runner.run(args, official=make_official("replace_param"), require_cuda=False)

    def test_smoke_one_processes_first_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build(Path(tmp))
            args.smoke_one = True
            manifest = runner.run(args, official=make_official(), require_cuda=False)
            self.assertEqual([r["low_name"] for r in manifest["rows"]], ["syn0_low.png"])

    def test_wrong_diffusers_version_fails_unless_accept_env_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build(root)
            torch_patch, saved = patch_versions("2.0.1+cu118", "0.0.20", "0.19.0")
            try:
                args.accept_env_drift = False
                with self.assertRaisesRegex(AssertionError, "diffusers"):
                    runner.run(args, official=make_official(), require_cuda=False)
                args.accept_env_drift = True
                args.out = root / "out2"
                manifest = runner.run(args, official=make_official(), require_cuda=False)
                check = manifest["environment_version_check"]
                self.assertTrue(check["drift"])
                self.assertFalse(check["matched"]["diffusers"])
                self.assertTrue(check["matched"]["torch"])
                self.assertTrue(check["matched"]["xformers"])
            finally:
                unpatch_versions(torch_patch, saved)


class VerifierMutationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = build(self.root)
        runner.run(self.args, official=make_official(), require_cuda=False)
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
        with gzip.open(path, "rb") as stream:
            tensor = torch.load(stream, weights_only=True)
        tensor = (tensor * 0.99).contiguous()
        runner.save_gz(tensor, path)
        rewrite_row(self.args.out, 1, output_tensor_sha256=verifier.thash(tensor),
                    output_file_sha256=verifier.sha256_file(path))
        self.assertFailsWith("map-back recomputation differs")

    def test_float_outside_official_bin(self):
        path = self.args.out / "syn0_low" / "output.pt.gz"
        with gzip.open(path, "rb") as stream:
            tensor = torch.load(stream, weights_only=True)
        tensor = (tensor + 2.0 / 255.0).contiguous()
        runner.save_gz(tensor, path)
        rewrite_row(self.args.out, 0, output_tensor_sha256=verifier.thash(tensor),
                    output_file_sha256=verifier.sha256_file(path))
        self.assertFailsWith("outside the official uint8 truncation bin")

    def test_nan_tensor(self):
        path = self.args.out / "syn0_low" / "output.pt.gz"
        with gzip.open(path, "rb") as stream:
            tensor = torch.load(stream, weights_only=True)
        tensor[0, 0, 0, 0] = float("nan")
        runner.save_gz(tensor, path)
        rewrite_row(self.args.out, 0, output_file_sha256=verifier.sha256_file(path))
        self.assertFailsWith("non-finite")

    def test_low_image_mutation(self):
        path = self.args.low_dir / "syn1_low.png"
        path.write_bytes(path.read_bytes() + b"x")
        self.assertFailsWith("low sha256 does not match frozen receipt")

    def test_nonzero_reference_reads(self):
        manifest_path = self.args.out / "output_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["reference_reads"] = 1
        manifest_path.write_text(json.dumps(manifest))
        self.assertFailsWith("reference_reads")

    def test_wrong_vae_pin(self):
        self.vargs.expected_vae_sha256 = hashlib.sha256(b"other").hexdigest()
        self.assertFailsWith("checkpoint_sha256")


if __name__ == "__main__":
    unittest.main()
