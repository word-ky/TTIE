"""Mutation tests for verify_generic_outputs.py and the make_generic derivation audit."""

import copy
import gzip
import hashlib
import json
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_generic  # noqa: E402
import verify_generic_outputs as vg  # noqa: E402

H, W = 8, 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thash(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def write_row(root, name, tensor, extra):
    directory = root / Path(name).stem
    directory.mkdir(parents=True)
    with gzip.open(directory / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)
    row = {"low_name": name, "low_sha256": "l" * 64, "shape": [1, 3, H, W], "dtype": "torch.float32",
           "output_tensor_sha256": thash(tensor), "output_file_sha256": sha(directory / "output.pt.gz"), **extra}
    (directory / "decision.json").write_text(json.dumps(row, indent=2))
    return row


class Fixture:
    def __init__(self, root):
        self.root = root
        self.names = ["a__0.png", "b__0.png", "b__1.png"]
        self.receipt = root / "low_receipt.json"
        self.receipt.write_text(json.dumps({"files": [
            {"name": n, "sha256": "l" * 64, "height": H, "width": W} for n in self.names]}))
        self.execution = root / "exec.json"
        self.execution.write_text("{}")
        common = {"low_receipt_sha256": sha(self.receipt), "reference_reads": 0, "metrics": 0}
        self.static = root / "static"
        rows = [write_row(self.static, n, torch.rand(1, 3, H, W), {}) for n in self.names]
        self.write(self.static, dict(common, count=3, model_state_unchanged=True, rows=rows))
        self.step0 = root / "step0"
        actives = [[False] * 4, [True, False, False, False], [False] * 4]
        rows = [write_row(self.step0, n, torch.rand(1, 3, H, W),
                          {"execution_manifest_sha256": sha(self.execution), "active": act})
                for n, act in zip(self.names, actives)]
        self.write(self.step0, dict(common, count=3, execution_manifest_sha256=sha(self.execution), rows=rows))
        self.ttt = root / "ttt"
        ttt_rows = []
        for row, n in zip(rows, self.names):
            if any(row["active"]):
                extra = {"execution_manifest_sha256": row["execution_manifest_sha256"], "decision_status": "TTT_EXECUTED",
                         "decision": {"selected_step": 20}, "selected_state_sha256": "s" * 64}
                ttt_rows.append(write_row(self.ttt, n, torch.rand(1, 3, H, W), extra))
            else:
                src = self.step0 / Path(n).stem
                dst = self.ttt / Path(n).stem
                dst.mkdir(parents=True)
                (dst / "output.pt.gz").write_bytes((src / "output.pt.gz").read_bytes())
                out = dict(row, decision_status="TTT_ABSTAIN_NO_ACTIVE_GATE", optimizer_updates=0,
                           decision={"selected_step": 0, "status": "TTT_ABSTAIN_NO_ACTIVE_GATE"},
                           selected_state_sha256=vg.ZERO_STATE_SHA256)
                (dst / "decision.json").write_text(json.dumps(out, indent=2))
                ttt_rows.append(out)
        self.write(self.ttt, dict(common, count=3, execution_manifest_sha256=sha(self.execution),
                                  step0_output_manifest_sha256=sha(self.step0 / "output_manifest.json"),
                                  rows=ttt_rows))

    @staticmethod
    def write(directory, manifest):
        (directory / "output_manifest.json").write_text(json.dumps(manifest, indent=2))


class VerifyGenericTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def ok(self):
        vg.verify("promptir", self.f.receipt, self.f.static, 3)
        vg.verify("ours_step0", self.f.receipt, self.f.step0, 3, execution_manifest=self.f.execution)
        return vg.verify("ours_ttt", self.f.receipt, self.f.ttt, 3, self.f.step0, self.f.execution)

    def test_clean_fixture_passes(self):
        receipt = self.ok()
        self.assertEqual(receipt["ttt_abstain_no_active_gate_count"], 2)

    def mutate_manifest(self, directory, fn):
        path = directory / "output_manifest.json"
        manifest = json.loads(path.read_text())
        fn(manifest)
        path.write_text(json.dumps(manifest, indent=2))

    def test_mutations_fail(self):
        mutations = [
            lambda: self.mutate_manifest(self.f.static, lambda m: m.update(reference_reads=1)),
            lambda: self.mutate_manifest(self.f.static, lambda m: m.update(metrics=1)),
            lambda: self.mutate_manifest(self.f.static, lambda m: m.update(promotable=False)),
            lambda: self.mutate_manifest(self.f.static, lambda m: m["rows"].pop()),
            lambda: self.mutate_manifest(self.f.static, lambda m: m["rows"][0].update(shape=[1, 3, W, H])),
            lambda: (self.f.static / "b__0" / "decision.json").write_text("{}"),
            lambda: self._nan_tensor(),
            lambda: self.f.receipt.write_text(self.f.receipt.read_text() + " "),
            lambda: self.mutate_manifest(self.f.ttt, lambda m: m["rows"][0].update(output_tensor_sha256="0" * 64)),
            lambda: self.mutate_manifest(self.f.step0, lambda m: m["rows"][1].update(active=[False] * 4)),
            lambda: self.mutate_manifest(self.f.ttt, lambda m: m["rows"][1].update(decision_status="TTT_ABSTAIN_NO_ACTIVE_GATE")),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.tearDown()
                self.setUp()
                mutate()
                with self.assertRaises((AssertionError, KeyError, RuntimeError, EOFError, OSError)):
                    self.ok()

    def _nan_tensor(self):
        directory = self.f.static / "a__0"
        tensor = torch.full((1, 3, H, W), float("nan"))
        with gzip.open(directory / "output.pt.gz", "wb", compresslevel=1) as stream:
            torch.save(tensor, stream)


class DerivationAuditTest(unittest.TestCase):
    def test_generic_files_equal_declared_derivation(self):
        for name in make_generic.SPECS:
            with self.subTest(name=name):
                committed = (HERE / name).read_text(encoding="utf-8").replace("\r\n", "\n")
                self.assertEqual(committed, make_generic.derive(name))
                py_compile.compile(str(HERE / name), doraise=True)

    def test_undeclared_edit_detected(self):
        name = "run_dctta_generic.py"
        tampered = make_generic.derive(name).replace("lr=2e-4", "lr=3e-4")
        self.assertNotEqual(tampered, make_generic.derive(name))


if __name__ == "__main__":
    unittest.main()
