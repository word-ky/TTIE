"""Synthetic-fixture tests for stage_target.py (no real target data)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_target as st  # noqa: E402

RNG = np.random.default_rng(0)


def dark(h=40, w=60, high=20):
    return RNG.integers(0, high, size=(h, w, 3), dtype=np.uint8)


def args(**kw):
    base = dict(dataset="sid", root=None, out=None, geometry="snr960x512", frames_per_sequence=0,
                smid_test_list=None, expected_count=None, list_only=False)
    base.update(kw)
    return SimpleNamespace(**base)


def make_sid(root):
    for folder, shorts in (("00001", 2), ("10003", 3), ("10006", 2)):
        (root / "short_sid2" / folder).mkdir(parents=True)
        (root / "long_sid2" / folder).mkdir(parents=True)
        for k in range(shorts):
            np.save(root / "short_sid2" / folder / f"{folder}_0{k}_0.1s.npy", dark())
        np.save(root / "long_sid2" / folder / f"{folder}_00_10s.npy", dark(high=250))


class StageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sid_stage_matches_official_loader_and_never_decodes_gt(self):
        make_sid(self.root)
        loaded = []
        original = np.load

        def spy(path, *a, **k):
            loaded.append(str(path))
            return original(path, *a, **k)

        with mock.patch.object(np, "load", spy):
            receipt = st.stage(args(root=self.root, out=self.out))
        self.assertTrue(loaded and all("short_sid2" in p for p in loaded))
        self.assertEqual(receipt["count"], 5)
        self.assertEqual(receipt["cluster_count"], 2)
        self.assertEqual(sorted(p.name for p in (self.out / "low").iterdir()),
                         [row["name"] for row in receipt["files"]])
        row = receipt["files"][0]
        self.assertEqual((row["height"], row["width"]), (512, 960))
        src = self.root / row["source_relpath"]
        official = cv2.resize(np.load(src), (960, 512)).astype(np.float32) / 255.
        official = official[:, :, [2, 1, 0]]
        cached = np.asarray(Image.open(self.out / "low" / row["name"])).astype(np.float32) / 255.
        self.assertTrue(np.array_equal(official, cached))
        self.assertAlmostEqual(row["luma_mean"], st.luma_mean(cached), places=12)
        opaque = json.loads((self.out / "reference_opaque_manifest.json").read_text())
        self.assertEqual(opaque["low_receipt_sha256"], st.sha256_file(self.out / "low_receipt.json"))
        for pair in opaque["pairs"]:
            self.assertEqual(pair["gt_raw_sha256"], st.sha256_file(self.root / pair["gt_relpath"]))
        self.assertEqual(opaque["unique_gt_count"], 2)
        self.assertNotIn("long_sid2", json.dumps(receipt))
        self.assertFalse(any("gt" in key for row in receipt["files"] for key in row))
        self.assertTrue(receipt["darkness"]["pass"])

    def test_mutation_decoding_gt_is_refused(self):
        make_sid(self.root)
        gt = self.root / "long_sid2" / "10003" / "10003_00_10s.npy"
        guard = st.DecodeGuard(st.reference_roots("sid", self.root))
        with self.assertRaises(st.ReferenceAccessError):
            st.decode_low(gt, "snr960x512", guard)
        with guard.active():
            with self.assertRaises(st.ReferenceAccessError):
                np.load(gt)
            with self.assertRaises(st.ReferenceAccessError):
                cv2.imread(str(gt))
            with self.assertRaises(st.ReferenceAccessError):
                Image.open(gt)

    def test_mutation_listing_gt_as_low_fails_closed(self):
        make_sid(self.root)
        real = st.list_sid

        def swapped(root):
            items = real(root)
            items[0]["low_path"] = items[0]["gt_path"]
            return items

        with mock.patch.object(st, "list_sid", swapped):
            with self.assertRaises(AssertionError):
                st.stage(args(root=self.root, out=self.out))

    def test_mutation_flag_off_refuses_to_stage(self):
        make_sid(self.root)
        with mock.patch.object(st, "REFERENCE_DECODE_FORBIDDEN", False):
            with self.assertRaises(st.ReferenceAccessError):
                st.stage(args(root=self.root, out=self.out))
        self.assertTrue(st.REFERENCE_DECODE_FORBIDDEN)

    def test_missing_gt_folder_fails(self):
        make_sid(self.root)
        for p in (self.root / "long_sid2" / "10006").iterdir():
            p.unlink()
        (self.root / "long_sid2" / "10006").rmdir()
        with self.assertRaises(AssertionError):
            st.stage(args(root=self.root, out=self.out))

    def test_duplicate_low_bytes_fail(self):
        make_sid(self.root)
        a, b = sorted((self.root / "short_sid2" / "10003").iterdir())[:2]
        b.write_bytes(a.read_bytes())
        with self.assertRaises(AssertionError):
            st.stage(args(root=self.root, out=self.out))

    def test_identical_gt_across_clusters_fails(self):
        make_sid(self.root)
        src = self.root / "long_sid2" / "10003" / "10003_00_10s.npy"
        (self.root / "long_sid2" / "10006" / "10006_00_10s.npy").write_bytes(src.read_bytes())
        with self.assertRaises(AssertionError):
            st.stage(args(root=self.root, out=self.out))

    def test_non_rgb_npy_fails(self):
        make_sid(self.root)
        np.save(self.root / "short_sid2" / "10003" / "10003_00_0.1s.npy", dark()[:, :, 0])
        with self.assertRaises(AssertionError):
            st.stage(args(root=self.root, out=self.out))

    def test_smid_first30_and_gt_filter(self):
        lq, gt = self.root / "SMID_LQ_np", self.root / "SMID_Long_np"
        for seq in ("0001", "0013", "0020"):
            (lq / seq).mkdir(parents=True)
            (gt / seq).mkdir(parents=True)
            for k in range(32 if seq == "0013" else 3):
                np.save(lq / seq / f"{k:04d}.npy", dark(8, 8))
            (gt / seq / "0000.ARW").write_bytes(b"raw" + seq.encode())
            (gt / seq / "half0001.npy").write_bytes(b"half" + seq.encode())
            np.save(gt / seq / "long.npy", dark(8, 8, 250))
        (gt / "text_list.txt").write_text("0013\n0020\n")
        listing = Path(self.tmp.name) / "list.txt"
        listing.write_text("0013\n0020\n")
        receipt = st.stage(args(dataset="smid", root=self.root, out=self.out, smid_test_list=listing,
                                frames_per_sequence=30))
        self.assertEqual(receipt["count"], 33)
        opaque = json.loads((self.out / "reference_opaque_manifest.json").read_text())
        self.assertTrue(all(p["gt_relpath"].endswith("long.npy") for p in opaque["pairs"]))
        listing.write_text("0013\n0099\n")
        with self.assertRaises(AssertionError):
            st.stage(args(dataset="smid", root=self.root, out=Path(self.tmp.name) / "o2",
                          smid_test_list=listing, frames_per_sequence=30))

    def test_sdsd_suffix_rule_and_count_mismatch(self):
        for folder, n in (("MVI_0898", 2), ("MVI_0898_2", 2), ("MVI_0001", 2)):
            for side in ("input", "GT"):
                (self.root / side / folder).mkdir(parents=True)
                for k in range(n):
                    np.save(self.root / side / folder / f"{k:04d}.npy", dark(8, 8, 250 if side == "GT" else 20))
        with mock.patch.dict(st.SDSD_TEST_DIRS, {"sdsd_out": ["MVI_0898"]}):
            receipt = st.stage(args(dataset="sdsd_out", root=self.root, out=self.out, frames_per_sequence=30))
            self.assertEqual(receipt["count"], 4)
            self.assertEqual(receipt["cluster_count"], 1)
            (self.root / "GT" / "MVI_0898" / "0001.npy").unlink()
            with self.assertRaises(AssertionError):
                st.stage(args(dataset="sdsd_out", root=self.root, out=Path(self.tmp.name) / "o2",
                              frames_per_sequence=30))

    def test_lsrw_pairs_exif_and_missing_high(self):
        for cam in ("Huawei", "Nikon"):
            for side in ("low", "high"):
                (self.root / cam / side).mkdir(parents=True)
                for k in range(2):
                    Image.fromarray(dark(16, 24, 250 if side == "high" else 20)).save(
                        self.root / cam / side / f"{k}.jpg", quality=95)
        receipt = st.stage(args(dataset="lsrw", root=self.root, out=self.out, geometry="native"))
        self.assertEqual(receipt["count"], 4)
        self.assertEqual(receipt["files"][0]["source_format"], "JPEG")
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(dark(16, 24)).save(self.root / "Nikon" / "low" / "1.jpg", exif=exif.tobytes())
        with self.assertRaises(AssertionError):
            st.stage(args(dataset="lsrw", root=self.root, out=Path(self.tmp.name) / "o2", geometry="native"))
        (self.root / "Nikon" / "high" / "1.jpg").unlink()
        with self.assertRaises(AssertionError):
            st.stage(args(dataset="lsrw", root=self.root, out=Path(self.tmp.name) / "o3", geometry="native"))

    def test_darkness_rule(self):
        self.assertTrue(st.darkness_summary([0.05] * 8 + [0.3] * 2)["pass"])
        self.assertFalse(st.darkness_summary([0.05] * 7 + [0.3] * 3)["pass"])  # 70% <= .15
        self.assertFalse(st.darkness_summary([0.11] * 10)["pass"])  # median > .10
        self.assertTrue(st.darkness_summary([0.10] * 10)["pass"])  # boundaries inclusive

    def test_list_only_decodes_nothing(self):
        make_sid(self.root)
        with mock.patch.object(np, "load", side_effect=AssertionError("decode")):
            st.stage(args(root=self.root, out=self.out, list_only=True))
        listing = json.loads((self.out / "test_list.json").read_text())
        self.assertEqual((listing["count"], listing["clusters"]), (5, 2))


if __name__ == "__main__":
    unittest.main()
