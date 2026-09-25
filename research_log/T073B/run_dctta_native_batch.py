"""Native PromptIR+DCTTA low-only outputs: official domain-level adaptation, then sealed 4K inference.

Adaptation follows the construction and loop order of the official
``train_test_promptir.py`` with the task-owned low-only loaders and the
unchanged upstream ``tta.SRTTA``/RDDM. After adaptation the adapted weights are
copied to CPU, every adaptation object is released, and a fresh PromptIR holding
exactly those weights runs the sealed static native schedule. No reference path
is accepted.
"""

import argparse
import gc
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from chunk_conv_schedule import schedule_depthwise, schedule_feedforward
from low_only_loader import LowOnlyPromptTestDataset, LowOnlyPromptTrainDataset
from run_low_only import load_promptir
from spill_forward_schedule import schedule_skip_spill

EXPECTED_COUNT = 150
WAVELET_SHA256 = "8b153d20d0d677f93721b7ee81f2ba9736dabf86bf125e62612ae47e2e87e5ac"
# Same five-task base as the frozen static PromptIR row.
CHECKPOINT_SHA256 = "206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a"
OUTPUT_BYTES_PER_IMAGE = 96 * 2**20
CLEANUP_MARGIN_BYTES = 256 * 2**20


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class RecordedLoader:
    """Yields the unchanged official batches, recording basename and patch hash."""

    def __init__(self, loader, record, expected=None):
        self.loader, self.record, self.expected = loader, record, expected

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.record.append([batch[0][0], tensor_sha256(batch[1])])
            if self.expected is not None:
                assert self.record[-1] == self.expected[len(self.record) - 1], "adaptation order drift"
            yield batch


class Degradation(torch.nn.Module):
    """Official ``train_test_promptir.Degradation`` (its unused ``sample`` omitted)."""

    def __init__(self, diffusion_class, options):
        super().__init__()
        self.gene_net = diffusion_class(options, debug=False)

    def train(self, low, pseudo_target, name):
        return self.gene_net.train(low, pseudo_target, name)


def build_adaptation(args, options, tta, diffusion_class):
    """Official order: seed 23, train set (reseeds 42), loader, PromptIR, SRTTA, RDDM."""
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trainset = LowOnlyPromptTrainDataset(args.low_dir, patch_size=options.patch_size)
    if args.smoke_one:
        trainset = Subset(trainset, [0])
    loader = DataLoader(
        trainset,
        batch_size=options.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=options.num_workers,
    )
    model = load_promptir(args.checkpoint)
    source_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model = tta.configure_model(options, model.cuda())
    params, trainable_names = tta.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=options.lr, betas=options.betas)
    adapted = tta.SRTTA(options, model, optimizer, logging.getLogger(__name__))
    degeneration_model = Degradation(diffusion_class, options).cuda()
    return loader, model, source_state, trainable_names, adapted, degeneration_model


def update_summary(source_state, adapted_state, trainable_names):
    assert list(adapted_state) == list(source_state)
    assert set(trainable_names) <= set(source_state)
    changed = {key: int((adapted_state[key] != source_state[key]).sum()) for key in source_state}
    changed_names = [key for key, count in changed.items() if count]
    summary = {
        "trainable_tensor_count": len(trainable_names),
        "trainable_elements": sum(source_state[key].numel() for key in trainable_names),
        "trainable_bytes": sum(
            source_state[key].numel() * source_state[key].element_size() for key in trainable_names
        ),
        "changed_tensor_count": len(changed_names),
        "changed_elements": sum(changed.values()),
        "changed_bytes": sum(changed[key] * adapted_state[key].element_size() for key in changed_names),
        "changed_outside_trainable": sorted(set(changed_names) - set(trainable_names)),
        "changed_tensor_names": changed_names,
    }
    assert not summary["changed_outside_trainable"]
    assert summary["changed_elements"] > 0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-one", action="store_true")
    parser.add_argument("--order-only", action="store_true")
    parser.add_argument("--expected-order", type=Path)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()
    sys.path.insert(0, str(args.source_root))
    import tta
    import net.model  # noqa: F401  official import order: PromptIR before RDDM
    from RDDM.net import ResidualDiffusionModel

    options = SimpleNamespace(
        lr=2e-4,
        betas=(0.9, 0.999),
        batch_size=1,
        num_workers=args.num_workers,
        patch_size=320,
        iterations=1,
        compute_fisher=1,
        fisher_ratio=0.6,
        teacher_weight=5,
    )
    receipt = json.loads(args.low_receipt.read_bytes())["files"]
    assert len(receipt) == EXPECTED_COUNT
    assert LowOnlyPromptTestDataset(args.low_dir).names == [item["name"] for item in receipt]
    for item in receipt:
        assert sha256_file(args.low_dir / item["name"]) == item["sha256"], item["name"]
    assert sha256_file("wavelet.mat") == WAVELET_SHA256
    expected = receipt[:1] if args.smoke_one else receipt
    if not args.order_only and not args.smoke_one:
        assert args.expected_order is not None, "promotable run requires the sealed --expected-order"
    if not args.order_only:
        assert shutil.disk_usage(args.out.parent).free >= OUTPUT_BYTES_PER_IMAGE * len(expected) + 2**30
        assert shutil.disk_usage(Path.cwd()).free >= 2 * 2**30
    args.out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.out / "adaptation.log", mode="w")],
        force=True,
    )
    run_facts = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "low_receipt_sha256": sha256_file(args.low_receipt),
        "wavelet_sha256": WAVELET_SHA256,
        "options": json.loads(json.dumps(vars(options))),
        "seeds": {"official_set_seed": 23, "train_dataset_reseed": 42},
        "adaptation_scope": "smoke_image0_only_nonfinal" if args.smoke_one else "all_target_low_domain_level",
        "torch_version": torch.__version__,
        "accelerate_env": {key: value for key, value in os.environ.items() if key.startswith("ACCELERATE_")},
    }
    assert run_facts["checkpoint_sha256"] == CHECKPOINT_SHA256

    torch.cuda.reset_peak_memory_stats()
    loader, model, source_state, trainable_names, adapted, degeneration_model = build_adaptation(
        args, options, tta, ResidualDiffusionModel
    )
    fisher_order, adaptation_order = [], []
    if args.order_only:
        for _batch in RecordedLoader(loader, fisher_order):
            pass
        for _batch in RecordedLoader(loader, adaptation_order):
            pass
        order = dict(run_facts, fisher=fisher_order, adaptation=adaptation_order)
        order_path = args.out / "adaptation_order.json"
        order_path.write_text(json.dumps(order, indent=2))
        print(f"ADAPTATION_ORDER_SHA256 {sha256_file(order_path)}", flush=True)
        return
    expected_order = json.loads(args.expected_order.read_bytes()) if args.expected_order else None
    if expected_order is not None:
        assert expected_order["options"] == run_facts["options"]
        assert expected_order["adaptation_scope"] == run_facts["adaptation_scope"]

    adaptation_started = time.perf_counter()
    if options.compute_fisher:
        adapted.compute_fisher(
            RecordedLoader(loader, fisher_order, expected_order and expected_order["fisher"])
        )
    for names, degraded, _unused_low_surrogate in RecordedLoader(
        loader, adaptation_order, expected_order and expected_order["adaptation"]
    ):
        restored = adapted(loader, degraded.cuda(), names, degeneration_model)
    torch.cuda.synchronize()
    adaptation_seconds = time.perf_counter() - adaptation_started
    adaptation_peak = torch.cuda.max_memory_reserved()
    assert len(fisher_order) == len(adaptation_order) == len(loader)
    if expected_order is not None:
        assert fisher_order == expected_order["fisher"]
        assert adaptation_order == expected_order["adaptation"]
    adapted_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert all(torch.isfinite(value).all() for value in adapted_state.values() if value.is_floating_point())
    adapted_sha = state_sha256(adapted_state)
    updates = update_summary(source_state, adapted_state, trainable_names)
    state_file = args.out / "adapted_state.pt"
    torch.save(adapted_state, state_file)

    # Memory-lifetime-only cleanup; the adapted weights are already copied above.
    del restored, names, degraded, _unused_low_surrogate
    del loader, model, adapted, degeneration_model
    gc.collect()
    torch.cuda.empty_cache()
    post_cleanup_allocated = torch.cuda.memory_allocated()
    post_cleanup_reserved = torch.cuda.memory_reserved()
    assert post_cleanup_allocated <= CLEANUP_MARGIN_BYTES, post_cleanup_allocated

    torch.cuda.reset_peak_memory_stats()
    model = load_promptir(args.checkpoint)
    model.load_state_dict(adapted_state, strict=True)
    model = model.cuda().eval()
    assert state_sha256(model.state_dict()) == adapted_sha
    # Same numeric settings as the static row's process (accelerate may enable TF32 via env).
    assert not torch.backends.cuda.matmul.allow_tf32
    run_facts["inference_numerics"] = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    scheduled_depthwise = schedule_depthwise(model, rows=256, threshold=250_000_000)
    scheduled_feedforward = schedule_feedforward(model, rows=64, threshold=250_000_000)
    schedule_skip_spill(model, threshold=1_000_000)
    dataset = LowOnlyPromptTestDataset(args.low_dir)
    inference_started = time.perf_counter()
    rows = []
    for index, item in enumerate(expected):
        row_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        name, low = dataset[index]
        low_path = args.low_dir / name
        low_sha = sha256_file(low_path)
        assert name == item["name"] and low_sha == item["sha256"]
        assert low.shape == (3, 2160, 3840)
        with torch.no_grad():
            output = model(low.unsqueeze(0).cuda())
        torch.cuda.synchronize()
        assert output.shape == (1, 3, 2160, 3840)
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        output_cpu = output.detach().cpu().contiguous()
        del output
        tensor_sha = hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest()
        output_dir = args.out / Path(name).stem
        output_dir.mkdir()
        output_file = output_dir / "output.pt.gz"
        with gzip.open(output_file, "wb", compresslevel=1) as stream:
            torch.save(output_cpu, stream)
        del output_cpu, low
        row = {
            "low_name": name,
            "low_sha256": low_sha,
            "shape": [1, 3, 2160, 3840],
            "dtype": "torch.float32",
            "output_tensor_sha256": tensor_sha,
            "output_file_sha256": sha256_file(output_file),
            "whole_run_seconds": time.perf_counter() - row_started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
        }
        (output_dir / "decision.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"{index + 1}/{len(expected)} {name} {row['whole_run_seconds']:.3f}s", flush=True)
    state_unchanged = all(
        torch.equal(value.detach().cpu(), adapted_state[key]) for key, value in model.state_dict().items()
    )
    assert state_unchanged
    manifest = dict(
        run_facts,
        method="PromptIR+DCTTA-five-task-domain-level-native-schedule"
        + ("-SMOKE-NONFINAL" if args.smoke_one else ""),
        promotable=not args.smoke_one,
        count=len(rows),
        expected_order_sha256=sha256_file(args.expected_order) if args.expected_order else None,
        fisher_order=fisher_order,
        adaptation_order=adaptation_order,
        adaptation_seconds=adaptation_seconds,
        adaptation_peak_gpu_memory_bytes=adaptation_peak,
        post_cleanup_allocated_bytes=post_cleanup_allocated,
        post_cleanup_reserved_bytes=post_cleanup_reserved,
        inference_seconds=time.perf_counter() - inference_started,
        inference_peak_gpu_memory_bytes=max(row["peak_gpu_memory_bytes"] for row in rows),
        whole_run_seconds=time.perf_counter() - started,
        gpu_name=torch.cuda.get_device_name(),
        source_state_sha256=state_sha256(source_state),
        adapted_state_sha256=adapted_sha,
        adapted_state_file_sha256=sha256_file(state_file),
        parameter_updates=updates,
        scheduled_depthwise_count=len(scheduled_depthwise),
        scheduled_feedforward_count=len(scheduled_feedforward),
        model_state_unchanged=state_unchanged,
        state_key_count=len(adapted_state),
        reference_reads=0,
        metrics=0,
        rows=rows,
    )
    manifest_path = args.out / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"OUTPUT_MANIFEST_SHA256 {sha256_file(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()
