"""PromptIR / DCTTA execution using only target low images.

This calls the pinned upstream model and TTA implementation. It does not
import the upstream paired-data entrypoints or accept a reference path.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from low_only_loader import LowOnlyPromptTestDataset, LowOnlyPromptTrainDataset


def load_promptir(checkpoint_path):
    from net.model import PromptIR

    model = PromptIR(decoder=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.replace("net.", ""): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model


def run(args):
    sys.path.insert(0, str(args.source_root))
    import tta
    from RDDM.net import ResidualDiffusionModel
    from utils.image_io import save_image_tensor

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options = SimpleNamespace(
        lr=2e-4,
        betas=(0.9, 0.999),
        batch_size=1,
        num_workers=16,
        patch_size=320,
        iterations=1,
        compute_fisher=1,
        fisher_ratio=0.6,
        teacher_weight=5,
    )
    model = load_promptir(args.checkpoint).cuda()

    if args.mode == "dctta":
        trainset = LowOnlyPromptTrainDataset(args.low_dir, patch_size=options.patch_size)
        if args.smoke_one:
            trainset = Subset(trainset, [0])
        train_loader = DataLoader(
            trainset,
            batch_size=options.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=options.num_workers,
        )
        model = tta.configure_model(options, model)
        params, _ = tta.collect_params(model)
        optimizer = torch.optim.Adam(params, lr=options.lr, betas=options.betas)
        adapted = tta.SRTTA(options, model, optimizer, logging.getLogger(__name__))

        class Degradation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gene_net = ResidualDiffusionModel(options, debug=False)

            def train(self, low, pseudo_target, name):
                return self.gene_net.train(low, pseudo_target, name)

        degeneration_model = Degradation().cuda()
        if options.compute_fisher:
            adapted.compute_fisher(train_loader)
        for names, degraded, _unused_low_surrogate in train_loader:
            adapted(train_loader, degraded.cuda(), names, degeneration_model)

    testset = LowOnlyPromptTestDataset(args.low_dir)
    if args.smoke_one:
        testset = Subset(testset, [0])
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=options.num_workers)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for names, degraded in test_loader:
        with torch.no_grad():
            restored = model(degraded.cuda())
        save_image_tensor(restored, str(args.output_dir / (Path(names[0]).stem + ".png")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "dctta"), required=True)
    parser.add_argument("--smoke-one", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
