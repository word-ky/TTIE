"""Compare task-owned Step0 with the first state of frozen T070-A trajectory."""

import argparse
import gzip

import torch

from research_log.T062CR2.core import trajectory
from research_log.T070A.infer import FinalOurs
from research_log.T073C.run_step0 import render_step0
from ttie.lolv2_gamma_core import native_rgb
from ttie.semantic_ttt import FixedObjective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = FinalOurs(args.manifest)
    low = native_rgb(args.low)
    step0, step0_gate = render_step0(model, low)
    full_gate = FixedObjective(model.scorer, low.cuda(), model.gate)
    trace = trajectory(low.cuda(), full_gate)
    assert torch.equal(step0_gate.active, full_gate.active)
    assert torch.equal(step0, trace["images"][0])
    with gzip.open(args.output, "rb") as stream:
        persisted = torch.load(stream, map_location="cpu", weights_only=True)
    assert torch.equal(step0, persisted)
    assert torch.isfinite(persisted).all()
    print("STEP0_EXACT_TRAJECTORY_PREFIX_AND_GZIP_ROUNDTRIP_PASS")


if __name__ == "__main__":
    main()
