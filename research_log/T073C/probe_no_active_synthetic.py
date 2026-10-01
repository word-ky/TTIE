"""Synthetic-only proof probe for the frozen no-active assertion boundary.

This does NOT authorize a target-method change. It removes only the assertion
in an in-memory copy of the frozen trajectory to inspect mathematical behavior.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch

from research_log.T062CR2 import core
from research_log.T070A import infer
from ttie.common_gain import CommonRegion2
from ttie.semantic_ttt import FixedObjective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = infer.FinalOurs(args.manifest)
    torch.manual_seed(23)
    low = torch.rand(1, 3, 352, 352)
    image = low.cuda()
    gate = FixedObjective(model.scorer, image, model.gate)
    gate.active.zero_()
    source = inspect.getsource(core.trajectory)
    original_assertion = "assert model.raw.numel()==12 and gate.active.any()"
    assert source.count(original_assertion) == 1
    namespace = dict(core.__dict__)
    exec(source.replace(original_assertion, "assert model.raw.numel()==12"), namespace)
    trace = namespace["trajectory"](image, gate)
    with torch.no_grad():
        step0 = CommonRegion2(gate.active).to(image)(image).detach().cpu()
    try:
        decision, _, _ = infer.select_trajectory(image, trace, model.model)
        selector = {"status": "selected", "selected_step": decision["selected_step"]}
    except AssertionError:
        selector = {"status": "frozen selector asserted on zero improvement"}
    results = {
        "input": "torch.manual_seed(23) synthetic 352x352 float32; gate.active forced all false",
        "modified_probe_only": "removed gate.active.any assertion from in-memory trajectory copy",
        "all_images_equal_step0": bool(torch.equal(trace["images"], step0.expand_as(trace["images"]))),
        "all_states_equal_step0": bool(torch.equal(trace["states"], trace["states"][0].expand_as(trace["states"]))),
        "selector": selector,
        "reference_reads": 0,
        "target_model_runs": 0,
        "metrics": 0,
    }
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
