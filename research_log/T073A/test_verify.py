"""Focused fail-closed mutations of the preregistration, without target data."""

import copy
import json

import verify


def main():
    registry = json.loads((verify.HERE / "method_registry.json").read_text(encoding="utf-8"))
    plan = json.loads((verify.HERE / "metric_plan.json").read_text(encoding="utf-8"))
    h = plan["historical_spec"]
    historical = json.loads(verify.git_bytes(h["commit"], h["path"]))
    m = registry["frozen_output_manifest"]
    manifest = json.loads(verify.git_bytes(m["commit"], m["path"]))
    verify.check(registry, plan, historical, manifest)
    mutations = [
        ("missing method", "registry", lambda x: x["rows"].pop()),
        ("duplicate method", "registry", lambda x: x["rows"][2].update(id="snr_aware")),
        ("false frozen row", "registry", lambda x: x["rows"][2].update(status="FROZEN_OUTPUTS")),
        ("wrong frozen count", "registry", lambda x: x["rows"][0].update(output_count=149)),
        ("GT flag", "registry", lambda x: x["rows"][4].update(target_gt_during_adaptation=True)),
        ("cohort size", "plan", lambda x: x["cohort"].update(size=149)),
        ("metric source", "plan", lambda x: x["metric_source"]["source_sha256"].update({"ttie/ssim_transfer.py": "0" * 64})),
        ("bootstrap seed", "plan", lambda x: x["bootstrap"].update(seed=1)),
        ("bootstrap count", "plan", lambda x: x["bootstrap"].update(resamples=9999)),
        ("comparison direction", "plan", lambda x: x.update(headline_direction="comparator minus ours_ttt")),
        ("reference read", "plan", lambda x: x["accounting"].update(reference_reads=1)),
    ]
    for name, target, mutate in mutations:
        r, p = copy.deepcopy(registry), copy.deepcopy(plan)
        mutate(r if target == "registry" else p)
        try:
            verify.check(r, p, historical, manifest)
        except AssertionError:
            continue
        raise AssertionError(f"mutation escaped verifier: {name}")
    print(json.dumps({"positive": "PASS", "rejected_mutations": len(mutations), "reference_reads": 0, "metrics": 0}))


if __name__ == "__main__":
    main()
