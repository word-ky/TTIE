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
        ("ZERO-IG mis-tiering", "registry", lambda x: x["rows"][8].update(tier=1)),
        ("PromptIR/DCTTA base mismatch", "registry", lambda x: x["rows"][3].update(shared_base_checkpoint_id="different_base")),
        ("premature source binding", "registry", lambda x: x["rows"][2].update(source_setting="model.ckpt", source_binding_state="BOUND", checkpoint_sha256="a" * 64)),
        ("DCTTA GT permission", "registry", lambda x: x["rows"][3].update(target_reference_access="ALLOW_GT")),
        ("DCTTA plan GT permission", "plan", lambda x: x["dctta_execution_gate"].update(target_reference_access="ALLOW_GT")),
    ]
    for name, target, mutate in mutations:
        r, p = copy.deepcopy(registry), copy.deepcopy(plan)
        mutate(r if target == "registry" else p)
        try:
            verify.check(r, p, historical, manifest)
        except AssertionError:
            continue
        raise AssertionError(f"mutation escaped verifier: {name}")
    bound_base = copy.deepcopy(registry["rows"][2])
    bound_adapted = copy.deepcopy(registry["rows"][3])
    for row in (bound_base, bound_adapted):
        row.update(source_binding_state="BOUND", source_setting="official source A", checkpoint_sha256="a" * 64)
    verify.check_promptir_shared_base(bound_base, bound_adapted)
    bound_adapted["checkpoint_sha256"] = "b" * 64
    try:
        verify.check_promptir_shared_base(bound_base, bound_adapted)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutation escaped verifier: future bound checkpoint mismatch")
    print(json.dumps({"positive": "PASS", "rejected_mutations": len(mutations) + 1, "reference_reads": 0, "metrics": 0, "model_runs": 0}))


if __name__ == "__main__":
    main()
