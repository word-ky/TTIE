"""Independent, low-only preregistration verifier. Does not import model/metric code."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_IDS = (
    "retinexformer", "snr_aware", "promptir", "promptir_dctta",
    "mr_illuminate", "quadprior", "ours_step0", "ours_ttt", "zero_ig", "gm_moe",
)
EXPECTED_FROZEN = {"retinexformer", "snr_aware"}
EXPECTED_TIER1 = set(EXPECTED_IDS[:8])
EXPECTED_MANIFEST_SHA = "73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001"
EXPECTED_METRIC_SHA = {
    "research_log/T071B/evaluate.py": "39ae608a0253f506dcd6412b8f4a83b3f966cdb12154c3c6409007865cebc9e7",
    "research_log/T071A/core.py": "45d6b92dee5fb3163da1ea1e9d331a2be36f8991a142bf221167c6d426871858",
    "ttie/ssim_transfer.py": "01d227a8b4caaf8f5705d9c18a5ef685367b6ccf80588d212be92830a04f8556",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=HERE)


def check_promptir_shared_base(base, adapted):
    assert base["shared_base_checkpoint_id"] == adapted["shared_base_checkpoint_id"] == "promptir_dctta_shared_base"
    if base["source_binding_state"] == adapted["source_binding_state"] == "PENDING_SOURCE_BINDING":
        assert base["source_setting"] == adapted["source_setting"] == "PENDING_SOURCE_BINDING"
        assert base["checkpoint_sha256"] is adapted["checkpoint_sha256"] is None
        assert base["binding_sha256"] is adapted["binding_sha256"] is None
    else:
        assert base["source_binding_state"] == adapted["source_binding_state"] == "BOUND"
        assert base["checkpoint_sha256"] and base["checkpoint_sha256"] == adapted["checkpoint_sha256"]
        assert base["source_setting"] and base["source_setting"] == adapted["source_setting"]


def check(registry, plan, historical, manifest):
    assert registry["task"] == plan["task"] == "T073-A"
    assert registry["cohort_size"] == plan["cohort"]["size"] == 150
    assert plan["cohort"]["geometry_wh"] == [3840, 2160]
    assert plan["cohort"]["dispatch_root_sha256"] == historical["cohort"]["dispatch_root_sha256"]
    assert plan["historical_spec"]["sha256"] == "df9c5e4a6c2c5ee8cf812c4535938ecb80d5f6b34fbc659d2fc8302923f93391"
    assert plan["historical_spec"]["immutable"] is True
    metric = plan["metric_source"]
    assert metric["commit"] == historical["metric_source"]["commit"] == "579c3691a80f5b7cadfd706a2fe6750876c53aa0"
    assert metric["entrypoint"] == historical["metric_source"]["entrypoint"]
    assert metric["ssim_entrypoint"] == historical["metric_source"]["ssim_entrypoint"]
    assert metric["source_sha256"] == EXPECTED_METRIC_SHA
    for field in ("reference_conversion", "output_conversion", "crop", "resize", "quantization", "brightness_matching"):
        assert metric[field] == historical["metric_source"][field]
    for path, expected in EXPECTED_METRIC_SHA.items():
        assert sha(git_bytes(metric["commit"], path)) == expected
    boot = plan["bootstrap"]
    assert boot["seed"] == historical["bootstrap"]["seed"] == 20260922
    assert boot["resamples"] == historical["bootstrap"]["resamples"] == 10000
    assert boot["sample_size"] == historical["bootstrap"]["sample_size"] == 150
    assert boot["indices_sha256"] == "f3348c731c348b52eed32160d6b4b2b904101e59a542e1c5dc22850e812da904"
    for field in ("replacement", "rng", "indices", "shared_stream", "interval", "quantiles", "quantile_method", "unit"):
        assert boot[field] == historical["bootstrap"][field]
    assert plan["win_rule"] == historical["endpoints"]["win_rule"]
    assert plan["headline_direction"] == "ours_ttt minus comparator per image"
    assert plan["adaptation_gain_direction"] == "adapted minus base per image"
    assert plan["adaptation_pairs"] == [["promptir_dctta", "promptir"], ["ours_ttt", "ours_step0"]]
    assert plan["sample_policy"]["all_canonical_pairs_required"] is True
    assert plan["sample_policy"]["exclusions"] == []
    assert plan["sample_policy"]["on_nonfinite_metric"] == historical["sample_policy"]["on_nonfinite_metric"]
    assert plan["accounting"] == {"reference_reads": 0, "metrics": 0, "inference_runs_this_task": 0}

    rows = registry["rows"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == len(EXPECTED_IDS) and tuple(ids) == EXPECTED_IDS
    assert registry["frozen_output_manifest"]["sha256"] == EXPECTED_MANIFEST_SHA
    assert manifest["classification"] == registry["frozen_output_manifest"]["classification"]
    assert manifest["reference_reads"] == manifest["metrics"] == 0
    assert set(manifest["methods"]) == EXPECTED_FROZEN
    for method in EXPECTED_FROZEN:
        assert manifest["methods"][method]["count"] == 150
        assert len(manifest["methods"][method]["rows"]) == 150
    by_id = {row["id"]: row for row in rows}
    assert {row["id"] for row in rows if row["tier"] == 1} == EXPECTED_TIER1
    assert by_id["zero_ig"]["tier"] == "preferred_additional"
    assert by_id["gm_moe"]["tier"] == 2
    base, adapted = by_id["promptir"], by_id["promptir_dctta"]
    check_promptir_shared_base(base, adapted)
    assert base["source_binding_state"] == adapted["source_binding_state"] == "PENDING_SOURCE_BINDING"
    assert adapted["reference_access_gate"] == plan["dctta_execution_gate"]["status"] == "LOW_ONLY_DCTTA_REQUIRED"
    assert plan["dctta_execution_gate"]["target_reference_access"] == "NONE"
    assert adapted["target_reference_access"] == "NONE: no GT/reference path resolution, enumeration, open, decode, cache, or dataset-object pass-through"
    assert "low-only wrapper" in plan["dctta_execution_gate"]["paired_loader"]
    assert "equivalence" in plan["dctta_execution_gate"]["before_model_execution"]
    for row in rows:
        assert row["target_gt_during_adaptation"] is False
        assert row["paradigm"] in {"fixed", "domain-level TTA", "zero-shot", "per-image episodic TTT"}
        assert row["source_setting"] and row["source_reference"] and row["transition_evidence"]
        if row["id"] in EXPECTED_FROZEN:
            assert row["status"] == "FROZEN_OUTPUTS"
            assert row["output_count"] == 150
            assert row["output_manifest_sha256"] == EXPECTED_MANIFEST_SHA
            assert row["checkpoint_sha256"] and row["config_sha256"] and row["binding_sha256"]
        else:
            assert row["status"] == "PENDING_OUTPUTS"
            assert row["output_count"] == 0 and row["output_manifest_sha256"] is None
    assert "all eight Tier-1 rows" in plan["reference_gate"]
    assert "ZERO-IG preferred-additional and GM-MoE secondary inclusion locked" in plan["reference_gate"]
    return {"classification": "UHDLL_EXPANDED_MAIN_TABLE_PREREG_SEALED", "rows": len(rows), "frozen_rows": sorted(EXPECTED_FROZEN), "pending_rows": len(rows) - len(EXPECTED_FROZEN), "reference_reads": 0, "metrics": 0}


def verify():
    registry_raw = (HERE / "method_registry.json").read_bytes()
    plan_raw = (HERE / "metric_plan.json").read_bytes()
    registry = json.loads(registry_raw)
    plan = json.loads(plan_raw)
    h = plan["historical_spec"]
    historical_raw = git_bytes(h["commit"], h["path"])
    assert sha(historical_raw) == h["sha256"]
    m = registry["frozen_output_manifest"]
    manifest_raw = git_bytes(m["commit"], m["path"])
    assert sha(manifest_raw) == m["sha256"]
    result = check(registry, plan, json.loads(historical_raw), json.loads(manifest_raw))
    result.update({"registry_sha256": sha(registry_raw), "metric_plan_sha256": sha(plan_raw), "historical_spec_sha256": sha(historical_raw), "baseline_output_manifest_sha256": sha(manifest_raw)})
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
