"""T074-C phase 4: frozen-T070-A Ours-TTT with declared knob overrides, selected on target GT.

User decision research_log/T074A/ours_target_tuning_decision.md; protocol_draft.md section 8. Run from the
frozen Ours source root (FinalOurs binds cwd-relative paths), with the frozen process environment.

* Refuses to start unless the reference gate receipt validates (metrics.Context).
* Knob overrides never edit frozen files. The four functions that hard-code a knob are re-derived at run time
  from the hash-bound frozen source text by the declared literal substitutions in DERIVATIONS (each must match
  exactly once); everything else (CLIP gate, renderer, CommonBox, features, probability model, rho) is the
  frozen code object itself. The default setting runs first and must reproduce the frozen Ours-TTT row's
  output tensors bit for bit, or the harness stops.
* Every setting tried is appended to a hash-chained JSONL log (knobs, round, grid SHA, per-image metrics, means,
  hashes). The default-reproduction check reruns at every process start, before any GT decode.
* Rounds share one --work directory: ``--grid tuning_grid_round1.json`` (round 1; a re-run may use a superset),
  ``--round2`` (grid built from the round-1 log by ``round2_grid``), then ``--materialize`` once.
* Selection (global, over the union of all rounds' COMPLETE records): highest image-mean PSNR; settings within
  PSNR_TIE_DB of the best tie and are ranked by mean RGB-SSIM, then by L1 distance from the defaults with each
  scalar component normalised by its range over the union of declared settings, then setting id.
  The selected outputs become the separate row ``ours_ttt_target_tuned`` with its own output manifest.
"""

import argparse
import ast
import copy
import gzip
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as metrics_mod  # noqa: E402
from reference_gate import require, sha256_bytes, sha256_file, load_json, utc, TUNED_ROW  # noqa: E402

PSNR_TIE_DB = 0.01  # protocol_draft.md section 8.3
LAMBDA_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)  # research_log/T067B/core.py GRID
DEFAULTS = {
    "q_joint": 1.053775168916056,
    "tau": [0.027419920079410076, 0.001673370413482167],
    "probability_threshold": 0.5,
    "lambda_value": 0.875,
    "lr": 0.03,
    "updates": 27,
    "loss_weights": [1.0, 10.0, 5.0],
    "exposure_target": 0.6,
}
KNOB_LOCATIONS = {
    "q_joint": "gate asset T022A_gate.json q_joint (FinalOurs.gate) -> ttie/joint_gate.py both_gates: active iff evidence > q_joint",
    "tau": "gate asset T022A_gate.json calibration.tau -> ttie/clip_signal.py decisions, ttie/joint_gate.py winning_evidence",
    "probability_threshold": "research_log/T067B/core.py choices: first step with probabilities[k]>=.5 (k_FS)",
    "lambda_value": "research_log/T070A/infer.py select_trajectory: c['lambda_value']==.875 (T067B GRID values only)",
    "lr": "research_log/T062CR2/core.py trajectory: Adam lr=.03",
    "updates": "research_log/T062CR2/core.py trajectory: range(28), step==27, selected=27",
    "loss_weights": "research_log/T062CR2/core.py trajectory: parts @ [1.,10.,5.] (spatial, exposure, colour)",
    "exposure_target": "research_log/T062A/core.py losses: (avg_pool16 mean - .60)^2",
}
TRAJECTORY_KNOBS = ("q_joint", "tau", "lr", "updates", "loss_weights", "exposure_target")
# (module, function, [(frozen text, replacement)], {global name: (module, derived function)})
DERIVATIONS = (
    ("research_log.T062A.core", "losses",
     (("-.60).square()", "-K['exposure_target']).square()"),), {}),
    ("research_log.T062CR2.core", "trajectory",
     (("lr=.03", "lr=K['lr']"),
      ("range(28)", "range(K['updates']+1)"),
      ("parts.new_tensor([1.,10.,5.])", "parts.new_tensor(K['loss_weights'])"),
      ("if step==27:break", "if step==K['updates']:break"),
      ("selected=27", "selected=K['updates']")),
     {"losses": "research_log.T062A.core.losses"}),
    ("research_log.T067B.core", "choices",
     (("probabilities[k]>=.5", "probabilities[k]>=K['probability_threshold']"),), {}),
    ("research_log.T070A.infer", "select_trajectory",
     (("c['lambda_value']==.875", "c['lambda_value']==K['lambda_value']"),),
     {"choices": "research_log.T067B.core.choices"}),
)


# ---------------------------------------------------------------- knobs and grid


def _float(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number")
    value = float(value)
    require(value == value and abs(value) != float("inf"), f"{name} must be finite")
    return value


def canonical_knobs(overrides):
    unknown = set(overrides) - set(DEFAULTS)
    require(not unknown, f"unknown knobs {sorted(unknown)}; exposed: {sorted(DEFAULTS)}")
    k = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(overrides)}
    k["q_joint"] = _float(k["q_joint"], "q_joint")
    require(isinstance(k["tau"], list) and len(k["tau"]) == 2, "tau must be a 2-list")
    k["tau"] = [_float(v, "tau") for v in k["tau"]]
    # phase34_decisions.md: tau is not tuned (its calibration maxima would become inconsistent)
    require(k["tau"] == DEFAULTS["tau"], "tau is fixed at the frozen calibration")
    k["probability_threshold"] = _float(k["probability_threshold"], "probability_threshold")
    require(0 < k["probability_threshold"] <= 1, "probability_threshold must be in (0, 1]")
    k["lambda_value"] = _float(k["lambda_value"], "lambda_value")
    require(k["lambda_value"] in LAMBDA_GRID, "lambda_value must be a T067B GRID value")
    k["lr"] = _float(k["lr"], "lr")
    require(k["lr"] > 0, "lr must be > 0")
    require(isinstance(k["updates"], int) and not isinstance(k["updates"], bool) and 1 <= k["updates"] <= 27,
            "updates must be an int in [1, 27] (frozen T065A step_fraction feature is k/27)")
    require(isinstance(k["loss_weights"], list) and len(k["loss_weights"]) == 3, "loss_weights must be a 3-list")
    k["loss_weights"] = [_float(v, "loss_weights") for v in k["loss_weights"]]
    require(min(k["loss_weights"]) >= 0 and max(k["loss_weights"]) > 0, "loss_weights must be >= 0, not all 0")
    k["exposure_target"] = _float(k["exposure_target"], "exposure_target")
    require(0 < k["exposure_target"] < 1, "exposure_target must be in (0, 1)")
    return k


def setting_id(knobs):
    return sha256_bytes(json.dumps(knobs, sort_keys=True, separators=(",", ":")).encode())


def expand_grid(spec):
    """Declared settings: explicit list, one-factor-at-a-time ({knob: values}, each value alone at defaults)
    and/or cartesian product; defaults first; duplicates of the defaults dropped, other duplicates refused."""
    raw = list(spec.get("settings", []))
    for name, values in spec.get("one_factor", {}).items():
        raw += [{name: value} for value in values]
    product = spec.get("product", {})
    names = sorted(product)
    raw += [dict(zip(names, combo)) for combo in itertools.product(*(product[n] for n in names))]
    settings = [canonical_knobs({})] + [canonical_knobs(s) for s in raw]
    ids = [setting_id(s) for s in settings]
    if ids.count(ids[0]) > 1:  # defaults declared explicitly as well
        settings = [settings[0]] + [s for s, i in zip(settings[1:], ids[1:]) if i != ids[0]]
        ids = [setting_id(s) for s in settings]
    require(len(set(ids)) == len(ids), "duplicate settings in grid")
    return settings


def flat(knobs):
    return [v for name in sorted(DEFAULTS) for v in (knobs[name] if isinstance(knobs[name], list) else [knobs[name]])]


def distance(knobs, settings):
    """L1 distance from the defaults; each scalar component divided by its range over the declared grid.

    Vector knobs count per component: a loss_weights change adds one term per changed weight (tau is fixed).
    """
    columns = list(zip(*(flat(s) for s in settings)))
    default = flat(canonical_knobs({}))
    return float(sum(abs(v - d) / (max(c) - min(c)) for v, d, c in zip(flat(knobs), default, columns) if max(c) > min(c)))


def select(records, settings):
    complete = [r for r in records if r["status"] == "COMPLETE"]
    require(complete, "no complete setting to select")
    best = max(r["summary"]["mean_psnr"] for r in complete)
    window = [r for r in complete if r["summary"]["mean_psnr"] >= best - PSNR_TIE_DB]
    chosen = min(window, key=lambda r: (-r["summary"]["mean_rgb_ssim"], distance(r["knobs"], settings), r["setting_id"]))
    return chosen, [r["setting_id"] for r in window]


def union_settings(records):
    """Every declared-and-logged setting across rounds (COMPLETE or FAILED), first occurrence order."""
    seen, out = set(), []
    for r in records:
        if r["setting_id"] not in seen:
            seen.add(r["setting_id"])
            out.append(r["knobs"])
    return out


def round2_grid(records, round1_settings):
    """Round-2 settings from the round-1 log (rule fixed by the research lead, 2026-09-26).

    For each knob varied alone in round 1: gain = max mean PSNR over its COMPLETE one-factor settings (default
    included) - default mean PSNR. Up to 3 knobs with the largest strictly positive gain (ties: higher SSIM of
    that knob's best setting, then knob name). Per selected knob, its top-2 values by mean PSNR among the same
    settings (ties: SSIM, then smaller normalised distance from default over the round-1 grid). Grid = full
    product of those values, all other knobs at default (a loss_weights vector is one value). FAILED settings
    are never counted. Returns (settings, explanation).
    """
    defaults = canonical_knobs({})
    r1 = {r["setting_id"]: r for r in records if r["round"] == 1 and r["status"] == "COMPLETE"}
    default = r1.get(setting_id(defaults))
    require(default is not None, "round 1 has no COMPLETE default setting")
    members = {}
    for r in r1.values():
        changed = [k for k in DEFAULTS if r["knobs"][k] != defaults[k]]
        if len(changed) == 1:
            members.setdefault(changed[0], [default]).append(r)

    def rank(r):
        return (-r["summary"]["mean_psnr"], -r["summary"]["mean_rgb_ssim"], distance(r["knobs"], round1_settings))

    gains = {}
    for knob, rs in members.items():
        best = min(rs, key=rank)
        gains[knob] = (best["summary"]["mean_psnr"] - default["summary"]["mean_psnr"], best["summary"]["mean_rgb_ssim"])
    chosen = sorted((k for k, (g, _) in gains.items() if g > 0), key=lambda k: (-gains[k][0], -gains[k][1], k))[:3]
    values = {k: [r["knobs"][k] for r in sorted(members[k], key=rank)[:2]] for k in chosen}
    settings = [canonical_knobs(dict(zip(chosen, combo))) for combo in itertools.product(*(values[k] for k in chosen))] \
        if chosen else []
    explanation = {"gains": {k: g for k, (g, _) in gains.items()}, "selected_knobs": chosen, "values": values}
    return settings, explanation


# ---------------------------------------------------------------- derivation from frozen source


def derive(knobs, binding):
    """Derived functions with knob literals replaced; frozen source bytes checked against the T070-A binding."""
    import importlib

    K = dict(knobs)
    derived, texts = {}, []
    for module_name, name, substitutions, links in DERIVATIONS:
        module = importlib.import_module(module_name)
        relpath = module_name.replace(".", "/") + ".py"
        raw = Path(module.__file__).read_bytes()
        require(sha256_bytes(raw) == binding[relpath], f"frozen source changed: {relpath}")
        text = raw.decode()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        source = ast.get_source_segment(text, node)
        for old, new in substitutions:
            require(source.count(old) == 1, f"{relpath}:{name}: {old!r} must occur exactly once")
            source = source.replace(old, new)
        namespace = dict(vars(module))
        namespace.update({g: derived[target] for g, target in links.items()})
        namespace["K"] = K
        exec(compile(source, f"<T074C derived {module_name}.{name}>", "exec"), namespace)
        derived[f"{module_name}.{name}"] = namespace[name]
        texts.append(source)
    grid = importlib.import_module("research_log.T067B.core").GRID
    require(tuple(float(g) for g in grid) == LAMBDA_GRID, "T067B GRID changed")
    return derived, sha256_bytes("\n\n".join(texts).encode())


def gate_receipt_for(gate, knobs):
    receipt = copy.deepcopy(gate)
    receipt["q_joint"] = knobs["q_joint"]
    receipt["calibration"]["tau"] = list(knobs["tau"])
    return receipt


def run_group(scorer, gate, prob_model, low, variants, device):
    """One gate + trajectory per image for settings sharing TRAJECTORY_KNOBS; per-variant stop selection.

    Returns {setting_id: (image [1,3,H,W] float32 cpu or None, record)}. Mirrors FinalOurs.__call__ and the
    sealed no-active rule (Step0 render CommonRegion2(active)(x) when the gate has no active region). An
    AssertionError raised by the frozen code under a knob setting (e.g. no step reaches the probability
    threshold) marks that image REJECTED for the affected settings; any other exception propagates.
    """
    import torch
    from ttie.common_gain import CommonRegion2
    from ttie.semantic_ttt import FixedObjective

    first = variants[0]
    x = low.to(device)
    objective = FixedObjective(scorer, x, gate_receipt_for(gate, first["knobs"]))
    active = objective.active.cpu().tolist()
    if not objective.active.any():
        with torch.no_grad():
            image = CommonRegion2(objective.active).to(x)(x).detach().cpu()
        record = {"status": "TTT_ABSTAIN_NO_ACTIVE_GATE", "selected_step": 0, "active": active}
        return {v["id"]: (image, dict(record)) for v in variants}
    try:
        trace = first["derived"]["research_log.T062CR2.core.trajectory"](x, objective)
    except AssertionError as exc:
        return {v["id"]: (None, {"status": "REJECTED", "active": active, "error": f"trajectory: {exc!r}"}) for v in variants}
    out = {}
    for v in variants:
        try:
            decision, _, _ = v["derived"]["research_log.T070A.infer.select_trajectory"](x, trace, prob_model)
        except AssertionError as exc:
            out[v["id"]] = (None, {"status": "REJECTED", "active": active, "error": f"selection: {exc!r}"})
            continue
        image = trace["images"][decision["selected_step"]].clone().contiguous()
        require(bool(torch.isfinite(image).all()), "nonfinite output")
        out[v["id"]] = (image, {"status": "TTT_EXECUTED", "selected_step": decision["selected_step"],
                                "active": active, "decision": decision})
    return out


# ---------------------------------------------------------------- append-only log


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_log(path):
    records, previous = [], None
    if not Path(path).exists():
        return records
    for number, line in enumerate(Path(path).read_text().splitlines()):
        record = json.loads(line)
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        require(record["record_index"] == number and record["prev_record_sha256"] == previous,
                f"tuning log chain broken at record {number}")
        require(sha256_bytes(canonical(body).encode()) == record["record_sha256"], f"tuning log record {number} altered")
        records.append(record)
        previous = record["record_sha256"]
    return records


def append_log(path, records, body):
    body = dict(body, record_index=len(records), prev_record_sha256=records[-1]["record_sha256"] if records else None)
    record = dict(body, record_sha256=sha256_bytes(canonical(body).encode()))
    with open(path, "a") as stream:
        stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    records.append(record)
    return record


# ---------------------------------------------------------------- outputs


def write_outputs(directory, images, rows):
    import torch

    tmp = Path(str(directory) + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for (name, image), row in zip(images, rows):
        d = tmp / Path(name).stem
        d.mkdir(parents=True)
        with gzip.open(d / "output.pt.gz", "wb", compresslevel=1) as stream:
            torch.save(image, stream)
        (d / "decision.json").write_text(json.dumps(dict(row, output_file_sha256=sha256_file(d / "output.pt.gz")), indent=2))
    os.replace(tmp, directory)


def prune(candidates, records):
    complete = [r for r in records if r["status"] == "COMPLETE"]
    best = max((r["summary"]["mean_psnr"] for r in complete), default=None)
    keep = {r["setting_id"] for r in complete if r["summary"]["mean_psnr"] >= best - PSNR_TIE_DB} if complete else set()
    for d in Path(candidates).iterdir():
        if d.name not in keep:
            shutil.rmtree(d)
    return keep


def materialize(work, chosen, window, ctx, records, settings, execution_manifest_sha256, log_path):
    row_dir = Path(work) / TUNED_ROW
    source = Path(work) / "candidates" / chosen["setting_id"]
    shutil.copytree(source, row_dir)  # refuses if the row already exists
    rows = []
    for item, image_record in zip(ctx.files, chosen["per_image"]):
        d = row_dir / Path(item["name"]).stem
        decision = json.loads((d / "decision.json").read_bytes())
        require(decision["output_tensor_sha256"] == image_record["output_tensor_sha256"]
                and sha256_file(d / "output.pt.gz") == decision["output_file_sha256"], f"tuned output changed: {item['name']}")
        rows.append({"low_name": item["name"], "low_sha256": item["sha256"], **decision})
    manifest = {
        "method_id": TUNED_ROW,
        "method": "Ours-TTT (target-tuned): frozen T070-A code, knobs selected on target GT",
        "target_gt_used_for_selection": True,
        "count": len(rows),
        "low_receipt_sha256": ctx.low_sha256,
        "gate_receipt_sha256": ctx.receipt_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "setting_id": chosen["setting_id"],
        "knobs": chosen["knobs"],
        "defaults": DEFAULTS,
        "knob_locations": KNOB_LOCATIONS,
        "no_active_abstentions": chosen["summary"]["abstentions"],
        "selection": {
            "rule": "max image-mean PSNR; within PSNR_TIE_DB: max mean RGB-SSIM, then min normalised L1 knob distance, then setting id",
            "scope": "union of all rounds' COMPLETE records; L1 ranges over the union of declared settings",
            "psnr_tie_db": PSNR_TIE_DB,
            "settings_declared": len(settings),
            "settings_complete": sum(r["status"] == "COMPLETE" for r in records),
            "settings_failed": sum(r["status"] != "COMPLETE" for r in records),
            "tie_window": window,
            "selected_record_sha256": chosen["record_sha256"],
            "tuning_log_sha256": sha256_file(log_path),
        },
        "rows": rows,
    }
    with open(row_dir / "output_manifest.json", "x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def round_file(work, number):
    return Path(work) / f"round{number}_grid.json"


def store_round(work, number, settings, source_sha256, explanation=None):
    """Persist a round's declared settings. Round 1 may only grow (superset) before round 2 exists;
    round 2 must equal the rule applied to the log. Returns the SHA256 of the stored round file."""
    path, ids = round_file(work, number), [setting_id(s) for s in settings]
    if path.exists():
        old, _ = load_json(path)
        if number == 1:
            require(not round_file(work, 2).exists(), "round 1 cannot change after round 2 started")
            require(set(old["setting_ids"]) <= set(ids), "a new round-1 grid must be a superset of the stored one")
        else:
            require(old["setting_ids"] == ids, "round-2 grid differs from the rule applied to the log")
            return sha256_file(path)
    payload = {"round": number, "source_sha256": source_sha256, "setting_ids": ids, "settings": settings,
               "explanation": explanation}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(tmp, path)
    return sha256_file(path)


def check_default_reproduction(model, lows, frozen_rows, binding, device="cuda:0"):
    """N2: at every process start (incl. resume), the default setting must reproduce the frozen Ours-TTT
    output tensors bit for bit. No GT is decoded before this passes."""
    knobs = canonical_knobs({})
    variant = {"id": "default", "knobs": knobs, "derived": derive(knobs, binding)[0]}
    for low, row in zip(lows, frozen_rows):
        image, _ = run_group(model.scorer, model.gate, model.model, low, [variant], device)["default"]
        require(image is not None and sha256_bytes(image.contiguous().numpy().tobytes()) == row["output_tensor_sha256"],
                f"default setting does not reproduce the frozen Ours-TTT row at {row['low_name']}; stopping")


def static_context(ctx):
    return {"target": ctx.target, "gate_receipt_sha256": ctx.receipt_sha256, "low_receipt_sha256": ctx.low_sha256,
            "reference_opaque_manifest_sha256": ctx.opaque_sha256, "ours_tuning.py": sha256_file(__file__),
            "metrics.py": sha256_file(metrics_mod.__file__), "psnr_tie_db": PSNR_TIE_DB}


def materialize_global(work, ctx, records, log_path):
    """Once, after round 1 and round 2 are complete: global selection over the union of all rounds."""
    require(records, "empty tuning log")
    done = {r["setting_id"] for r in records}
    for r in records:
        require(r["context"] == records[0]["context"], "tuning log mixes contexts")
    base = static_context(ctx)
    require({k: records[0]["context"][k] for k in base} == base, "tuning log was written under other code/inputs")
    round1, _ = load_json(round_file(work, 1))
    require(set(round1["setting_ids"]) <= done, "round 1 incomplete")
    require(round_file(work, 2).exists(), "round 2 not run (run --round2 even when its grid is empty)")
    round2, _ = load_json(round_file(work, 2))
    rebuilt, _ = round2_grid(records, round1["settings"])
    require(round2["setting_ids"] == [setting_id(s) for s in rebuilt], "round-2 grid differs from the rule applied to the log")
    require(set(round2["setting_ids"]) <= done, "round 2 incomplete")
    union = union_settings(records)
    chosen, window = select(records, union)
    return materialize(work, chosen, window, ctx, records, union, records[0]["context"]["execution_manifest_sha256"], log_path)


def main():
    require(__debug__, "run without -O: the frozen code's assert statements are part of the method")
    sys.path.insert(0, os.getcwd())  # frozen Ours source root
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--low-dir", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path)
    p.add_argument("--stage-target", type=Path)
    p.add_argument("--manifest", type=Path, required=True, help="frozen T070-A execution manifest")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--grid", type=Path, help="round 1: declared grid (tuning_grid_round1.json)")
    mode.add_argument("--round2", action="store_true", help="round 2: grid built from the round-1 log")
    mode.add_argument("--materialize", action="store_true", help="global selection -> ours_ttt_target_tuned (once)")
    p.add_argument("--work", type=Path, required=True, help="one work directory for all rounds")
    a = p.parse_args()
    ctx = metrics_mod.Context(a.target, a.low_receipt, a.opaque_manifest, a.gate_receipt, a.rows_dir, a.stage_target)
    a.work.mkdir(parents=True, exist_ok=True)
    log_path = a.work / "tuning_log.jsonl"
    records = read_log(log_path)
    done = {r["setting_id"] for r in records}
    if a.materialize:
        manifest = materialize_global(a.work, ctx, records, log_path)
        print(json.dumps({k: manifest[k] for k in ("setting_id", "knobs", "no_active_abstentions", "selection")}, indent=2))
        print(f"TUNED_ROW_MANIFEST_SHA256 {sha256_file(a.work / TUNED_ROW / 'output_manifest.json')}", flush=True)
        return
    if a.grid is not None:
        round_no = 1
        grid_spec, grid_sha = load_json(a.grid)
        settings = expand_grid(grid_spec)
        store_round(a.work, 1, settings, grid_sha)
    else:
        round_no = 2
        round1, _ = load_json(round_file(a.work, 1))
        require(set(round1["setting_ids"]) <= done, "round 1 incomplete; finish it before --round2")
        settings, explanation = round2_grid(records, round1["settings"])
        grid_sha = store_round(a.work, 2, settings, sha256_file(round_file(a.work, 1)), explanation)
        print(json.dumps({"round2": explanation, "settings": len(settings)}), flush=True)

    from research_log.T070A.infer import FinalOurs
    from ttie.lolv2_gamma_core import native_rgb

    model = FinalOurs(a.manifest)
    require(model.gate["q_joint"] == DEFAULTS["q_joint"] and model.gate["calibration"]["tau"] == DEFAULTS["tau"],
            "DEFAULTS differ from the frozen gate asset")
    frozen_ttt, frozen_sha = load_json(ctx.rows_dir / "ours_ttt" / "output_manifest.json")
    require(frozen_sha == ctx.receipt["rows"]["ours_ttt"]["output_manifest_sha256"], "frozen ours_ttt manifest changed")
    require(frozen_ttt["execution_manifest_sha256"] == model.manifest_sha256, "frozen ours_ttt used another execution manifest")
    lows = []
    for item in ctx.files:
        path = a.low_dir / item["name"]
        require(sha256_file(path) == item["sha256"], f"low changed: {item['name']}")
        lows.append(native_rgb(path))
    binding = model.manifest["source_binding"]
    context = dict(static_context(ctx), execution_manifest_sha256=model.manifest_sha256,
                   derivation_sha256=derive(DEFAULTS, binding)[1])
    for r in records:
        require(r["context"] == context, "tuning log written under a different context; use a new --work directory")
    check_default_reproduction(model, lows, frozen_ttt["rows"], binding)

    candidates = a.work / "candidates"
    candidates.mkdir(exist_ok=True)
    for d in list(candidates.iterdir()):
        if d.name not in done:
            shutil.rmtree(d)  # partial outputs of a setting that never reached the log
    references = {}
    groups = {}
    for knobs in settings:
        sid = setting_id(knobs)
        if sid in done:
            continue
        key = json.dumps({n: knobs[n] for n in TRAJECTORY_KNOBS}, sort_keys=True)
        groups.setdefault(key, []).append({"id": sid, "knobs": knobs, "derived": derive(knobs, binding)[0]})
    for variants in groups.values():
        started = time.perf_counter()
        results = {v["id"]: [] for v in variants}
        for index, low in enumerate(lows):
            for sid, pair in run_group(model.scorer, model.gate, model.model, low, variants, "cuda:0").items():
                results[sid].append(pair)
            print(f"group of {len(variants)} settings: {index + 1}/{len(lows)}", flush=True)
        for v in variants:
            body = {"setting_id": v["id"], "knobs": v["knobs"], "is_default": v["knobs"] == canonical_knobs({}),
                    "round": round_no, "grid_sha256": grid_sha, "context": context, "utc": utc(),
                    "runtime_seconds": time.perf_counter() - started}
            hashes = [None if image is None else sha256_bytes(image.contiguous().numpy().tobytes())
                      for image, _ in results[v["id"]]]
            if body["is_default"]:
                for item, row, digest in zip(ctx.files, frozen_ttt["rows"], hashes):
                    require(digest == row["output_tensor_sha256"],
                            f"default setting does not reproduce the frozen Ours-TTT row at {item['name']}; stopping")
            rejected = [dict(rec, name=item["name"]) for item, (image, rec) in zip(ctx.files, results[v["id"]]) if image is None]
            if rejected:
                append_log(log_path, records, dict(body, status="FAILED", rejected=rejected, per_image=[], summary=None))
                continue
            per_image, images = [], []
            for index, (item, (image, rec), tensor_sha) in enumerate(zip(ctx.files, results[v["id"]], hashes)):
                if index not in references:
                    references[index] = ctx.reference(index)
                m = metrics_mod.score(image[0].permute(1, 2, 0).numpy(), references[index], ctx.core)
                row = {"name": item["name"], "cluster": item["cluster"], "output_tensor_sha256": tensor_sha,
                       "shape": list(image.shape), "dtype": str(image.dtype), "setting_id": v["id"], **rec, **m}
                per_image.append(row)
                images.append((item["name"], image))
            summary = dict(metrics_mod.row_summary(per_image), abstentions=sum(r["status"] != "TTT_EXECUTED" for r in per_image))
            complete = [r["summary"]["mean_psnr"] for r in records if r["status"] == "COMPLETE"]
            if summary["mean_psnr"] >= max(complete + [summary["mean_psnr"]]) - PSNR_TIE_DB:
                write_outputs(candidates / v["id"], images, per_image)
            append_log(log_path, records, dict(body, status="COMPLETE", per_image=per_image, summary=summary))
            prune(candidates, records)
            print(json.dumps({"setting_id": v["id"], **summary}), flush=True)
    records = read_log(log_path)
    require({r["setting_id"] for r in records} >= {setting_id(s) for s in settings}, f"round {round_no} incomplete")
    print(f"ROUND {round_no} COMPLETE: {len(settings)} declared, {len(records)} log records", flush=True)

if __name__ == "__main__":
    main()
