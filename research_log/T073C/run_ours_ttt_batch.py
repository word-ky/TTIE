"""Run the unchanged frozen Ours-TTT call per preregistered UHD-LL low image."""

import argparse
import json
from pathlib import Path

import torch

from research_log.T063A.common import sha
from research_log.T070A.infer import FinalOurs
from research_log.T073C.run_ours_ttt import run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-one", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.low_receipt.read_bytes())["files"]
    assert len(expected) == 150
    if args.smoke_one:
        expected = expected[:1]
    model = FinalOurs(args.manifest)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, item in enumerate(expected, 1):
        torch.cuda.reset_peak_memory_stats()
        record = run_one(
            model,
            args.low_dir / item["name"],
            args.out / Path(item["name"]).stem,
            expected_low_sha256=item["sha256"],
        )
        rows.append(record)
        print(f"{index}/{len(expected)} {item['name']} {record['whole_run_seconds']:.3f}s", flush=True)
    summary = {
        "method": "Ours-TTT",
        "count": len(rows),
        "execution_manifest_sha256": sha(args.manifest),
        "low_receipt_sha256": sha(args.low_receipt),
        "rows": rows,
    }
    (args.out / "output_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"OUTPUT_MANIFEST_SHA256 {sha(args.out / 'output_manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
