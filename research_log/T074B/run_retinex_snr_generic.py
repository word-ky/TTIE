"""T074-B generic RetinexFormer / SNR-Aware low-only outputs from the frozen T072-AZ exporters.

Imports the accepted exporters' ``load_model``/``forward`` unchanged from the
T072-AZ runtime, after verifying every bound file hash. Low decoding repeats the
exporters' own ``main`` lines. The SNR 512-query-row attention schedule is used
only with ``--query-row-schedule`` (4K geometry); the default is the official
unscheduled forward. Outputs: float32 ``[1,3,H,W]`` RGB (lossless permute of the
exporter's HWC array) in ``<stem>/output.pt.gz`` + ``decision.json`` + manifest.
No reference path is accepted.
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BINDINGS_SHA256 = "36cd7cdf3a2e8b885948b2ee420ffae459ef3c6a457477021653052cd01da012"
RETINEX_ARCH_SHA256 = "1567c89d55285a3c1a4d4ca33f288b1fd98dc6b6bbdf971eda02d745f9087631"
CHUNK_ATTENTION_SHA256 = "147bddc3f5957cb550a490b9ead1206ebd8a14fdac660e1edd40d69f79dd402d"
SNR_PARAMETER_SHA256 = "11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receipt(path, low_dir, expected_count):
    files = json.loads(Path(path).read_bytes())["files"]
    names = [item["name"] for item in files]
    assert len(files) == expected_count, f"receipt has {len(files)}, declared {expected_count}"
    assert names == sorted(names) and len(set(names)) == len(names)
    assert len({Path(n).stem for n in names}) == len(names)
    assert all(n.endswith(".png") for n in names), "canonical PNG cache required"
    assert sorted(p.name for p in Path(low_dir).iterdir()) == names, "low dir must hold exactly the receipt files"
    return files


def verify_binding(method, runtime, bindings_path):
    assert sha256_file(bindings_path) == BINDINGS_SHA256
    binding = json.loads(Path(bindings_path).read_bytes())[method]
    for rel, digest in binding["files"].items():
        path = Path(rel) if Path(rel).is_absolute() else runtime / rel
        assert sha256_file(path) == digest, f"bound file changed: {rel}"
    return binding


def read_low(method, path):
    import cv2

    if method == "retinexformer":  # retinex_exporter.main
        img = np.float32(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)) / 255.
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.  # snr_exporter.main
    return torch.from_numpy(np.ascontiguousarray(raw[:, :, [2, 1, 0]].transpose(2, 0, 1))).float()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", choices=["retinexformer", "snr_aware"], required=True)
    p.add_argument("--runtime-root", type=Path, required=True, help="T072-AZ runtime")
    p.add_argument("--bindings", type=Path, required=True, help="T072-AZ seal/baseline_bindings.json")
    p.add_argument("--chunk-attention", type=Path, help="T072-AZ continuation/chunk_attention.py")
    p.add_argument("--query-row-schedule", action="store_true", help="SNR 4K schedule; default official")
    p.add_argument("--low-dir", type=Path, required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-count", type=int, default=0)
    a = p.parse_args()
    started = time.perf_counter()
    runtime = a.runtime_root.resolve()
    files = load_receipt(a.low_receipt, a.low_dir, a.expected_count)
    binding = verify_binding(a.method, runtime, a.bindings)
    assert not (a.query_row_schedule and a.method != "snr_aware")
    sys.path.insert(0, str(runtime))
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if a.method == "retinexformer":
        from ttie.retinex_exporter import forward, load_model

        config = json.loads((runtime / binding["config"]).read_bytes())
        assert sha256_file(config["architecture"]) == RETINEX_ARCH_SHA256
        model = load_model(binding["checkpoint"], runtime / binding["config"])
        run = lambda x: forward(x, model)  # noqa: E731
        before = after = None
    else:
        from ttie.snr_exporter import forward, load_model, parameter_hash

        model = load_model(binding["checkpoint"], runtime / binding["config"], binding["source"])
        if a.query_row_schedule:
            assert sha256_file(a.chunk_attention) == CHUNK_ATTENTION_SHA256
            sys.path.insert(0, str(a.chunk_attention.parent))
            from chunk_attention import install

            install()
        before = parameter_hash(model)
        assert before == SNR_PARAMETER_SHA256
        run = lambda x: forward(x, model)  # noqa: E731
    cohort = files[:a.smoke_count] if a.smoke_count else files
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for index, item in enumerate(cohort):
            row_started = time.perf_counter()
            path = a.low_dir / item["name"]
            low_sha = sha256_file(path)
            assert low_sha == item["sha256"], item["name"]
            x = read_low(a.method, path)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            y = run(x)
            torch.cuda.synchronize()
            assert y.shape == (item["height"], item["width"], 3) and y.dtype == np.float32
            assert np.isfinite(y).all()
            tensor = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1).unsqueeze(0).contiguous()
            output_dir = a.out / Path(item["name"]).stem
            output_dir.mkdir()
            output_file = output_dir / "output.pt.gz"
            with gzip.open(output_file, "wb", compresslevel=1) as stream:
                torch.save(tensor, stream)
            row = {
                "low_name": item["name"],
                "low_sha256": low_sha,
                "shape": [1, 3, item["height"], item["width"]],
                "dtype": "torch.float32",
                "output_tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                "exporter_hwc_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "output_file_sha256": sha256_file(output_file),
                "whole_run_seconds": time.perf_counter() - row_started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
            }
            (output_dir / "decision.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(f"{index + 1}/{len(cohort)} {item['name']} {row['whole_run_seconds']:.3f}s", flush=True)
    if a.method == "snr_aware":
        after = parameter_hash(model)
        assert after == before
    manifest = {
        "method": f"{a.method}-lolv2real-generic",
        "promotable": not a.smoke_count,
        "count": len(rows),
        "bindings_sha256": BINDINGS_SHA256,
        "checkpoint_sha256": binding["files"][binding["checkpoint"]],
        "config_sha256": binding["files"][binding["config"]],
        "accepted_binding_sha256": binding["accepted_binding_sha256"],
        "low_receipt_sha256": sha256_file(a.low_receipt),
        "query_row_schedule": bool(a.query_row_schedule),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "whole_run_seconds": time.perf_counter() - started,
        "gpu_name": torch.cuda.get_device_name(),
        "reference_reads": 0,
        "metrics": 0,
        "rows": rows,
    }
    path = a.out / "output_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"OUTPUT_MANIFEST_SHA256 {sha256_file(path)}", flush=True)


if __name__ == "__main__":
    main()
