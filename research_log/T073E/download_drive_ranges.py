"""Download an official public Drive checkpoint in parallel byte ranges.

The local single-stream download was measured at ~0.1 MB/s. This changes only
asset transfer, not checkpoint content or any model computation.
"""

import argparse
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_range(url, start, stop, part):
    expected = stop - start + 1
    if part.exists() and part.stat().st_size == expected:
        return part
    response = requests.get(url, headers={"Range": f"bytes={start}-{stop}"}, stream=True, timeout=120)
    response.raise_for_status()
    assert response.status_code == 206
    assert response.headers["Content-Range"].startswith(f"bytes {start}-{stop}/")
    with part.open("wb") as stream:
        for block in response.iter_content(chunk_size=1024 * 1024):
            stream.write(block)
    assert part.stat().st_size == expected
    return part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mb", type=int, default=64)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    warning = requests.get(
        f"https://drive.google.com/uc?id={args.id}&export=download&confirm=t", timeout=30
    )
    warning.raise_for_status()
    uuid = re.search(r'name="uuid" value="([^"]+)"', warning.text).group(1)
    url = (
        f"https://drive.usercontent.google.com/download?id={args.id}"
        f"&export=download&confirm=t&uuid={uuid}"
    )
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
    probe.raise_for_status()
    assert probe.status_code == 206
    size = int(probe.headers["Content-Range"].split("/")[-1])
    probe.close()
    width = args.chunk_mb * 1024 * 1024
    jobs = [
        (index, start, min(size - 1, start + width - 1))
        for index, start in enumerate(range(0, size, width))
    ]
    print(f"FILE_BYTES {size} CHUNKS {len(jobs)} WORKERS {args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_range, url, start, stop, args.out.with_name(args.out.name + f".part-{index:03}")): index
            for index, start, stop in jobs
        }
        for future in as_completed(futures):
            future.result()
            print(f"CHUNK {futures[future] + 1}/{len(jobs)}", flush=True)
    digest = hashlib.sha256()
    assembling = args.out.with_name(args.out.name + ".assembling")
    with assembling.open("wb") as result:
        for index, _, _ in jobs:
            with args.out.with_name(args.out.name + f".part-{index:03}").open("rb") as part:
                while block := part.read(1024 * 1024):
                    result.write(block)
                    digest.update(block)
    assert assembling.stat().st_size == size
    os.replace(assembling, args.out)
    print(f"SHA256 {digest.hexdigest()} BYTES {size}", flush=True)


if __name__ == "__main__":
    main()
