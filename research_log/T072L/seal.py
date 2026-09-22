"""Seal metadata and deterministic random indices; never open target payloads."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def bootstrap_indices():
    return np.random.Generator(np.random.PCG64(20260922)).integers(
        0, 150, size=(10000, 150), dtype=np.int64)

def index_digest(indices):
    return hashlib.sha256(indices.astype('<i8').tobytes(order='C')).hexdigest()

if __name__ == '__main__':
    receipt = {
        'analysis_spec_sha256': hashlib.sha256((HERE/'analysis_spec.json').read_bytes()).hexdigest(),
        'bootstrap_indices_sha256': index_digest(bootstrap_indices()),
        'index_encoding': 'little-endian int64 C order',
        'numpy_version': np.__version__,
        'inference_runs': 0, 'reference_reads': 0, 'real_metrics': 0,
    }
    (HERE/'seal.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))
