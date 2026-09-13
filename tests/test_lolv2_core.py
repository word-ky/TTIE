import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.prepare_t022a import selected_paths
from ttie.lolv2_core import native_rgb


def test_split_filename_only_order_invariant():
    paths=[f'Train/Low/low{i:05d}.png' for i in range(1,690)]
    chosen=selected_paths(paths)
    assert chosen==selected_paths(paths[::-1])
    assert len(chosen)==len(set(chosen))==100
    import hashlib
    expected=sorted(paths,key=lambda p:hashlib.sha256(b'TTIE-T022A-seed7|'+p.encode()).digest())[:100]
    assert chosen==expected


def test_native_loader_preserves_pixels_and_size():
    data=np.random.default_rng(7).integers(0,256,(401,603,3),dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'low.png';Image.fromarray(data).save(path)
        result=native_rgb(path)
    assert result.shape==(1,3,401,603)
    assert torch.equal(result,torch.from_numpy(data).permute(2,0,1)[None].float()/255.)
