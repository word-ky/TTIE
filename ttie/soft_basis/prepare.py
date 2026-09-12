"""Extract only persisted label-free tensors/gates for a fixed list of old episode directories."""
import argparse
from pathlib import Path
import torch
from ..routing.provenance import verify_source
from ..stop_receipt import sha
from .score import SOURCES, read, write, tensor_sha

METHOD = 'region2_ttt_energy_sobolev'


def main():
    p = argparse.ArgumentParser()
    for name in ('audit', 'episode-list', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-sha', required=True); a = p.parse_args()
    code = verify_source(a.source_sha, SOURCES)
    names = a.episode_list.read_text(encoding='utf-8-sig').splitlines()
    assert len(names) == len(set(names)) == 120
    a.output.mkdir(parents=True, exist_ok=True); entries = []
    for name in names:
        directory = a.audit/name
        receipt = read(directory/'label_free_receipt.json')
        paths = ('outputs.pt', METHOD+'/decisions.json')
        expected = (receipt['episode']['outputs.pt']['sha256'], receipt[METHOD]['decisions.json']['sha256'])
        assert all(sha(directory/path) == digest for path, digest in zip(paths, expected))
        outputs = torch.load(directory/'outputs.pt', mmap=True, weights_only=True, map_location='cpu')
        gate = read(directory/METHOD/'decisions.json')['gate']
        image = outputs['identity']['image'].clone(); corners = outputs[METHOD]['grid'].clone()
        file = Path(name).name+'.pt'
        torch.save(dict(image=image, corners=corners, gate=gate), a.output/file)
        entries.append(dict(episode=name, file=file, sha256=sha(a.output/file),
                            pixels_sha256=tensor_sha(image), corners_sha256=tensor_sha(corners),
                            source_files=dict(zip(paths, expected))))
    write(a.output/'inputs.json', dict(source_sha=a.source_sha, source_code_sha256=code,
          original_audit=str(a.audit), episode_list_sha256=sha(a.episode_list), episodes=entries))
    print('Prepared', len(entries), 'persisted label-free episodes', flush=True)


if __name__ == '__main__': main()
