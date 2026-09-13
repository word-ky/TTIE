"""Bind canonical filenames before reading image payloads; deploy low images only."""
import argparse
import hashlib
import json
import struct
import tarfile
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PREFIX='LOL-v2/Real_captured/'


def selected_paths(paths):
    return sorted(paths,key=lambda p: hashlib.sha256(('TTIE-T022A-seed7|'+p).encode()).hexdigest())[:100]


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--low-tar',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        names={n[len(PREFIX):] for n in z.namelist() if n.startswith(PREFIX) and n.endswith('.png')}
        train=sorted(n for n in names if n.startswith('Train/Low/'))
        test=sorted(n for n in names if n.startswith('Test/Low/'))
        assert len(train)==689 and len(test)==100
        pairs=[dict(low=n,normal=n.replace('/Low/low','/Normal/normal')) for n in train+test]
        assert all(r['normal'] in names for r in pairs) and len(names)==1578
        chosen=selected_paths(train)
        filename_binding=dict(relative_root=PREFIX,rule='SHA256("TTIE-T022A-seed7|" + relative_path), ascending, first100',
                              selected=chosen,remaining_train=589,official_test=100,
                              frozen_before_payload_read_utc=datetime.now(timezone.utc).isoformat())
        binding_bytes=(json.dumps(filename_binding,indent=2)+'\n').encode()
        (args.out/'filename_binding.json').write_bytes(binding_bytes)
        # Only encoded PNG bytes and IHDR dimensions are inspected here;
        # no normal-light pixels are decoded and no image-quality metric is run.
        records=[]
        with tarfile.open(args.low_tar,'w:gz') as tar:
            for row in pairs:
                record=dict(row)
                for kind in ['low','normal']:
                    data=z.read(PREFIX+row[kind]);assert data[:8]==b'\x89PNG\r\n\x1a\n'
                    record[kind+'_sha256']=hashlib.sha256(data).hexdigest()
                    record[kind+'_size']=list(struct.unpack('>II',data[16:24]))
                    if kind=='low' and row['low'] in chosen:
                        info=tarfile.TarInfo(row['low']);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                assert record['low_size']==record['normal_size']
                records.append(record)
        by_low={r['low']:r for r in records}
        split=dict(filename_binding_sha256=hashlib.sha256(binding_bytes).hexdigest(),relative_root=PREFIX,
                   selected=[by_low[n] for n in chosen])
        (args.out/'split.json').write_text(json.dumps(split,indent=2)+'\n')
        manifest=dict(dataset='Canonical LOL-v2 Real',relative_root=PREFIX,train_pairs=689,test_pairs=100,
                      dimensions_from='PNG IHDR only, no pixel decode',records=records,
                      normal_pixels_read=False,official_test_evaluated=False,
                      split_sha256=hashlib.sha256((args.out/'split.json').read_bytes()).hexdigest())
        (args.out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:v for k,v in manifest.items() if k!='records'}))


if __name__=='__main__':main()
