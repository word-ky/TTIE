"""Export only gates from the already-frozen Stage-A tensors for independent mask replay."""
from pathlib import Path
import torch,json,hashlib,sys
folder=Path(sys.argv[1]);f=json.loads((folder/'freeze.json').read_bytes());rows=[]
for row in f['rows']:
    p=folder/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    saved=torch.load(p,map_location='cpu',weights_only=True)
    rows.append(dict(index=row['index'],stage_a_file_sha256=row['sha256'],gate=saved['gate']))
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+'\n')
