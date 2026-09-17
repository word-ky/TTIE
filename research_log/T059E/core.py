"""Nested image split and row-selective access to immutable mixed caches."""
import io,struct,zipfile,hashlib,math
import torch
from research_log.T059C2.core import Metadata,split_manifest

def nested_split(banks):
    old=split_manifest(banks);groups={v:i for i,v in enumerate(old['ordered_image_ids'])};sides={'train':[],'heldout':[],'outer':[]};rows={k:[] for k in sides};images={k:[] for k in sides}
    for image in old['ordered_image_ids']:
        rem=groups[image]%5;side='outer' if rem==0 else 'heldout' if rem==1 else 'train';images[side].append(image)
    for bi,b in enumerate(banks):
        rem=groups[b['image_id']]%5;side='outer' if rem==0 else 'heldout' if rem==1 else 'train';sides[side].append(bi)
    for row in old['canonical']:
        rem=groups[row['image_id']]%5;side='outer' if rem==0 else 'heldout' if rem==1 else 'train';rows[side].append(row['index'])
    assert [len(images[k]) for k in ['train','heldout','outer']]==[48,16,16]
    assert [len(sides[k]) for k in ['train','heldout','outer']]==[240,80,80]
    assert rows['outer']==old['row_indices']['heldout'] and sorted(rows['train']+rows['heldout'])==old['row_indices']['train']
    assert all(not set(images[a])&set(images[b]) for a,b in [('train','heldout'),('train','outer'),('heldout','outer')])
    return dict(ordered_image_ids=old['ordered_image_ids'],group_indices=groups,canonical=old['canonical'],image_ids=images,bank_indices=sides,row_indices=rows,zero_overlap=True,nonouter_partition_exact=True)

def selected_rows(path,wanted,key,shape):
    with zipfile.ZipFile(path) as z:
        prefix=next(n[:-8] for n in z.namelist() if n.endswith('/data.pkl'));raw_meta=z.read(prefix+'data.pkl');meta=Metadata(io.BytesIO(raw_meta)).load();d=meta[key];indices=meta['indices'];width=math.prod(shape)
        assert d['offset']==0 and d['size']==(len(indices),*shape) and d['stride'][0]==width and d['storage'][1]=='FloatStorage'
        assert z.read(prefix+'byteorder')==b'little';member=z.getinfo(prefix+'data/'+d['storage'][2]);assert member.compress_type==zipfile.ZIP_STORED and member.file_size==len(indices)*width*4
        positions=[(j,i) for j,i in enumerate(indices) if i in wanted]
    result={};ranges=[]
    with open(path,'rb') as f:
        f.seek(member.header_offset);header=f.read(30);n,e=struct.unpack_from('<HH',header,26);start=member.header_offset+30+n+e
        for j,i in positions:
            offset=start+j*width*4;f.seek(offset);raw=f.read(width*4);assert len(raw)==width*4
            result[i]=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone().reshape(shape)
            ranges.append(dict(index=i,offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return result,dict(file=str(path),key=key,metadata_sha256=hashlib.sha256(raw_meta).hexdigest(),ranges=ranges)

def gate_values(stats):
    pairs={'relative_value':(stats['bank_relative_huber'],.07650849781930447),'detail_positive':(stats['detail']['positive_fraction'],.75),'detail_cosine':(stats['detail']['median_cosine'],.5),'legacy_positive':(stats['legacy']['positive_fraction'],.95),'legacy_cosine':(stats['legacy']['median_cosine'],.9)}
    gates={k:dict(actual=v,threshold=t,margin=t-v if k=='relative_value' else v-t,passed=v<=t if k=='relative_value' else v>=t) for k,(v,t) in pairs.items()}
    return dict(gates=gates,classification='bank-relative recipe shows nested source-development transfer' if all(v['passed'] for v in gates.values()) else 'bank-relative recipe not supported on fixed nested split')
