"""Fixed image partition and selected-row reader for accepted AF torch ZIP files."""
import collections,io,json,pickle,struct,zipfile
from pathlib import Path
import torch

def split_manifest(banks):
    ids=list(dict.fromkeys(b['image_id'] for b in banks));counts=collections.Counter(b['image_id'] for b in banks)
    assert len(banks)==400 and len(ids)==80 and set(counts.values())=={5}
    groups={v:i for i,v in enumerate(ids)};sides={'train':[], 'heldout':[]};rows={'train':[], 'heldout':[]};canonical=[]
    for bi,b in enumerate(banks):
        side='heldout' if groups[b['image_id']]%5==0 else 'train';sides[side].append(bi)
        for si in range(b['states']):
            i=len(canonical);canonical.append(dict(index=i,bank_index=bi,state_index=si,image_id=b['image_id']));rows[side].append(i)
    image_ids={side:[v for v in ids if ('heldout' if groups[v]%5==0 else 'train')==side] for side in sides}
    assert len(canonical)==7346 and len(sides['train'])==320 and len(sides['heldout'])==80
    assert len(image_ids['train'])==64 and len(image_ids['heldout'])==16 and not set(image_ids['train'])&set(image_ids['heldout'])
    assert sorted(rows['train']+rows['heldout'])==list(range(7346)) and not set(rows['train'])&set(rows['heldout'])
    return dict(ordered_image_ids=ids,image_group_indices=groups,entries_per_image=dict(counts),bank_indices=sides,image_ids=image_ids,row_indices=rows,canonical=canonical,zero_image_overlap=True,complete_partition=True)

def rebuild(storage,offset,size,stride,*unused):return dict(storage=storage,offset=offset,size=size,stride=stride)
class Metadata(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
        if (module,name)==('torch','FloatStorage'):return 'FloatStorage'
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        raise ValueError((module,name))
    def persistent_load(self,value):return value

def read_reference_rows(path,wanted):
    """Read only selected float32 rows; never torch.load a mixed-supervision chunk."""
    with zipfile.ZipFile(path) as z:
        prefix=next(n[:-len('data.pkl')] for n in z.namelist() if n.endswith('/data.pkl'))
        meta=Metadata(io.BytesIO(z.read(prefix+'data.pkl'))).load();d=meta['g_R'];indices=meta['indices']
        assert d['offset']==0 and d['size']==(len(indices),1,1,8,8) and d['stride']==(64,64,64,8,1)
        assert d['storage'][1]=='FloatStorage' and d['storage'][4]==len(indices)*64 and z.read(prefix+'byteorder')==b'little'
        member=z.getinfo(prefix+'data/'+d['storage'][2]);assert member.compress_type==zipfile.ZIP_STORED and member.file_size==len(indices)*256
        positions=[(j,i) for j,i in enumerate(indices) if i in wanted]
    result={};ranges=[]
    with open(path,'rb') as f:
        f.seek(member.header_offset);header=f.read(30);name_len,extra_len=struct.unpack_from('<HH',header,26);start=member.header_offset+30+name_len+extra_len
        for j,i in positions:
            offset=start+j*256;f.seek(offset);raw=f.read(256);assert len(raw)==256
            result[i]=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone().reshape(1,1,8,8)
            ranges.append(dict(index=i,offset=offset,bytes=256))
    return result,ranges

def numeric_gates(legacy,detail):
    values={'detail_positive':(detail['positive_fraction'],.75),'detail_cosine':(detail['median_cosine'],.50),'legacy_positive':(legacy['positive_fraction'],.95),'legacy_cosine':(legacy['median_cosine'],.90),'value_huber':(legacy['value_huber'],.07650849781930447)}
    gates={k:dict(actual=v,threshold=t,margin=t-v if k=='value_huber' else v-t,passed=v<=t if k=='value_huber' else v>=t) for k,(v,t) in values.items()}
    return dict(gates=gates,classification='image-held-out dual-tangent generalization supported' if all(g['passed'] for g in gates.values()) else 'image-held-out dual-tangent generalization not supported under fixed split')
