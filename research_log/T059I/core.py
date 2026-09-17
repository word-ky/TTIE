import torch,io,pickle,collections,zipfile,struct,math,hashlib
LIMIT=.07650849781930447
LABELS=['fixed five-image consensus does not preserve inner-train scalar consistency; estimator smoothing is not a supported rescue','T059-G scalar failure is consistent with single-donor estimator instability; 28-D scalar support transfers under fixed five-image consensus','conditional scalar mismatch persists under fixed five-image consensus']
def classify(train,held):return LABELS[0] if train>LIMIT else LABELS[1] if held<=LIMIT else LABELS[2]
def rebuild(storage,offset,size,stride,*unused):return dict(storage=storage,offset=offset,size=size,stride=stride)
class Metadata(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
  if module=='torch' and name in ['FloatStorage','DoubleStorage','LongStorage']:return name
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  raise ValueError((module,name))
 def persistent_load(self,v):return v
def tensor_only(path,side,key):
 with zipfile.ZipFile(path) as z:
  prefix=next(n[:-8] for n in z.namelist() if n.endswith('/data.pkl'));raw=z.read(prefix+'data.pkl');d=Metadata(io.BytesIO(raw)).load()[side][key];member=z.getinfo(prefix+'data/'+d['storage'][2]);assert member.compress_type==zipfile.ZIP_STORED and z.read(prefix+'byteorder')==b'little'
 dtype={'FloatStorage':torch.float32,'DoubleStorage':torch.float64,'LongStorage':torch.int64}[d['storage'][1]];width=torch.empty((),dtype=dtype).element_size();n=math.prod(d['size']);assert d['stride']==(1,) and len(d['size'])==1
 with open(path,'rb') as f:
  f.seek(member.header_offset);header=f.read(30);nl,el=struct.unpack_from('<HH',header,26);offset=member.header_offset+30+nl+el+d['offset']*width;f.seek(offset);rawdata=f.read(n*width);assert len(rawdata)==n*width
 return torch.frombuffer(bytearray(rawdata),dtype=dtype).clone(),dict(side=side,key=key,file=str(path),offset=offset,bytes=len(rawdata),sha256=hashlib.sha256(rawdata).hexdigest(),metadata_sha256=hashlib.sha256(raw).hexdigest())
def five(query,train,qi,ti,global_ids,loo=False,device='cuda:0'):
 assert torch.equal(global_ids,global_ids.sort().values);images=ti.unique(sorted=True);tr=train.to(device,dtype=torch.float64);groups=[(ti==i).nonzero().flatten().to(device) for i in images];chosen=[];dist=[];donors=[];ties=[];cross=[]
 for start in range(0,len(query),64):
  d=(query[start:start+64].to(device,dtype=torch.float64)[:,None,:]-tr[None,:,:]).square().sum(-1);ds=[];ix=[];ts=[]
  for ids in groups:
   v=d[:,ids];mn,j=v.min(1);ds.append(mn.cpu());ix.append(ids[j].cpu());ts.append((v==mn[:,None]).sum(1).cpu())
  ds=torch.stack(ds,1);ix=torch.stack(ix,1);ts=torch.stack(ts,1)
  for j in range(len(ds)):
   eligible=[k for k in range(len(images)) if not loo or images[k]!=qi[start+j]]
   order=sorted(eligible,key=lambda k:(float(ds[j,k]),int(global_ids[ix[j,k]]),int(images[k])))[:5];assert len(order)==5
   chosen.append(ix[j,order]);dist.append(ds[j,order]);donors.append(images[order]);ties.append(ts[j,order]);cross.append(sum(float(ds[j,k])==float(ds[j,order[4]]) for k in eligible))
 return dict(index=torch.stack(chosen),global_index=global_ids[torch.stack(chosen)],distance=torch.stack(dist),image=torch.stack(donors),within_image_tie_count=torch.stack(ties),fifth_distance_tie_count=torch.tensor(cross))
