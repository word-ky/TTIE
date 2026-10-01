import torch,io,pickle,collections,zipfile,struct,math,hashlib
def rebuild(storage,offset,size,stride,*unused):return dict(storage=storage,offset=offset,size=size,stride=stride)
class Metadata(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
  if module=='torch' and name in ['FloatStorage','DoubleStorage','LongStorage']:return name
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  raise ValueError((module,name))
 def persistent_load(self,v):return v
def tensor_only(path,key):
 with zipfile.ZipFile(path) as z:
  prefix=next(n[:-8] for n in z.namelist() if n.endswith('/data.pkl'));raw=z.read(prefix+'data.pkl');d=Metadata(io.BytesIO(raw)).load()[key];member=z.getinfo(prefix+'data/'+d['storage'][2]);assert member.compress_type==zipfile.ZIP_STORED and z.read(prefix+'byteorder')==b'little'
 dtype={'FloatStorage':torch.float32,'DoubleStorage':torch.float64,'LongStorage':torch.int64}[d['storage'][1]];width=torch.empty((),dtype=dtype).element_size();n=math.prod(d['size']);assert d['stride']==(1,) and len(d['size'])==1
 with open(path,'rb') as f:
  f.seek(member.header_offset);header=f.read(30);nl,el=struct.unpack_from('<HH',header,26);offset=member.header_offset+30+nl+el+d['offset']*width;f.seek(offset);rawdata=f.read(n*width);assert len(rawdata)==n*width
 return torch.frombuffer(bytearray(rawdata),dtype=dtype).clone(),dict(key=key,file=str(path),offset=offset,bytes=len(rawdata),sha256=hashlib.sha256(rawdata).hexdigest(),metadata_sha256=hashlib.sha256(raw).hexdigest())
