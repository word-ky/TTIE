# T059-P embeddings
The unchanged vectors.pt (178,676,960 bytes) is gzip-compressed and split solely for GitHub blob delivery. No precision changes.
Reconstruct in this directory before running verify.py:
```python
import pathlib,json,gzip,hashlib
p=pathlib.Path(".");m=json.loads((p/"vectors_archive.json").read_text());chunks=[]
for v in m["parts"]:
 b=(p/v["name"]).read_bytes();assert hashlib.sha256(b).hexdigest()==v["sha256"];chunks.append(b)
b=b"".join(chunks);assert hashlib.sha256(b).hexdigest()==m["gzip_sha256"]
v=gzip.decompress(b);assert hashlib.sha256(v).hexdigest()==m["original_sha256"];(p/"vectors.pt").write_bytes(v)
```
Full raw evidence is also preserved under both remote TTIE project disks.
