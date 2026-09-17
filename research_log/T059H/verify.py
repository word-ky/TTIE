import pathlib,sys,json,hashlib,numpy as np,torch,math
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'result.json').read_text());root=pathlib.Path(r['input_root']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in r['inputs_before'].items():assert sha(root/n)==h==r['inputs_after'][n]
m=torch.load(root/'maps.pt',map_location='cpu',weights_only=True);v=torch.load(root/'evaluation_rows.pt',map_location='cpu',weights_only=True)
d={s:m[s]['neighbor']['distance'].numpy() for s in ['train','heldout']};sorted_d=np.sort(d['train']);n=len(sorted_d);b=[]
for i in range(1,10):
 pos=(n-1)*(i/10);lo=int(np.floor(pos));hi=int(np.ceil(pos));b.append(float(sorted_d[lo]+(sorted_d[hi]-sorted_d[lo])*(pos-lo)))
assert np.allclose(b,r['bins_frozen']['boundaries'],atol=1e-12,rtol=1e-12)
b=np.array(r['bins_frozen']['boundaries']);indices={s:np.searchsorted(b,d[s],side='right') for s in d};loss={}
for s in d:
 idx=m[s]['neighbor']['index'];assert torch.equal(v[s]['pred'],v['train']['target'][idx]);err=np.abs(v[s]['pred'].numpy()-v[s]['target'].numpy());loss[s]=np.where(err<1,np.float32(.5)*err*err,err-np.float32(.5));assert float(torch.from_numpy(loss[s]).mean())==r['original_huber'][s]
weighted=0.
for i,record in enumerate(r['bins']):
 a=indices['train']==i;c=indices['heldout']==i;assert a.sum()==record['train_rows'] and c.sum()==record['held_rows'];mean=float(loss['train'][a].astype('float64').mean());weight=float(c.sum())/len(c)
 assert math.isclose(mean,record['train_mean_huber'],abs_tol=1e-12) and weight==record['held_fraction'];weighted+=weight*mean
 if c.any():assert math.isclose(float(loss['heldout'][c].astype('float64').mean()),record['held_mean_huber'],abs_tol=1e-12)
 else:assert record['held_mean_huber'] is None
assert math.isclose(weighted,r['distance_reweighted_train_huber'],abs_tol=1e-12);assert sum(x['train_rows'] for x in r['bins'])==4357 and sum(x['held_rows'] for x in r['bins'])==1529
classification='T059-G scalar failure is consistent with nearest-distance support shift under fixed train-decile reweighting' if weighted>.07650849781930447 else 'nearest-distance shift alone does not explain T059-G scalar failure; conditional scalar mismatch remains';assert classification==r['classification'];assert all(x==0 for x in r['counters'].values());assert r['bins_frozen']['persisted_utc']<=r['scalar_pairs_opened_utc']
z=dict(verification='PASS',independent_distance_reweighted_train_huber=weighted,independent_classification=classification,rows_accounted_for=True,quantiles_independently_replayed=True,result_sha256=sha(out/'result.json'));(out/'verification.json').write_text(json.dumps(z,indent=2));print(json.dumps(z))
