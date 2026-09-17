import pathlib,json,hashlib,os,datetime,argparse,torch
from research_log.T059I.core import tensor_only
from research_log.T059J.core import floor,classify,LIMIT
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');G=ROOT/'runs/20260918-050302-ttie-t059g-nn/artifacts/T059G';I=ROOT/'runs/20260918-073334-ttie-t059i-five/artifacts/T059I';ISOURCE=ROOT/'releases/20260918-072929-ttie-t059i-five'
EXPECTED={str(G/'maps.pt'):'a40c269baed073499defcfb651600cc7ff852cc47134cc2cd4e5c2283bb807c4',str(G/'result.json'):'46177cfda13e3c00323b50c837981996aa313237de328aa4f5c632b3c95814d4',str(I/'maps.pt'):'289bee4e567a8daaa8180b2a7b58acbc4044ad26dbbe215aa92a818bf727ac81',str(I/'predictions.pt'):'1a09e1b591d890223e0041f0d60cbddd112a716f25c888ac5c3862269dd534f6',str(I/'result.json'):'20108a83f00334b9d66d668425bb9171e53cf29da9c7090a6134b15b778981cf'}
SCALAR_SHA='671a9b3afc74f2ffecab4c0359dbec902942a43d166f31af1cfd3d7cf763cf1b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);bindings=json.loads(pathlib.Path('research_log/T059J/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in bindings.items());assert {n:sha(pathlib.Path(n)) for n in EXPECTED}==EXPECTED
 old=json.loads((I/'result.json').read_text());assert all(sha(ISOURCE/n)==h for n,h in old['source_binding'].items());maps=torch.load(I/'maps.pt',map_location='cpu',weights_only=True);gm=torch.load(G/'maps.pt',map_location='cpu',weights_only=True);g=json.loads((G/'result.json').read_text())
 for s,n,images in [('train',4357,48),('heldout',1529,16)]:
  assert maps[s]['index'].shape==(n,5) and gm[s]['global_index'].tolist()==g['split']['row_indices'][s] and len(gm[s]['image'].unique())==images
  assert torch.equal(maps[s]['global_index'],gm['train']['global_index'][maps[s]['index']]);assert torch.equal(maps[s]['image'],gm['train']['image'][maps[s]['index']]);assert all(len(x.unique())==5 for x in maps[s]['image']);assert not (maps[s]['image']==gm[s]['image'][:,None]).any();assert torch.equal(maps[s]['global_index'][:,0],gm[s]['neighbor']['global_index'])
 assert old['original_G_Huber_exact']=={'train':.06445551663637161,'heldout':.23172274231910706} and old['consensus_huber']=={'train':.08473703265190125,'heldout':.2230137139558792}
 return old,maps,gm,bindings

def freeze(out):
 old,maps,gm,b=setup();out.mkdir(parents=True,exist_ok=False);opened=utc();train,receipt=tensor_only(G/'evaluation_rows.pt','train','target');tables={s:train[maps[s]['index']] for s in maps};prediction=torch.load(I/'predictions.pt',map_location='cpu',weights_only=True);assert all(torch.equal(tables[s].mean(1),prediction[s]) for s in maps)
 save(out/'donor_values.pt',tables);write(out/'donor_values_persisted.json',dict(label='REFERENCE_ORACLE_ONLY',training_scalar_opened_utc=opened,training_scalar_receipt=receipt,persisted_utc=utc(),sha256=sha(out/'donor_values.pt'),accepted_inputs=EXPECTED,source_binding=b,accepted_I_source_bindings=old['source_binding'],held_scalar_reads=0,full_mixed_scalar_SHA_deferred=True));print('DONOR_VALUES_FROZEN',utc())

def evaluate(out):
 old,maps,gm,b=setup();marker=json.loads((out/'donor_values_persisted.json').read_text());assert sha(out/'donor_values.pt')==marker['sha256'];opened=utc();assert marker['persisted_utc']<opened;before=dict(EXPECTED);before[str(G/'evaluation_rows.pt')]=sha(G/'evaluation_rows.pt');assert before[str(G/'evaluation_rows.pt')]==SCALAR_SHA
 tables=torch.load(out/'donor_values.pt',map_location='cpu',weights_only=True);pred=torch.load(I/'predictions.pt',map_location='cpu',weights_only=True);targets={};baseline={};consensus={};stats={};rows={};access=[]
 for s in ['train','heldout']:
  targets[s],receipt=tensor_only(G/'evaluation_rows.pt',s,'target');access.append(receipt);oldpred,receipt=tensor_only(G/'evaluation_rows.pt',s,'pred');access.append(receipt)
  assert torch.equal(tables[s],targets['train'][maps[s]['index']]);assert torch.equal(oldpred,tables[s][:,0]);assert torch.equal(pred[s],tables[s].mean(1))
  baseline[s]=float(torch.nn.functional.huber_loss(oldpred,targets[s]));consensus[s]=float(torch.nn.functional.huber_loss(pred[s],targets[s]));assert baseline[s]==old['original_G_Huber_exact'][s] and consensus[s]==old['consensus_huber'][s]
  stats[s],rows[s]=floor(tables[s],targets[s])
 assert access[0]['sha256']==marker['training_scalar_receipt']['sha256'];after={n:sha(pathlib.Path(n)) for n in before};assert before==after;assert all(sha(pathlib.Path(n))==h for n,h in b.items());assert all(sha(ISOURCE/n)==h for n,h in old['source_binding'].items())
 save(out/'oracle_rows.pt',rows);r=dict(status='DONE',label='REFERENCE_ORACLE_ONLY',classification=classify(stats['train']['oracle_floor_huber'],stats['heldout']['oracle_floor_huber']),statistics=stats,threshold=LIMIT,margins={s:LIMIT-stats[s]['oracle_floor_huber'] for s in stats},baseline_G_exact=baseline,consensus_I_exact=consensus,inputs_before=before,inputs_after=after,donor_marker=marker,held_scalar_opened_utc=opened,scalar_access=access,source_binding=b,oracle_rows_sha256=sha(out/'oracle_rows.pt'),counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_neighbor_searches','new_source_image_opens','new_feature_forwards','reference_gradient_recomputations','detail_gradient_tensor_reads','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc())
 write(out/'result.json',r);print(json.dumps({k:r[k] for k in ['classification','statistics','margins']}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);a=a.parse_args();(freeze if a.stage=='freeze' else evaluate)(a.out)
