import argparse,time
from PIL import Image
from scripts.run_t036a import models,initialize
from ttie.lolv2_gamma_core import native_rgb
from ttie.energy_model import load_energy
from research_log.T060CR1.core import *

def main(out):
    initialize();start=utc();out.mkdir(parents=True,exist_ok=False)
    cohort=json.loads((HERE/'cohort.json').read_bytes());binding=json.loads((HERE/'source_binding.json').read_bytes());inputs=cohort['inputs']
    firewall(set(inputs)|{str(Path(n).resolve()) for n in binding}|{str((HERE/'source_binding.json').resolve())},out)
    validate(binding);validate(inputs);assert sha(HERE/'cohort.json')==json.loads((HERE/'cohort_freeze.json').read_bytes())['sha256']
    allowed={r['path'] for r in cohort['rows']};opened=[];original=Image.open
    def open_low(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
    Image.open=open_low
    scorer,receipt,head014=models(cohort['assets']);headE=load_energy(HEADS['E'])
    before={h:{k:thash(v) for k,v in head.state_dict().items()} for h,head in [('014',head014),('E',headE)]};records=[]
    for r in cohort['rows']:
        low=native_rgb(Path(r['path'])).cuda();begin=time.perf_counter();result,t,decision,traces=trajectory(low,scorer,receipt,head014,headE)
        d=out/f"{r['index']:03d}";d.mkdir();save_tensor(d/'output.pt',{k:result[k].detach().cpu().clone() for k in ['image','raw','grid']})
        save_tensor(d/'trajectory.pt',{k:t[k] for k in ['states','grids','scores','features']});save_tensor(d/'gradients.pt',traces)
        atomic_json(d/'decision.json',dict(selection=decision,gate=t['gate'],diagnostics=t['diagnostics']))
        records.append(dict(**r,selected_step=decision['selected_step'],updates=t['diagnostics']['steps'],seconds=time.perf_counter()-begin,files={p.name:sha(p) for p in d.iterdir()},persisted_utc=utc()))
        atomic_json(out/'progress.json',records);print(json.dumps(dict(completed=len(records),selected_step=decision['selected_step'],seconds=records[-1]['seconds'])),flush=True)
        del result,t,traces
    assert opened==[r['path'] for r in cohort['rows']];after={h:{k:thash(v) for k,v in head.state_dict().items()} for h,head in [('014',head014),('E',headE)]};assert before==after
    validate(binding);validate(inputs)
    atomic_json(out/'freeze.json',dict(started_utc=start,completed_utc=utc(),rows=records,opened_images=opened,source_bindings=binding,inputs=inputs,cohort_sha256=sha(HERE/'cohort.json'),head_state_before=before,head_state_after=after,normalizations={'014':head014.normalization(),'E':headE.normalization()},normal_reference_reads=0,reference_gradient_reads=0,baseline_outcome_reads=0,official_test_access=0,inference_reference_leakage=0,optimizer_steps=sum(r['updates'] for r in records),physical_gpu=1,gpu=torch.cuda.get_device_name()))
    print('ALL100_OUTPUTS_FROZEN',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
