import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t035',Path(__file__).resolve().parents[1]/'T035A_oracle/core.py')
t035=importlib.util.module_from_spec(spec);spec.loader.exec_module(t035)
o=t035.prior
torch=t035.torch
json=o.json
sha=o.sha
write=o.write
utc=o.utc
native=o.native
LABEL=o.LABEL
SPLIT_SHA=o.SPLIT_SHA
frozen_model=t035.frozen_model
optimize_start=t035.optimize_start
physical=t035.common_physical
PRIOR_FREEZE='c74e911b6784dd9d201cbe10c68124e36502ab41bcdd210d21fdf881e173c89e'
def thash(x):return o.hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def verdict(mean,median):
    return 'T035 common-gain oracle materially underconverged' if mean>=1. and median>=.75 else 'T035 common-gain oracle material underconvergence not supported under fixed extension'

PRIOR_CSV='d531c45ce670f5b84c2b74059e931350e21e49c7d1c70d15229f4fbb26a24d72'
