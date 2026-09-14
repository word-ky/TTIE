"""Fixed coordinate groups, accepted T029 scalar convention and T038 verdict."""
from research_log.T029A_alignment.common import alignment,summarize,sha,thash,utc,write,LABEL
from pathlib import Path
import json
import numpy as np
import torch
COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
PRIOR_FREEZE='46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4'
PRIOR_METRICS='cdbd7fec76194153db645133d5d35dd43f6f9d852ebbd6674756d77d1ad7aee0'
STEPS=[0,10,20,30,40]
def masks(active):
    total=active.reshape(1,1,2,2).expand(1,3,2,2).clone()
    legacy=total.clone();legacy[:,2]=False
    gain=total.clone();gain[:,:2]=False
    return dict(legacy=legacy,gain=gain,total=total)
def aggregate(rows):
    return {group:dict(**summarize([r['groups'][group] for r in rows]),
        **{name+'_median':float(np.median([r['groups'][group][name] for r in rows])) for name in ['energy_norm','reference_norm','dot']}) for group in ['legacy','gain','total']}
def classify(groups):
    gain=groups['gain'];legacy=groups['legacy']
    good=(gain['cosine_median'] is not None and gain['positive_dot_fraction'] is not None and legacy['positive_dot_fraction'] is not None
          and gain['cosine_median']<=-.25 and gain['positive_dot_fraction']<=.35
          and legacy['positive_dot_fraction']-gain['positive_dot_fraction']>=.20)
    return 'gain-specific mismatch supported' if good else 'gain-specific mismatch not supported / shared-or-mixed field failure'
def patterns(rows):
    from collections import Counter
    return dict(Counter(('legacy_valid' if r['groups']['legacy']['positive_dot'] else 'legacy_invalid')+' / '+('gain_valid' if r['groups']['gain']['positive_dot'] else 'gain_invalid') for r in rows))
