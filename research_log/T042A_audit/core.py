"""One fixed early legacy state and one fixed gain; accepted scalar conventions."""
from research_log.T041A_audit.core import alignment,aggregate,masks,sha,thash,utc,write,COHORT,PRIOR_FREEZE,probe_raw
from pathlib import Path
import json,torch
LABEL='REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
GAINS=[1.75]
LEGACY_STEP=10
def classify(total):
    yes=total['positive_dot_fraction'] is not None and total['cosine_median'] is not None and total['positive_dot_fraction']>=.57 and total['cosine_median']>=.06415200731653636
    return 'late real legacy-state effect supported' if yes else 'late real legacy-state effect not supported / mixed'
