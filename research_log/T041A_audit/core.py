"""T041 fixed-gain real selected-state audit; accepted donor conventions."""
from research_log.T038A_attribution.core import alignment,aggregate,masks,sha,thash,utc,write,COHORT,PRIOR_FREEZE
from research_log.T039A_tangent.core import probe_raw
from pathlib import Path
import json,torch
LABEL='REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
GAINS=[1.25,1.75]
SOURCE_POSITIVE=0.796633554084
SOURCE_COSINE=0.763459378857
def classify(total):
    yes=(total['positive_dot_fraction'] is not None and total['cosine_median'] is not None and SOURCE_POSITIVE-total['positive_dot_fraction']>=.2 and SOURCE_COSINE-total['cosine_median']>=.25)
    return 'real selected-state field deficit beyond source high-gain supported' if yes else 'real selected-state field deficit beyond source high-gain not supported / mixed'
