"""T040 fixed high-gain probes; all rendering/gradient conventions reuse T039."""
from research_log.T039A_tangent.core import *
GAINS=[1.5,1.75]
def classify(s):
    a,b=s['legacy'],s['gain']
    yes=(a['cosine_median'] is not None and b['cosine_median'] is not None and a['positive_dot_fraction']-b['positive_dot_fraction']>=.2 and a['cosine_median']-b['cosine_median']>=.25)
    return 'source high-gain tangent deficit supported' if yes else 'source high-gain tangent deficit not supported / target-domain state shift remains stronger'
