# T059-R
Authorization c855b29275a203b6de2c110c5d576ca038db0a7c. Reuse Q-aligned E/M persisted prediction rows; recheck original selected p storages and metadata. No target tensors until descriptor freeze.
Descending uncertainty rank is one-based; percentile from most uncertain = 100*(rank-1)/(N-1). Top quartile/decile use this fixed percentile <=25/<=10. For N=63, these are ranks 1..16 and 1..7. Fixed before computing descriptors or opening targets. Exact u ties: smallest bank ID.
CPU only; no model/feature forwards.
