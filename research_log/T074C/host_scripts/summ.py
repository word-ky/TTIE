import json, sys, statistics as st
m = json.load(open(sys.argv[1] + "/output_manifest.json"))
r = m["rows"]; s = [x.get("whole_run_seconds", 0) for x in r]; v = [x.get("peak_gpu_memory_bytes", 0) for x in r]
extra = {k: m[k] for k in m if k not in ("rows",) and not isinstance(m[k], (list, dict))}
print(json.dumps({"n": len(r), "sum_s": round(sum(s), 1), "median_s": round(st.median(s), 3), "max_peak_gib": round(max(v) / 2**30, 3), **extra}))
