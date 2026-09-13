# T020-C commands and outcomes

- `python -m pytest tests/test_deadband_probe.py tests/test_direction_selector_run.py -q`:3 passed15.30s.
- `python -m pytest tests/test_nonspatial_oof.py tests/test_deadband_probe.py tests/test_direction_selector_run.py -q`:5 passed8.40s.
- `autodl-deploy.ps1 -Tag ttie-t020c-frozen -Source .autodl/T020C_stage`:release20260913-204127-ttie-t020c-frozen.
- Source Git metadata initialization initially failed because index text used CRLF; LF conversion plus empty-index refill repaired transport only. Source preflight PASS, locally and remotely.
- `autodl-run.ps1 -Name ttie-t020c-features -Cmd 'bash scripts/run_t020c_a6000.sh a4f9893c4486bd7ade6138e5f44047d29e9285c0'`:run20260913-204217-ttie-t020c-features, exit0.
- `python -m ttie.nonspatial_oof train --output research_log/T020C_run --features research_log/remote_runs/20260913-204217-ttie-t020c-features/artifacts/features --source-sha a4f9893c4486bd7ade6138e5f44047d29e9285c0`:10 heads,120 decisions frozen.
- `python research_log/T020C_verify.py replay --output research_log/T020C_run --features research_log/remote_runs/20260913-204217-ttie-t020c-features/artifacts/features`:PASS, no held-out reference read.
- `python -m ttie.nonspatial_oof evaluate --output research_log/T020C_run`:development OOF negative,3/5.
- `python research_log/T020C_verify.py metrics --output research_log/T020C_run --features research_log/remote_runs/20260913-204217-ttie-t020c-features/artifacts/features`:PASS.
- All local Python commands use D:/anaconda3/python.exe; remote GPU uses TTIE/.venv/bin/python with CUDA_VISIBLE_DEVICES=1.
