import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).parents[1]/'research_log/T037A_diagnostic/core.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def test_earlier_rescue_excludes_selected_and_uses_first_best_tie():
    q=dict(baseline=dict(psnr=[10,12,11],ssim=[.5,.6,.5]),common=dict(psnr=[12,12,9],ssim=[.4,.7,.3]))
    r=d.image_diagnostic(0,'low',q,dict(baseline=1,common=2))
    assert r['common_reference_best_psnr_step']==0 and r['psnr_headroom']==3
    assert r['prior_psnr_loss'] and r['earliest_psnr_rescue_step']==0
    assert r['common_reference_best_ssim_step']==1
    r=d.image_diagnostic(0,'low',q,dict(baseline=1,common=0))
    assert not r['earlier_psnr_reaches_baseline']
def test_both_frozen_verdict_conditions_required():
    assert d.verdict(.75,15)=='strong late-selection headroom'
    assert d.verdict(.749,29)=='limited/mixed late-selection headroom'
    assert d.verdict(2.,14)=='limited/mixed late-selection headroom'
