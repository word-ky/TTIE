from research_log.T068A.core import capped_step,select
from research_log.T067B.core import choices

def test_complete_caps_interval_and_endpoints():
    for fs in range(28):
        for interior in range(fs,28):
            for cap in range(28):
                got=capped_step(fs,interior,cap)
                assert fs<=got<=interior
                assert got==(fs if cap<fs else interior if cap>=interior else cap)
            assert capped_step(fs,interior,27)==interior

def test_no_cap_reproduces_interpolation():
    for fs in range(28):
        old=next(r for r in choices([1-k/27 for k in range(28)],[.1]*fs+[.5]*(28-fs),27) if r['lambda_value']==.875)
        assert capped_step(old['k_FS'],old['selected_step'],27)==old['selected_step']
        assert capped_step(old['k_FS'],old['selected_step'],0)==old['k_FS']

def test_ranking_all_four_keys_and_gate_filter():
    def row(k,w,m,med,passed=True):return dict(K=k,worst_delta_t026=w,mean_delta_psnr=m,median_delta_psnr=med,gates={'all':passed})
    rows=[row(27,-1,3,1),row(26,-1,3,1),row(25,-1,3,2),row(24,-1,4,0),row(23,0,2,0),row(22,1,9,9,False)]
    assert select(rows)[2]==[23,24,25,27,26]
    assert select(rows)[1]=='ABS_STEP_CAP_DEV_CANDIDATE_FROZEN'
    assert select(rows[:2])[0]['K']==27 and select(rows[:2])[1]=='ABS_STEP_CAP_DEV_NO_GAIN'
