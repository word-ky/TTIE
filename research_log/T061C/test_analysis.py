import copy
import unittest
from research_log.T061C.analyze import paired,summarize

def fixture():
    steps=[];images=[]
    for i in range(100):
        low=f'image{i}'
        images.append(dict(index=str(i),low=low,common_selected_step='40',baseline_selected_step='40',
            common_selected_psnr='10.4',baseline_selected_psnr='10',common_selected_ssim='.5',baseline_selected_ssim='.5'))
        for method in ['baseline','common']:
            for k in range(41):
                steps.append(dict(index=str(i),low=low,method=method,step=str(k),psnr=str(10+k/100 if method=='common' else 10),ssim='.5'))
    return steps,images

class Tests(unittest.TestCase):
    def test_fixed_step_lookup_and_negative_gate(self):
        rows=paired(*fixture());r=summarize(rows)
        self.assertEqual(len(rows),100)
        self.assertTrue(all(x['step11_psnr']==10.11 for x in rows))
        self.assertAlmostEqual(r['stats']['mean_psnr_vs_t036'],-.29)
        self.assertEqual(r['stats']['counts_vs_t036']['regress'],100)
        self.assertEqual(r['stats']['counts_vs_t026']['improve'],100)
        self.assertEqual(r['verdict'],'NEGATIVE')

    def test_required_table_failures(self):
        for case in ['missing','duplicate','nonfinite','order','selected']:
            steps,images=fixture()
            if case=='missing':steps.pop()
            if case=='duplicate':steps[-1]=steps[-2].copy()
            if case=='nonfinite':steps[-1]['psnr']='nan'
            if case=='order':images[0],images[1]=images[1],images[0]
            if case=='selected':images[0]['common_selected_psnr']='0'
            with self.subTest(case=case),self.assertRaises(AssertionError):paired(steps,images)

if __name__=='__main__':unittest.main()
