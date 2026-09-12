from pathlib import Path
import tempfile
import unittest
import torch
from ttie.stop_quality import QualityHead,train_head,save_head,load_head,RECIPE


class StopQualityTests(unittest.TestCase):
    def test_fixed_recipe_train_only_normalization_and_exact_repeat(self):
        torch.manual_seed(19);features=torch.randn(30,33);features[:,3]=1.
        mse=torch.linspace(0.,.1,30)
        a,ha=train_head(features,mse);b,hb=train_head(features,mse)
        self.assertEqual(len(ha),100);self.assertEqual(ha,hb);self.assertEqual(RECIPE['batch_size'],256)
        self.assertTrue(torch.equal(a.x_mean,features.double().mean(0).float()));self.assertEqual(float(a.x_scale[3]),1.)
        self.assertTrue(torch.equal(a.y_mean,(mse.double()+1e-6).log().mean().float()))
        for x,y in zip(a.state_dict().values(),b.state_dict().values()):self.assertTrue(torch.equal(x,y))
        self.assertTrue(all(not p.requires_grad for p in a.parameters()))
        with self.assertRaises(TypeError):train_head(features,mse,calibration_features=features*100)

    def test_checkpoint_frozen_scores_and_cpu_cuda_selection(self):
        torch.manual_seed(7);features=torch.randn(20,33);head,_=train_head(features,torch.linspace(.001,.1,20))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.pt';save_head(head,path);other=load_head(path)
            expected=head(features);actual=other(features)
            self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(actual,other(features)))
            self.assertEqual(int(expected.argmin()),int(actual.argmin()))
            if torch.cuda.is_available():
                other=other.cuda();a=other(features);b=other(features)
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.allclose(expected,a.cpu(),atol=1e-5,rtol=1e-5))
                self.assertEqual(int(expected.argmin()),int(a.argmin()))
