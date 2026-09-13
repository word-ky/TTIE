import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ttie.direction_probe import DirectionHead, RECIPE, axis_features
from ttie.direction_selector import DirectionSelector, cross_features, load_selector, INFERENCE_SOURCES


class DirectionSelectorTests(unittest.TestCase):
    def fixture(self, folder):
        torch.manual_seed(7); selector = DirectionSelector(DirectionHead(), DirectionHead())
        for axis in ('x', 'y'):
            torch.save(dict(state_dict=getattr(selector, 'head_' + axis).state_dict(), recipe=RECIPE), folder / ('head_' + axis + '.pt'))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')}
        receipt = dict(recipe=RECIPE, files_sha256=hashes, inference_code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in INFERENCE_SOURCES})
        path = folder / 'selector_frozen.json'; path.write_text(json.dumps(receipt), encoding='utf-8')
        return selector, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_five_feature_api_exact_donor_equivalence_and_ties(self):
        f = torch.arange(3*9*28, dtype=torch.float32).reshape(3,9,28)
        actual = cross_features(f[:,4], f[:,1], f[:,7], f[:,3], f[:,5]); prior = axis_features(f)
        for i in (0,1): self.assertTrue(torch.equal(actual[i], prior[:,i]))
        self.assertEqual(list(inspect.signature(DirectionSelector.predict).parameters), ['self','center','x_lower','x_upper','y_lower','y_upper'])
        x = DirectionHead(); y = DirectionHead()
        with torch.no_grad():
            for head in (x,y):
                for v in head.parameters(): v.zero_()
            y.net[-1].bias.copy_(torch.tensor([0.,1.,1.]))
        output = DirectionSelector(x,y).predict(*[torch.zeros(28)]*5)
        self.assertEqual(output, [dict(x_logits=[0.,0.,0.], y_logits=[0.,1.,1.], x_class=0, y_class=1, bx=.5, by=.4, score_index=3, hard_index=9)])

    def test_save_reload_and_reference_artifact_mutation_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); selector, pinned = self.fixture(folder)
            inputs = [torch.randn(4,28) for _ in range(5)]; expected = selector.predict(*inputs)
            forbidden = ['reference_mse.json','targets.json','condition.json','image_ids.json','oracle.json']
            for name in forbidden: (folder/name).write_text('{"value":1}', encoding='utf-8')
            real_read = Path.read_bytes
            def guarded(path):
                if path.name in forbidden: raise AssertionError('Reference artifact opened')
                return real_read(path)
            with patch.object(Path, 'read_bytes', guarded), patch('subprocess.check_output', side_effect=AssertionError('No Git or target access during inference')):
                self.assertEqual(load_selector(folder,pinned).predict(*inputs),expected)
                for name in forbidden: (folder/name).write_text('{"value":999}', encoding='utf-8')
                self.assertEqual(load_selector(folder,pinned).predict(*inputs),expected)
                for name in forbidden: (folder/name).unlink()
                self.assertEqual(load_selector(folder,pinned).predict(*inputs),expected)

    def test_pinned_receipt_and_head_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); _,pinned=self.fixture(folder)
            with self.assertRaisesRegex(ValueError,'receipt hash'): load_selector(folder,'0'*64)
            with (folder/'head_x.pt').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Head hash'): load_selector(folder,pinned)


if __name__ == '__main__': unittest.main()
