"""Synthetic-only adversarial substitutions, including self-consistent reseals."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
def module(name):
    definition = importlib.util.spec_from_file_location(name, HERE/(name+'.py'))
    value = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(value)
    return value
verifier = module('verify')
producer = module('seal')

class ContractTests(unittest.TestCase):
    def test_accepted_source_provenance_and_seal(self):
        self.assertEqual(verifier.verify()['classification'], 'UHDLL_ANALYSIS_SPEC_SEALED')

    def test_self_consistent_unauthorized_changes_rejected(self):
        original = json.loads((HERE/'analysis_spec.json').read_text())
        changes = [
            (('cohort','dispatch_root_sha256'), '0'*64),
            (('methods','ours'), '0'*40),
            (('methods','retinexformer'), '0'*64),
            (('methods','snr_aware'), '0'*64),
            (('metric_source','commit'), '0'*40),
            (('metric_source','blobs','ttie/ssim_transfer.py'), '0'*40),
            (('bootstrap','seed'), 7), (('bootstrap','resamples'), 9999),
            (('bootstrap','sample_size'), 149), (('bootstrap','shared_stream'), False),
            (('sample_policy','exclusions'), ['first image']),
            (('sample_policy','on_missing_duplicate_nonfinite_output_geometry_binding_or_reference_mapping_error'), 'drop row'),
            (('information_boundary','prefreeze_reference_access'), True),
            (('information_boundary','reference_access'), 'after first method freezes'),
            (('cohort','images'), 149), (('cohort','jobs'), 449),
            (('endpoints','comparison_direction'), 'baseline minus ours'),
            (('endpoints','win_rule'), 'delta >= 0'),
            (('claims','extra_metrics'), ['LPIPS']),
        ]
        for path, replacement in changes:
            with self.subTest(path=path):
                changed = copy.deepcopy(original)
                node = changed
                for key in path[:-1]: node = node[key]
                node[path[-1]] = replacement
                raw = json.dumps(changed).encode()
                receipt = json.loads((HERE/'seal.json').read_text())
                receipt['analysis_spec_sha256'] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises(ValueError): verifier.verify_spec(raw, receipt)

    def test_bootstrap_reproduction_and_pair_sharing(self):
        first = producer.bootstrap_indices()
        np.testing.assert_array_equal(first, producer.bootstrap_indices())
        self.assertEqual(producer.index_digest(first), verifier.INDEX_SHA)
        self.assertEqual(first.shape, (10000,150))
        self.assertTrue(((first >= 0)&(first < 150)).all())
        # Four synthetic paired endpoints use this single matrix. Translation
        # and scaling preserve paired bootstrap sample means and intervals.
        delta = np.linspace(-1,1,150)
        means = delta[first].mean(axis=1)
        translated = (delta+2)[first].mean(axis=1)
        scaled = (delta*3)[first].mean(axis=1)
        np.testing.assert_allclose(translated, means+2, atol=1e-14)
        np.testing.assert_allclose(scaled, means*3, atol=1e-14)
        ci = np.quantile(means,[.025,.975],method='linear')
        np.testing.assert_allclose(np.quantile(translated,[.025,.975],method='linear'),ci+2)

if __name__ == '__main__': unittest.main(verbosity=2)
