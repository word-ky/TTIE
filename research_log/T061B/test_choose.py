import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from research_log.T061B.choose import select

def fixture():
    rows, frozen = [], []
    for i in range(60):
        keys = dict(order=i, index=i, bank_index=i, image_id=i // 4)
        values = [1.0] * 41
        values[3] = values[9] = 2.0 + i
        rows.append(dict(**keys, A=dict(psnr=values)))
        frozen.append(dict(**keys, methods=dict(A=dict(states=41, updates=40))))
    return rows, frozen

class Tests(unittest.TestCase):
    def test_all_anchors_unweighted_and_earliest_tie(self):
        means, ties, step = select(*fixture())
        self.assertEqual(means[3], 31.5)
        self.assertEqual(ties, [3, 9])
        self.assertEqual(step, 3)

    def test_required_invalid_inputs_stop(self):
        original, frozen = fixture()
        for case in ['missing_anchor', 'duplicate', 'missing_step', 'nonfinite', 'cohort']:
            rows = copy.deepcopy(original)
            if case == 'missing_anchor': rows.pop()
            if case == 'duplicate': rows[1]['bank_index'] = 0
            if case == 'missing_step': rows[0]['A']['psnr'].pop()
            if case == 'nonfinite': rows[0]['A']['psnr'][0] = float('nan')
            if case == 'cohort': rows[0]['image_id'] = 999
            with self.subTest(case=case), self.assertRaises(AssertionError):
                select(rows, frozen)

    def test_guard_denies_nonallowlisted_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / 'source.json'; allowed.write_text('{}')
            output = root / 'output'; output.mkdir()
            denied = root / 'T037_dev_summary.json'; denied.write_text('{}')
            code = '''from pathlib import Path
from choose import install_guard
import sys
a,o,d=map(Path,sys.argv[1:])
reads=install_guard([a],o)
assert a.read_text()=='{}'
try:
    d.read_text()
except PermissionError:
    print('DENIED')
else:
    raise AssertionError('guard permitted development file')
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, str(allowed), str(output), str(denied)], cwd=Path(__file__).parent, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('DENIED', result.stdout)

if __name__ == '__main__':
    unittest.main()
