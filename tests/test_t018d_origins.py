import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('origins',ROOT/'research_log/T018D_origins.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PACK=ROOT/'research_log/T018D_origin_objects.pack'
RECEIPT=json.loads((ROOT/'research_log/T018D_run/selector_frozen.json').read_text())


class OriginTests(unittest.TestCase):
    def test_all_fifteen_original_locations_match_frozen_sha256(self):
        expected={(RECEIPT['source_sha'],p):h for p,h in RECEIPT['source_code_sha256'].items()}
        expected.update({(v['commit'],v['path']):v['sha256'] for v in RECEIPT['input_artifact_hashes'].values()})
        actual=module.read_origins(PACK,expected)
        self.assertEqual({k:hashlib.sha256(v).hexdigest() for k,v in actual.items()},expected)

    def test_incorrect_commit_is_rejected_even_with_unchanged_payload_hash(self):
        item=RECEIPT['input_artifact_hashes']['score_config']
        with self.assertRaises(subprocess.CalledProcessError):module.read_origins(PACK,[('0'*40,item['path'])])

    def test_incorrect_path_is_rejected_even_with_unchanged_payload_hash(self):
        item=RECEIPT['input_artifact_hashes']['score_config']
        with self.assertRaises(subprocess.CalledProcessError):module.read_origins(PACK,[(item['commit'],'research_log/not-the-recorded-origin.json')])


if __name__=='__main__':unittest.main()
