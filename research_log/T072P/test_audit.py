import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from audit import audit_one,validate_manifest
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
    def test_missing(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            row=audit_one(Path(d)/'absent','0'*64)
            self.assertFalse(row['match']);self.assertFalse(row['exists'])
    def test_digest_mismatch(self):
        self.assertFalse(audit_one(HERE/'audit.py','0'*64)['match'])
    def test_target_and_reference_injection(self):
        m=json.loads((HERE/'manifest.json').read_bytes());h=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        validate_manifest(m,h)
        for path in ('/target/input/1003_UHD_LL.JPG','/target/GT/image.png','/target/clean/image.png'):
            changed=copy.deepcopy(m);changed['files'][path]='0'*64
            with self.assertRaises(ValueError):validate_manifest(changed,h)
if __name__=='__main__':unittest.main(verbosity=2)
