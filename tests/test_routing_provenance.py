import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from ttie.routing.provenance import verify_source


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'core.autocrlf', 'false')
        self.path = self.root / 'scientific.py'
        self.path.write_bytes(b'value = 1\n')
        self.commit = self.save()

    def git(self, *args):
        return subprocess.run(['git', '-c', 'user.name=Fixture', '-c',
                               'user.email=fixture@example.invalid', *args],
                              cwd=self.root, check=True, capture_output=True).stdout

    def save(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD').decode().strip()

    def check(self, commit=None, paths=('scientific.py',)):
        return verify_source(commit or self.commit, paths, root=self.root)

    def test_correct_commit_and_bookkeeping_changes_pass(self):
        (self.root / 'notes.md').write_text('committed bookkeeping')
        self.save()
        (self.root / 'notes.md').write_text('uncommitted bookkeeping')
        self.assertEqual(self.check(), {
            'scientific.py': hashlib.sha256(self.path.read_bytes()).hexdigest()})

    def test_stale_commit_fails_and_new_commit_passes(self):
        self.path.write_bytes(b'value = 2\n')
        new_commit = self.save()
        with self.assertRaisesRegex(ValueError, 'differs from'):
            self.check()
        self.check(new_commit)

    def test_modified_runtime_file_fails(self):
        self.path.write_bytes(b'value = 1\r\n')
        with self.assertRaisesRegex(ValueError, 'differs from'):
            self.check()

    def test_staged_change_with_restored_runtime_bytes_fails(self):
        self.path.write_bytes(b'value = 2\n')
        self.git('add', 'scientific.py')
        self.path.write_bytes(b'value = 1\n')
        with self.assertRaisesRegex(ValueError, 'Uncommitted scientific'):
            self.check()

    def test_missing_git_blob_fails(self):
        (self.root / 'missing.py').write_bytes(b'present only at runtime\n')
        with self.assertRaises(subprocess.CalledProcessError):
            self.check(paths=('missing.py',))

    def test_missing_runtime_file_fails(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_invalid_sha_fails(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.check('not-a-commit')

    def test_git_command_failure_is_not_ignored(self):
        with patch('ttie.routing.provenance.subprocess.run',
                   side_effect=subprocess.CalledProcessError(128, ['git'])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.check()

    def test_directory_without_git_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.CalledProcessError):
                verify_source(self.commit, ('scientific.py',), root=tmp)

    def pilot_args(self):
        args = ['pilot', '--source-sha', self.commit]
        for key in ('manifest', 'source-manifest', 'images', 'model-identity',
                    'prototypes', 'receipt', 't006-images', 'energy', 'control',
                    'energy-receipt'):
            args += ['--' + key, str(self.root / key)]
        return args + ['--output', str(self.root / 'audit')]

    def test_failed_guard_precedes_assets_models_evaluation_and_output(self):
        from ttie.routing import pilot
        self.path.write_bytes(b'value = 2\n')
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv', self.pilot_args()))
            stack.enter_context(patch.object(pilot, 'T015_SOURCES', ('scientific.py',)))
            stack.enter_context(patch.object(pilot, 'verify_source', side_effect=
                lambda source, paths: verify_source(source, paths, root=self.root)))
            untouched = [stack.enter_context(patch.object(owner, name)) for owner, name in (
                (pilot.torch, 'manual_seed'), (pilot, 'verify_receipt'),
                (pilot.FrozenCLIP, 'from_checkpoint'), (pilot.torch, 'load'),
                (pilot, 'load_energy'), (pilot, 'load_image'), (pilot, 'evaluate'),
                (pilot, 'write'))]
            with self.assertRaisesRegex(ValueError, 'differs from'):
                pilot.main()
            for mock in untouched:
                mock.assert_not_called()
        self.assertFalse((self.root / 'audit').exists())

    def test_correct_guard_allows_preflight_only_after_verification(self):
        from ttie.routing import pilot
        calls = []

        def checked(source, paths):
            result = verify_source(source, paths, root=self.root)
            calls.append('verified')
            return result

        class PreflightReached(Exception):
            pass

        def stop_at_assets(*args):
            calls.append('asset preflight')
            raise PreflightReached

        with patch('sys.argv', self.pilot_args()), \
                patch.object(pilot, 'T015_SOURCES', ('scientific.py',)), \
                patch.object(pilot, 'verify_source', side_effect=checked), \
                patch.object(Path, 'read_text', return_value='{}'), \
                patch.object(pilot, 'verify_receipt', side_effect=stop_at_assets), \
                patch.object(pilot.FrozenCLIP, 'from_checkpoint') as model, \
                patch.object(pilot, 'evaluate') as evaluate:
            with self.assertRaises(PreflightReached):
                pilot.main()
            model.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(calls, ['verified', 'asset preflight'])
        self.assertFalse((self.root / 'audit').exists())


if __name__ == '__main__':
    unittest.main()
