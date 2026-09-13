"""Resolve original commit:path bytes from a minimal archived Git object pack."""
from pathlib import Path
import subprocess
import tempfile


def read_origins(pack, pairs):
    with tempfile.TemporaryDirectory() as directory:
        git = ['git', '--git-dir', directory]
        subprocess.run(git + ['init', '--bare', '-q'], check=True, capture_output=True)
        subprocess.run(git + ['index-pack', '--stdin'], input=Path(pack).read_bytes(), check=True, capture_output=True)
        return {(commit, path): subprocess.check_output(git + ['show', commit + ':' + path], stderr=subprocess.PIPE)
                for commit, path in pairs}
