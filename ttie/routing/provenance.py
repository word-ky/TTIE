"""Bind future runtime scientific files to a declared Git commit before launch."""
import hashlib
from pathlib import Path
import subprocess


# The accepted donor inventory, routing modules, and T015 entry points.
# Explicit paths keep bookkeeping outside the scientific freeze boundary.
T015_SOURCES = tuple("""
ttie/__init__.py ttie/adapt.py ttie/clip_audit.py ttie/clip_signal.py ttie/demo.py
ttie/energy_bank.py ttie/energy_io.py ttie/energy_metrics.py ttie/energy_model.py
ttie/energy_pilot.py ttie/energy_receipt.py ttie/energy_ttt.py
ttie/geometry_audit.py ttie/geometry_summary.py ttie/isp.py ttie/joint_audit.py
ttie/joint_gate.py ttie/learned_prototypes.py ttie/natural.py ttie/offline_geometry.py
ttie/projected_metrics.py ttie/projected_pilot.py ttie/projected_ttt.py
ttie/prototype_audit.py ttie/regularization.py ttie/relative_audit.py ttie/relative_clip.py
ttie/residual_data.py ttie/residual_metrics.py ttie/residual_pilot.py ttie/residual_ttt.py
ttie/restoration_metrics.py ttie/restoration_pilot.py ttie/safety_summary.py
ttie/safety_sweep.py ttie/semantic_ttt.py ttie/sobolev_io.py ttie/sobolev_metrics.py
ttie/sobolev_pilot.py ttie/sobolev_receipt.py ttie/sobolev_source.py ttie/sobolev_train.py
ttie/stop_io.py ttie/stop_metrics.py ttie/stop_pilot.py ttie/stop_quality.py
ttie/stop_receipt.py ttie/stop_trajectory.py ttie/suite.py ttie/summarize.py
ttie/routing/__init__.py ttie/routing/core.py ttie/routing/io.py
ttie/routing/metrics.py ttie/routing/pilot.py ttie/routing/provenance.py
scripts/prepare_t015.py scripts/run_t015_a6000.sh
""".split())


def verify_source(source_sha, paths, *, root=None):
    """Return verified byte hashes; Git, file, or provenance failures abort launch."""
    root = Path.cwd() if root is None else Path(root)

    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    commit = git('rev-parse', '--verify', '--end-of-options',
                 source_sha + '^{commit}').decode().strip()
    hashes = {}
    for path in paths:
        blob = git('show', commit + ':' + path)
        runtime = (root / path).read_bytes()
        if runtime != blob:
            raise ValueError(f'Scientific file differs from {commit}: {path}')
        hashes[path] = hashlib.sha256(runtime).hexdigest()
    if git('rev-parse', '--is-inside-work-tree').strip() == b'true':
        dirty = git('status', '--porcelain', '--untracked-files=all', '--', *paths)
        if dirty:
            raise ValueError('Uncommitted scientific paths: ' + dirty.decode().strip())
    return hashes
