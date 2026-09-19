# Restoring the recorded T060-D-R2 runtime

The commands in `report.md` are the commands executed in the recorded A6000
release. They are not standalone commands for an otherwise empty checkout.
The fixed remote project root is `/home/wenchang/asdasdsad/wjq/TTIE`.

`infer.py` reads the preflight receipt from
`ROOT/research_log/T060DR2_preflight/preflight.json`. `verify.py` additionally
reads its 60 sibling gradient records (`000.pt` through `059.pt`). The committed
`preflight.json` is an audit copy of the same receipt; moving only that file
does not restore the verifier's required records. The original runtime path
and code remain unchanged to preserve the completed experiment's source and
input bindings.

## Restore the preflight directory

Use `T060DR2_recovery.tar.gz`, described in `archives.json`:

- SHA256: `0e25e8e8cdbf75201e3a3ab870f6729ca27d673b8076cccb5552a4a33f12b486`.
- Size: 6,645,681 bytes.
- Archive member `preflight/preflight.json` matches `run_binding.json`:
  `343517e135aba509c377305aa24437b5570db7635fb3022131e13cbbc2c0b33b`.
- All 60 `preflight/NNN.pt` members match the per-record hashes in that receipt.

When the runtime preflight directory is absent, restore it from the verified
recovery archive, rather than regenerating a new receipt:

```bash
ROOT=/home/wenchang/asdasdsad/wjq/TTIE
ARCHIVE="$ROOT/shared/t060dr2/T060DR2_recovery.tar.gz"
mkdir -p "$ROOT/research_log/T060DR2_preflight"
tar -xzf "$ARCHIVE" \
  -C "$ROOT/research_log/T060DR2_preflight" \
  --strip-components=1 --wildcards 'preflight/*'
```

Regenerating the preflight changes its timestamp and receipt hash, even when
its scientific checks pass. It therefore does not reproduce the bound receipt
used by the completed experiment.

## Other required artifacts

Restore the exact `source/` tree from the recovery archive, or use the source
commits recorded in `report.md`. The recovery source tree contains the storage
resume wrapper as well as the unchanged trajectory implementation. Restore the
external source-bank tensors, model checkpoints and configuration at the paths
and hashes listed in the preflight receipt's `inputs`; these large inputs are
not supplied by a Git checkout or by the compact recovery archive.

Full frozen-output verification also requires the `artifacts/` tree from
`T060DR2_raw.tar.gz`, including every `images.pt` and `output.pt`. The compact
recovery archive deliberately omits these large tensors. The raw archive is
physically on F; its home path is a symlink, not an independent second copy.
See `archives.json` for both paths and the raw archive hash.

This restoration note does not authorize a new experiment or rerun the
completed source audit. No executable source, tolerance, scientific criterion,
frozen receipt, or existing result was changed for this documentation repair.
