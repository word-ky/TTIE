# T061-B source-only freeze

Authorization: `43d7a04fc22a74b84ef3029cd9c01197d7c0aaa1`.
Selection rule was fixed before the disclosed T061-A exposure by
`5da5a57e0b481a98d5087aacf7afde8771f80217`. No blindness claim is made.

Reuse the accepted `23c98d16c159e8f51c5147647281ae749741dade`
T060-D-R2 metrics and freeze JSON bytes. Both local files have been compared
byte-for-byte with the accepted Git blobs. The source `infer.py` routes method
A directly to `ttie.common_gain_ttt.trajectory`, region2, 40 updates; its saved
freeze identifies the 60 anchors and 41 states. No new renderer, optimizer,
metric computation, image access, or T059-E invocation is needed.

The explicit runtime data allow-list is `choose.INPUTS`: only the accepted
`research_log/T060DR2/evidence/metrics.json` and `freeze.json`. The metrics
container includes B values, but the selector accesses only A PSNR. The
script and Python standard-library bootstrap finish before installing an
audit hook and before any input-data access; after installation every Python
file open must match the two exact paths or the new output directory. Output
writes use exclusive creation, flush and fsync; a read-back SHA-256 is saved.
The hook is an auditable restriction on this minimal Python program, not an
OS sandbox. No subprocess or native file reader is used by the selector.

One bounded increment: implement source arithmetic, explicit file guard,
and independent Decimal replay. Focused synthetic tests cover all-anchor
unweighted means, earliest tie, missing/duplicate/nonfinite/cohort failures,
and denial of an actual non-allow-listed read. Commit/push tested scripts
before the sole real selector run. Run only CPU standard-library arithmetic;
GPU and new adaptation are unnecessary. Then independently verify the frozen
manifest and report it without any development evaluation.
