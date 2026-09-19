# T061-C fixed candidate evaluation

Authorization: `9bece47c33da0465faf5165e2f9bca74fb183c32`.
Reuse the accepted T061-B manifest and two T037-A CSVs, last committed at
`c0d84b1d3c7e6af186c28ca736d6ac2bc752d35c`. Preparation has only hashed the
development bytes and checked them against Git; no quality field has been
parsed or displayed. Exact bindings are constants in `analyze.py`.

One implementation increment: prepare an fsynced evaluation intent with the
immutable step, bindings and five preregistered gates; a separate invocation
then reads development fields for offline evaluation. Reuse the accepted CSV
schema and selected-output metrics, with no render, optimizer, model or image
access. Test on synthetic tables before committing/pushing scripts, then run
prepare once, evaluate once and independently verify all 100 fixed lookups.
CPU standard-library arithmetic is sufficient. No alternate candidate or
post-hoc selection. T061-A's historical summary exposure remains disclosed;
this is evaluation of the independently frozen T061-B candidate.
