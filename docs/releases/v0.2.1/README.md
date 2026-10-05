# MiDM v0.2.1 — benchmark and architecture atlas

Documentation and analysis release; no model weight, tokenizer, loader or inference configuration changes.

- Adds 13 model/settings comparisons with provenance and five figures in PNG/SVG: matched parameter performance, contextual size/performance, public cohort comparison, architecture diagram, historical architecture control.
- Counts trainable adapter + pointer tensors exactly: 30,476,801 (4B), 40,112,129 (9B). Nominal full base sizes remain separate.
- Completes coverage explanations for Typed Decisions, Kev transfer, JevBench, DeepSWE, Terminal-Bench, SQL/Python and routing. Unmeasured MiDM DeepSWE/Terminal-Bench and official JevBench composite are not invented.
- Preserves regressions, unknown proprietary sizes, published-versus-measured provenance and incompatible metric caveats.
- Corrects stale root comparison/citation text. Includes aggregate JSON, source hashes and a network-free plotting script.

Full atlas: https://github.com/YeoHoonYun/midm-decision-models/blob/v0.2.1/docs/evaluations/20261005/atlas/README.md

Version DOI: https://doi.org/10.5281/zenodo.23164651

Previous weight release: https://github.com/YeoHoonYun/midm-decision-models/releases/tag/v0.2.0
