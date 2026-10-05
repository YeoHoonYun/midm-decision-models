# Four-benchmark scoring — 2026-10-05

Stored predictions rescored; BF16 primary matched run. No new inference implied.

| Benchmark | 4B | 9B |
|---|---:|---:|
| typed_decisions_test | 1577/2000 (78.85%) | 1586/2000 (79.30%) |
| jevbench_public | 165/231 (71.43%) | 178/231 (77.06%) |
| jevbench_hard | 51/111 (45.95%) | 60/111 (54.05%) |

CLM DeepSWE fresh replay: 31/38 (81.58%). MiDM DeepSWE not scored: raw candidate inputs missing.
Terminal-Bench not scored: matching traces/head/protocol absent. Published 87.6% is not substituted as a measurement.
JevBench values are public-item accuracy, not official composite points. Typed-decisions values are hard-label argmax accuracy; compare published distribution-based scores only after matching the metric.
Missing results are null, never zero. See SCORES.json for source hashes and protocol.
