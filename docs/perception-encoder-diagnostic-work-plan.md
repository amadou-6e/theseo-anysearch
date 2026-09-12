# Bounded encoder diagnostics

Issue #358. Disposition: retain. Preparation only; no new training results.

Governing protocol:
[specs amendment at aa5c020](https://github.com/amadou-6e/specs/blob/aa5c0201b6527f8c31127db15c3a3a1c96a7c9c3/projects/theseo-anysearch/python/perception-encoder-diagnostics.md).

The standalone `garden.pilots.diagnostic_routing` module separates validity stops,
diagnostic routing, and promotion. It deliberately does not modify LG1-LG3 or
the terminated topology contracts. Caller-supplied outcomes must be computed from
the frozen diagnostic protocol; this router is not a metric scorer or GPU runner.

## Work packages

| Package | Ownership | Depends on | Acceptance |
| --- | --- | --- | --- |
| Preparation | Router, tests, specs amendment | None | Typed evidence; bounded routing; old verdicts unchanged |
| Execution binding | New diagnostic harness and frozen manifest | Preparation protocol, LG2 artifacts | Exact source/spec/checkpoint hashes; all fit settings frozen; runtime cap enforcement |
| D0 | Pipeline sanity | Execution binding | Verified fixture targets, metric invariants and positive-control fit |
| D1 | Visible-input supervised reference | D0 | Three seeds, identical observation/query restrictions to D2 |
| D2 | Frozen-feature probe ladder | D0 | Three capacities, three seeds, selection-only model/threshold choice |
| Assessment | Compact report and routing evidence | D1, D2 or exhausted budget | Per-family results, uncertainty, remaining explanations and next action |

D1 and D2 are independent after D0. Conditional D3/D4 interventions require frozen
settings and available budget; they are not implemented by the preparation PR.
Quality failure is retained evidence, not permission to alter a completed study.
Invalid data or exhausted budgets still prohibit further fitting.

No merges are authorized. This branch starts from the current integration branch,
without importing unmerged LG1-LG3 implementations. The execution binding must
record any approved stacking exception before consuming those implementations.
