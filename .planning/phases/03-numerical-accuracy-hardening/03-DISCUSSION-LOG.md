# Phase 3: Numerical Accuracy Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-27
**Phase:** 3-Numerical Accuracy Hardening
**Areas discussed:** Kahan placement, Softmax surface, Golden suite semantics, Regression gate placement

---

## Kahan placement

| Option | Description | Selected |
|--------|-------------|----------|
| (a) per-leaf Kahan | Leaves accumulate with running compensation `c`; tree merge stays naive | |
| (b) top-level only | Single Kahan call at the result | |
| (c) full pairwise | Each merge level recomputes `c` (O(eps) error, ~1.5-2x cost) | |
| (d) hybrid — per-leaf only | Per-leaf compensated loop, naive merge; ~5% overhead, O(log n * eps) | ✓ |

**User's choice:** (d) Hybrid — per-leaf Kahan only
**Notes:** User asked about Kahan placement, then how to lock ACC-01, then whether prod() is in scope, then prod compensation method.

| Q | Options | Selected |
|---|---------|----------|
| Test that locks ACC-01 | golden fixture / inline cancellation test / rely on existing | **Add pathological_inputs golden file** |
| prod() in scope? | yes / defer | **Apply Kahan to prod() too** |
| prod compensation method | log-sum-exp / direct multiplicative / defer scope | **log-sum-exp product (Recommended)** |

---

## Softmax surface

| Option | Description | Selected |
|--------|-------------|----------|
| (a) NDArray.softmax(axis) public | First-class numja op | |
| (b) sklearn.utils.NumericalStability | sklearn-only helper | |
| (c) numja.NumericStable package-private | Reusable across modules | ✓ |
| (d) inline per sklearn consumer | No shared primitive | |

**User's choice:** Package-private NumericStable (Recommended)
**Notes:** User clarified with "tận dụng được code thì tốt" (reuse what we have). Flagged that Java package-private can't cross numja→sklearn; pivoted to **public class but NOT in facade** approach.

| Q | Options | Selected |
|---|---------|----------|
| Wiring (sklearn sees package-private class) | public class not in facade / sklearn.utils only / both mirror | **Public class, NOT in facade** (per user free-text "tận dụng được code thì tốt") |
| Reuse existing stable softmax | extract + add log primitives / extract only / leave inline | **Extract existing + add log primitives** |
| How to ACC-02 test | extend golden loader / new JUnit / modify existing softmax_extreme | **Extend existing GoldenReferenceTest loader** |

---

## Golden suite semantics

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Keep soft-fail | No behavior change | |
| (b) Flip GoldenReferenceTest to hard-fail | Single class, no duplication | |
| (c) New AccuracyHardeningTest hard-fail + soft Golden | Two classes coexist | ✓ |
| (d) JUnit Category filter | Filter at mvn level | |

**User's choice:** (c) New AccuracyHardeningTest hard-fail (Recommended)

| Q | Options | Selected |
|---|---------|----------|
| Coverage of AccuracyHardeningTest | mirror all 7 / only Phase 3 additions / only proven-passing | **(a) Mirror all 4 + 3 Phase 3 = 7 tests** |

---

## Regression gate placement

| Option | Description | Selected |
|--------|-------------|----------|
| (a) Maven failsafe integration test | CI-style, slow | |
| (b) In-process JUnit timing test | Fast, JVM-noisy | |
| (c) scripts/check_regression.ps1 | Manual, matches existing pattern | ✓ |
| (d) Combined fast + full | Two artifacts to maintain | |

**User's choice:** (c) scripts/check_regression.ps1

| Q | Options | Selected |
|---|---------|----------|
| Baseline source | fresh 03-only / extend 02 / 03-only with new ops | **(b) Extend Phase 2 baseline JSON (Recommended)** |

---

## Claude's Discretion

- Exact `tolerance_rel` numeric values inside per-fixture JSON (schema locked, values per Phase 1 baseline pattern of ~10x machine epsilon)
- Whether `ParallelOps` exposes `compensatedSum()` method or inlines the leaf loop (placement locked, internal style free)
- `GoldenReferenceTest` refactor to share `load()` + `check()` via `GoldenFixtures` utility (forced by D-08)

## Deferred Ideas

- **Vector API for Kahan inner loop** — bandwidth-bound, SIMD helps leaf throughput. Re-evaluate Phase 5/6.
- **EJML mt- variant for matmul/elementwise** — vendor decision deferred to Phase 5 (ComputeBackend abstraction).
- **ComputeBackend abstraction** — Phase 5 scope.
- **`multiply_elementwise` 10⁷ 1.42x EJML SIMD ceiling** — Phase 5 can reroute via ComputeBackend; Phase 3 doesn't reopen.
- **NaN handling in `prod()`** — Phase 3 Kahan variant preserves existing behavior; documented, no change.
- **Vector API for hot loops (CPU-03)** — deferred per Phase 2 verdict; JDK 25 still preview.
