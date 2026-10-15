---
phase: 08-performance-optimization
plan: 04
subsystem: sklearn-clustering
tags: [kmeans, hot-path, fusion, intermediate-reduction]

requires:
  - phase: 08-performance-optimization
    provides: Prior OPT-02 intermediate-reduction passes (08-01, 08-02, 08-03)
provides:
  - Fused KMeans assign+inertia single-pass loop with single centroid mirror
  - Single NumJa wrap of final centroids per fit
affects: [phase-08-plan-05, sklearn-clustering]

tech-stack:
  added: []
  patterns: [fused assign+inertia pass, single centroid mirror with in-place update]

key-files:
  modified:
    - modules/sklearn/src/main/java/sklearn/cluster/KMeans.java

key-decisions:
  - "Accumulate per-cluster sums inside the fused assign loop so centroid recompute costs no second sample pass."
  - "Keep convergence on pre-update inertia with sqrt-then-square dist*dist math for a bit-identical trajectory."
  - "Leave predict on its existing single-extract shape; no safe bulk-copy gain to chase there."
patterns-established:
  - "One centroid mirror per fit, wrapped into NDArray once after the loop."
requirements-completed: [OPT-02]
duration: 15 min
completed: 2026-10-15
---

# Phase 08 Plan 04: KMeans Assign+Inertia Fusion Summary

**Fused KMeans' three per-iteration passes into one assign+inertia loop over a single centroid mirror, with zero result change.**

## Performance
- **Duration:** 15 min
- **Started:** 2026-10-15T12:50:00Z
- **Completed:** 2026-10-15T13:05:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced `assignClusters` + `computeCentroids` + `computeInertia` (three sample passes, two centroid extracts, one NDArray alloc per iteration) with one `assignAndInertia(data, centroidsData, sums, counts)` pass that assigns labels, accumulates pre-update inertia from the same distance computation, and accumulates per-cluster sums in the same scan.
- Kept one `double[][] centroidsData` mirror for the whole fit, updated in place via `System.arraycopy` after the convergence check; `this.centroids = NumJa.array(centroidsData)` now runs once after the loop.
- Preserved exact inertia math (`dist*dist` with `dist = euclideanDistance`, sqrt then square), pre-update convergence comparison, `n_iter_` break/assign semantics, and empty-cluster zero rows.
- Left `predict` on its existing single-extract shape; removed all dead private methods.

## Task Commits
1. **Task 1: Fuse assign+inertia with single centroid mirror** - `cafe5be` (feat)
2. **Task 2: Full sklearn suite green + commit** - `cafe5be` (feat)

## Files Created/Modified
- `modules/sklearn/src/main/java/sklearn/cluster/KMeans.java` - Fused loop (fit body lines 54-97, `assignAndInertia` lines 165-194); 37 insertions, 52 deletions.

## Decisions Made
- Accumulated per-cluster sums inside the fused loop (label known within the same `i` iteration, same `i=0..n-1` addition order) so the means are bit-identical with no second pass.
- Updated the mirror only on non-converged iterations so a convergence break keeps pre-update centroids, matching the old break-without-update behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. `ParallelRegressionTest` did not flake on either run (numja-core 47/47 green both times), so no rerun was needed.

## Test Results

- `mvn -pl modules/sklearn -am test "-Dtest=TestSklearnJUnit"`: 1/1 green, BUILD SUCCESS.
- `mvn -pl modules/sklearn -am test` (full): numja-core 47/47, pandas-wrapper 27/27, sklearn 22/22 (1 pre-existing skip in GoldenReferenceTest), BUILD SUCCESS.
- `mvn -pl modules/numja test`: 47/47, BUILD SUCCESS.
- Public API check: `fit`/`predict`/`getCentroids`/`getLabels`/`getNIter` signatures unchanged.

## User Setup Required

None.

## Next Phase Readiness

- KMeans leg of OPT-02 is done; 08-05 can proceed on the remaining hot paths.
- No API or behavioral migration needed by downstream consumers.

---
*Phase: 08-performance-optimization*
*Completed: 2026-10-15*

## Self-Check: PASSED
- Confirmed `assignAndInertia` present, `extract2DArray(centroids)` count is 1 (predict only), `NumJa.array(centroidsData)` single wrap present.
- Confirmed commit `cafe5be` contains only `KMeans.java` and the working tree is clean for that file.
