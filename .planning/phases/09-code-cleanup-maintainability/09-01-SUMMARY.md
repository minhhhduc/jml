# 09-01 SUMMARY — Fix stale docs, untrack binaries/caches (MNT-01)

## What changed
Task 1 — stale doc contradictions (CONCERNS.md section 4), docs only:
- `docs/BUILD_AND_SETUP.md`: removed all Python-package instructions
  (`pip install`, `src/numja/tests.py`, pytest/sphinx/twine sections);
  replaced with the real Windows/PowerShell Maven flow
  (`C:\Users\Admin\.maven\maven-3.9.15\bin` on PATH per shell +
  `mvn -pl <module> -am test`). CI example now runs the module-scoped
  `mvn -pl modules/numja,modules/pandas,modules/sklearn test` on
  windows-latest / JDK 17+25. Verified facts while editing: no `mvnw`
  wrapper exists in repo; Maven 3.9.15 present; default JDK Temurin 25.0.3.
- `README.md`: (a) "targeting Java 11 through Java 25+" replaced with what
  the build actually does — release jars compile `--release 11`
  (`scripts/build_core.ps1:37`), Maven modules use source/target 17,
  runs on JDK 17/25; compatibility above the build JDKs explicitly marked
  unverified / upper bound unknown. (b) "zero-dependency" corrected to
  stdlib-only production core + vendored math/plot deps in `dist/libs/`.
- `MODULES_INVENTORY.md`: had no `zero-dependency` string (CONCERNS.md
  pointer was stale — the claim lived in README, fixed there); added an
  explicit "Dependency posture" note naming the vendored jars and versions.
- Dead code: full scan of `modules/*/src/main` (114 private declarations,
  temp script, repo untouched) found **zero** private methods with no
  callers repo-wide (spot-checked `gate`, `checkSize`, `activeCeiling`,
  `calculateOptimalThreads`, `updateEnvironment`, `buildChart`,
  `computeDeterminant`, KMeans helpers — all called). No deletions made
  (also required: HARD FENCE forbids `modules/*/src` edits). Nothing
  deferred — nothing found.

Task 2 — untrack binaries/caches:
- `.gitignore`: appended `libs/proguard-7.3.2/`, `dist/*.jar` (+
  `dist/**/*.jar` so `dist/libs/*.jar` stays ignored — bare `dist/*.jar`
  does not cross `/` in gitignore semantics), `.gradle/`,
  `**/.gradle/`, `confusion_matrix.png`.
- `git rm -r --cached`: `libs/proguard-7.3.2` (221 files, incl. 100
  `.gradle` cache files under examples/), all 9 `dist/*.jar`
  (5 top-level + 4 `dist/libs/` vendored deps), root
  `confusion_matrix.png`. Never `-f`, no worktree deletes.
- `confusion_matrix.png` decision: repo-wide markdown grep finds NO doc
  referencing it as committed sample output (only CONCERNS.md flags it as
  a tracked-binary concern) → treated as generated artifact, untracked.
- `libs/` non-proguard files (9: ejml/commons-math/jfreechart jars,
  install ps1, module jars) left tracked — out of this plan's scope.

## Tracking state (post-commit)
- `git ls-files "libs/proguard-7.3.2"` → 0 (was 221)
- `git ls-files "dist/*.jar"` → 0 (was 9)
- `git ls-files | Select-String "\.gradle"` → 0 (was 100)
- `git ls-files | Select-String "confusion_matrix"` → 0 (was 1)
- On-disk asserts all True: `libs/proguard-7.3.2`, `dist`,
  `confusion_matrix.png`, `dist/numja.jar`,
  `dist/libs/ejml-core-0.43.1.jar`,
  `libs/proguard-7.3.2/lib/proguard.jar`.

## Test / smoke results
- Task 1 automated verify: PASS
  (`pip install` absent + `mvn -pl` present; README upper-bound wording;
  `zero-dependency` absent from MODULES_INVENTORY).
- Task 2 smoke: `mvn -q -pl modules/numja,modules/pandas,modules/sklearn
  test` → exit 0 (compile + tests do not depend on untracked paths).
- Fence honored: zero edits under `modules/*/src`, `bench/*/src`,
  `scripts/`, `pom.xml`, `*-PLAN.md` (build_core.ps1/poms only read).

## Concurrency note
- `modules/sklearn/.../KMeans.java` worktree modification by the
  concurrent 08-04 executor left untouched (unstaged, not in index).
- Pre-existing `M .planning/ROADMAP.md` and untracked
  `.planning/.omp-ui-state.json`, `08-04/08-05-PLAN.md` left alone;
  SUMMARY is the only `.planning` file staged.

## Commit
- Message: `chore(09-01): fix stale docs, untrack binaries/caches`
- Staged: docs (3 files) + `.gitignore` + index deletions + this SUMMARY.
- `git status` clean apart from the other executor's / pre-existing entries.
