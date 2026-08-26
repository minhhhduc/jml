# Technology Stack

> Last updated: 2026-08-25

## Summary

NumJa is a **closed-source Java machine learning library** distributed as five obfuscated JAR modules simulating the Python data-science API surface (NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn). Source lives in `modules/` but is git-ignored; only compiled JARs in `dist/`, examples, scripts, and docs are tracked.

## Languages & Runtime

| Language | Version / Notes |
|----------|-----------------|
| Java | `--release 11` compile target ([scripts/build_core.ps1:41](../../scripts/build_core.ps1)); docs claim compatibility "Java 11 through Java 25+" |
| PowerShell | Build/install orchestration (`scripts/*.ps1`) |
| Batch (cmd) | Per-module build/install wrappers (`scripts/build/*.bat`, `scripts/install/*.bat`) |
| Python | Legacy/dev only — `.venv/` present, [docs/BUILD_AND_SETUP.md](../../docs/BUILD_AND_SETUP.md) mentions `src/numja/tests.py` which no longer exists in tree |

## Build System

**No Maven/Gradle.** Plain `javac` via PowerShell:

- [scripts/build_core.ps1](../../scripts/build_core.ps1) — main builder. Discovers all `*.java` under `modules/` (excluding `src/test` and `target`), writes `build/sources.txt`, compiles to `build/classes` with classpath from `libs/*.jar`, packages per-module JARs.
- [scripts/build_check.ps1](../../scripts/build_check.ps1) — post-build sanity check.
- [scripts/build/build_obfuscated.ps1](../../scripts/build_obfuscated.ps1) + `build_obfuscated.bat` — ProGuard obfuscation pass producing release JARs.
- ProGuard 7.3.2 vendored at `libs/proguard-7.3.2/` (+ zip).

## Dependencies

Runtime (vendored, tracked in `dist/libs/`):

| Library | Version | Used by |
|---------|---------|---------|
| EJML core/ddense | 0.43.1 | `numja.core.NDArray` wraps `DMatrixRMaj`; linear algebra |
| commons-math3 | 3.6.1 | statistics/special functions |
| jfreechart | 1.5.3 | plotting backend for matplotlib/seaborn modules |

Tooling (git-ignored): ProGuard 7.3.2 (`libs/proguard-7.3.2/`).

## Artifacts Produced

```
dist/
├── numja.jar        # NDArray + linear algebra core
├── pandas.jar       # DataFrame/Series/CSV
├── matplotlib.jar   # 2D plotting, imshow
├── seaborn.jar      # statistical visualizations
├── sklearn.jar      # full ML suite (~50 classes)
├── libs/            # third-party runtime jars
└── datasets/        # 11 bundled CSV datasets (iris, titanic, ...)
```

Intermediate outputs (git-ignored): `build/classes/`, `out/classes/`, per-module `modules/*/target/{classes,test-classes}`.

## Configuration

- `.vscode/tasks.json` — IDE build tasks
- No dependency manager config file exists in the repo (pom.xml is git-ignored and absent)
- `.planning/config.json` — GSD workflow config (this project)

## Constraints

- Closed-source paradigm: `modules/`, `libs/`, `pom.xml`, `proguard.pro`, `.github/` are all in `.gitignore`
- Consumers get only `dist/` binaries — zero-install offline execution model
