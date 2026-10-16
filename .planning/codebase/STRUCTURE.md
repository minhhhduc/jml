# Directory Structure

> Last updated: 2026-08-25

## Layout

```
java_ml/
├── .gitignore              # ignores modules/, libs/, build outputs (closed-source model)
├── README.md               # API-reference-style docs, math formulations
├── MODULES_INVENTORY.md    # full class inventory + build status
├── QUICK_REFERENCE.md
├── CHANGELOG.md
├── dist/                   # TRACKED release artifacts
│   ├── numja.jar, pandas.jar, matplotlib.jar, seaborn.jar, sklearn.jar
│   ├── libs/               # EJML, commons-math3, jfreechart jars
│   └── datasets/           # 11 CSV datasets (iris, titanic, wine, ...)
├── modules/                # GIT-IGNORED source code
│   ├── numja/src/main/java/numja/{core,linalg,config}/
│   ├── pandas/src/main/java/pandas/
│   ├── matplotlib/src/main/java/matplotlib/
│   ├── seaborn/src/main/java/seaborn/
│   ├── sklearn/src/main/java/sklearn/{cluster,datasets,decomposition,ensemble,
│   │      impute,linear_model,metrics,model_selection,naive_bayes,neighbors,
│   │      neural_network,pipeline,preprocessing,svm,tree,utils}/
│   └── */target/           # per-module compiled classes
├── examples/               # TRACKED example drivers (main()-based smoke tests)
│   ├── core/, pandas/, plot/, TestComprehensive.java, NumJaClientExample.java
├── scripts/                # TRACKED build/run tooling
│   ├── build_core.ps1, build_check.ps1, install_core.ps1, nj.ps1, run_example.ps1
│   ├── build/*.bat + build_obfuscated.ps1   # per-module + ProGuard builds
│   └── install/*.bat                        # per-module jar installs
├── libs/                   # GIT-IGNORED tooling (proguard-7.3.2)
├── docs/                   # TRACKED guides (GETTING_STARTED, BUILD_AND_SETUP,
│                           #   HUONG_DAN_SU_DUNG [VN], INSTALL_JAVA_NO_MAVEN,
│                           #   QUICK_START_OFFLINE, PROJECT_STRUCTURE, JAVA_README)
├── build/, out/            # git-ignored compile output (build/classes, sources.txt)
└── .planning/              # GSD planning docs (this directory tree)
```

## Key Locations

| Task | Location |
|------|----------|
| Core array/linalg source | `modules/numja/src/main/java/numja/core/NDArray.java` |
| Linear algebra | `modules/numja/src/main/java/numja/linalg/LinAlg.java` |
| DataFrame | `modules/pandas/src/main/java/pandas/DataFrame.java` |
| ML models | `modules/sklearn/src/main/java/sklearn/**` |
| Build entry point | `scripts/build_core.ps1` |
| Obfuscated release | `scripts/build/build_obfuscated.ps1` → `dist/*.jar` |
| Datasets for tests/examples | `dist/datasets/*.csv` |

## Naming Conventions

- Java packages: lowercase module names (`numja`, `pandas`, `sklearn`) mirroring Python package names; sklearn subpackages use Python-style snake_case (`linear_model`, `naive_bayes`)
- Classes named after their Python equivalents (`StandardScaler`, `GridSearchCV`, `MLPClassifier`, `KMeans`)
- Example classes prefixed `Test*` (they are demo drivers, not unit tests)

## Notes

- `modules/*/src` was not found on disk during mapping — only `modules/*/target/classes` exist locally. The `.gitignore` excludes `modules/` entirely, so source availability depends on the local machine. Verify before planning source-level work.
- No Maven/Gradle files tracked (`pom.xml` ignored).
