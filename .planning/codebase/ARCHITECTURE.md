# Architecture

> Last updated: 2026-08-25

## Pattern

**Modular layered library** — five decoupled, interoperable JAR modules mirroring the Python data-science stack. No server, no UI; consumers link the jars from `dist/`.

```
            ┌─────────────────────────────────────────┐
            │  Consumer code (examples/, user apps)   │
            └──────┬────────┬────────┬───────┬────────┘
                   │        │        │       │
              sklearn    pandas  matplotlib seaborn
                   │        │        │       │
                   └───┬────┴───┬────┴───────┘
                       ▼        ▼
                    numja (NDArray + LinAlg core)
                       │
              EJML / commons-math3 / JFreeChart
```

Dependency direction: everything depends downward on `numja`; nothing depends upward. `sklearn` additionally consumes `pandas` DataFrames.

## Layers

1. **Core numeric layer** — `numja`: NDArray container (wraps EJML `DMatrixRMaj`), elementwise ops (`ArrayOps`), decompositions (`LinAlg`, `SVD`, `QR`, `Eigen`, `Lstsq`), thread pool config.
2. **Data layer** — `pandas`: CSV/JSON loaders, labeled DataFrame/Series, `.loc`/`.iloc` indexers, stats/groupby.
3. **Presentation layer** — `matplotlib` (line/scatter plots, pixel-level imshow), `seaborn` (statistical themes, heatmap).
4. **ML layer** — `sklearn`: ~50 classes across supervised classifiers/regressors, preprocessing, model selection (train_test_split/GridSearchCV/CrossValidation/Pipeline), clustering, decomposition, neural networks, metrics, datasets.

## Key Abstractions

- `numja.core.NDArray` — central tensor type; 1D vector + 2D matrix representations, reshape/transpose/vectorized ops
- `numja.NumJa` — static factory entry point (`zeros`, `ones`, `linspace`, `array`, `mean`, `std`) used via `import static numja.NumJa.*`
- `pandas.DataFrame` / `pandas.Series` — labeled data containers with separate indexer objects (`DataFrameLoc`, `DataFrameILoc`)
- `sklearn.pipeline.Pipeline` — sequential transform chain (Imputer → Scaler → Model)
- `sklearn.utils.ParallelUtils` — shared multi-threaded executor (~60% default core allocation)

## Entry Points

Library API only — no main application. Example drivers in `examples/`:
- `examples/core/TestCore.java`, `examples/pandas/TestPandas.java`, `examples/plot/*.java` — per-module smoke tests via `main()`
- `examples/TestComprehensive.java` (686 lines) — cross-module integration exercise
- `examples/NumJaClientExample.java` — downstream-consumer style example

Run via [scripts/run_example.ps1](../../scripts/run_example.ps1) / `scripts/nj.ps1`.

## Data Flow

CSV file → `Pandas.read_csv` → `DataFrame` → slicing/stats or conversion to `NDArray` → `sklearn` fit/predict → results plotted via `matplotlib`/`seaborn` → PNG/console output.

## Distribution & Protection Model

Source in `modules/` is git-ignored (closed-source). Release pipeline: compile (`--release 11`) → ProGuard obfuscation (`scripts/build/build_obfuscated.ps1`, ProGuard 7.3.2) → obfuscated jars committed to `dist/` so clones run without building.

Full class inventory: [MODULES_INVENTORY.md](../../MODULES_INVENTORY.md).
