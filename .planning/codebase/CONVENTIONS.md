# Conventions

> Last updated: 2026-08-25

## API Design Philosophy

**Python-equivalent surface.** Every public API mirrors the Python library it simulates — class names (`StandardScaler`, `GridSearchCV`, `KMeans`), method names, and parameter semantics follow scikit-learn/NumPy/Pandas conventions so Python users transfer directly. Recent commit history shows docs strictly maintained in "API reference style".

- Static factory entry: `import static numja.NumJa.*;` then `array(...)`, `zeros(...)`, `mean(...)` — NumPy-function style
- Fluent-ish containers: `NDArray`, `DataFrame`, `Series` with `.loc` / `.iloc` indexer sub-objects
- sklearn lifecycle: constructor → `.fit(X, y)` → `.predict(X)` / `.transform(X)`

## Code Style (inferred from examples + inventory)

- Java 11 language level (`--release 11`)
- Standard Java naming: PascalCase classes, camelCase methods, package-per-module
- snake_case **package segments only** where mirroring Python (`sklearn.linear_model`, `sklearn.naive_bayes`) — deliberate exception to Java conventions for parity
- Vietnamese comments/strings appear in example code (e.g., `examples/core/TestCore.java`: "Tạo ma trận", "Định thức") — bilingual codebase; user-facing docs include a full Vietnamese guide (`docs/HUONG_DAN_SU_DUNG.md`)

## Error Handling

Not systematically observable from tracked files (source is git-ignored). Examples show happy-path usage without try/catch. Treat error-handling conventions as unknown until source is inspected locally.

## Performance Patterns

- Multi-threading via shared pool: `numja.config.ThreadPoolConfig`, `sklearn.utils.ParallelUtils` (~60% default core allocation)
- Parallel implementations noted in inventory: GEMM matmul (cache-friendly), RandomForest, GridSearchCV (`n_jobs=-1`), KNN K-D searches
- Delegates heavy numerics to EJML rather than hand-rolling

## Documentation Style

- README = strict API reference with math formulations (LaTeX), no code examples (enforced per commit d478327)
- Separate practical guides under `docs/` (getting started, offline quick start, no-maven install)
- Emoji-decorated status checklists in MODULES_INVENTORY.md
