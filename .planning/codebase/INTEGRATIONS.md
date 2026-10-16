# External Integrations

> Last updated: 2026-08-25

## Summary

**None.** NumJa is a fully offline, self-contained library. No network calls, no external APIs, no databases, no auth providers, no webhooks.

## What exists (in-process only)

| Integration | Type | Detail |
|-------------|------|--------|
| EJML 0.43.1 | In-JVM library | Dense matrix engine behind `numja.core.NDArray` |
| commons-math3 3.6.1 | In-JVM library | Statistical routines |
| JFreeChart 1.5.3 | In-JVM library | Chart rendering backend for `matplotlib`/`seaborn` modules |
| CSV datasets | Local files | `dist/datasets/*.csv` loaded by `pandas.Pandas.read_csv` and `seaborn.Seaborn.load_dataset` |

## Explicitly absent

- HTTP clients / REST consumers
- Databases / ORMs
- Message queues
- Cloud services / auth providers

This is a deliberate design goal: "zero-install, portable offline execution model" ([README.md](../../README.md)).
