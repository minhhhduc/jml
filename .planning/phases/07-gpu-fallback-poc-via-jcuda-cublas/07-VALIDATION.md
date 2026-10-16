---
phase: 7
slug: gpu-fallback-poc-via-jcuda-cublas
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-12
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for the isolated JCuda/cuBLAS GPU fallback POC.

## Test Infrastructure

| Property | Value |
|---|---|
| **Current-session validation** | Python static source/JSON assertions plus `git diff --check` |
| **Deferred runtime validation** | Maven/JAR and Colab T4 runs, documented in `07-RESULTS.md` |
| **Config file** | Root `pom.xml` |
| **Runtime commands** | Deferred only; not run in this code-only session |

## Sampling Rate

- **After every source task:** Run static checks for the changed source, POM, notebook JSON, and absence of forbidden legacy/credential material.
- **After every plan wave:** Run `git diff --check` and the Phase 7 static contract checks.
- **Before a future runtime decision:** First package successfully on Linux; only then run the JAR and retain the Colab T4 evidence.
- **Current-session scope:** Do not run Maven, resolve dependencies, create or execute a JAR, or run Colab.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Current-session Check | Status |
|---|---|---|---|---|---|---|---|
| 07-01-01 | 01 | 1 | GPU-04 | T-07-01 | Isolated pinned Maven declarations; production modules remain free of `org.jcuda` | Static POM/module assertions | ✅ green |
| 07-01-02 | 01 | 1 | GPU-04 | T-07-02 | Authored no-device and native-init paths emit complete NO-GO schema | Static `GemmBench` assertions | ✅ green |
| 07-02-01 | 02 | 2 | GPU-04 | T-07-01/T-07-03 | Authored Colab preflight checks T4/CUDA and requires zero Java exit plus exactly one `result=OK` | Notebook JSON/text assertions | ✅ green |
| 07-02-02 | 02 | 2 | GPU-04 | T-07-02 | Evidence template records future Frobenius, speedup, transfer, device, environment, and conclusion | Template text assertions | ✅ green |

*Runtime evidence remains pending until a future authorized Linux/Colab run.*

## Wave 0 Requirements

- Existing Maven/JUnit infrastructure covers code validation.
- The bench module and Colab notebook are Phase 7 deliverables, not prerequisite test infrastructure.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|---|---|---|---|
| T4 GPU numerical and performance gates | GPU-04 | A future authorized Colab run needs T4 hardware | Run the notebook after a successful Linux package; require zero Java exit, exactly one `result=OK`, then retain complete key=value output in `07-RESULTS.md`. |
| Production API/dependency freeze | GPU-04 | Scope assertion across repo | Statically confirm `NumJa.java` remains at 61 public static methods, `ArrayOps.java` remains at 34, and `modules/` contains no `org.jcuda`. |

## Validation Sign-Off

- [x] Every implementation task has an automated or manual verification path.
- [x] Sampling continuity is defined for each task and wave.
- [x] No framework setup is required.
- [x] No watch-mode commands are used.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** source-only delivery complete; future runtime evidence pending
