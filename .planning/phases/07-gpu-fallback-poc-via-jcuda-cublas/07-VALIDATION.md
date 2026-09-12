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
| **Framework** | JUnit 4 (repo-wide); bench self-validates by output gates |
| **Config file** | Root `pom.xml` |
| **Quick run command** | `mvn -q -pl bench/jcuda-poc -am package -DskipTests` |
| **Full suite command** | `mvn test` |
| **Estimated runtime** | Build: ~1 min; full suite: project-dependent |

## Sampling Rate

- **After every task commit:** Build `bench/jcuda-poc`, then run its shaded JAR locally and confirm a complete `GPU_ABSENT`/`NO-GO` key=value result when no CUDA device is present.
- **After every plan wave:** Run `mvn test`.
- **Before `/gsd:verify-work`:** Run the full suite and retain a Colab T4 notebook result.
- **Max feedback latency:** ~2 minutes locally; Colab T4 is a manual phase gate.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---|---|---|---|---|---|---|---|---|---|
| 07-01-01 | 01 | 1 | GPU-04 | T-07-01 | Isolated Maven deps; production modules contain no `org.jcuda` dependency | smoke/static | `mvn -q -pl bench/jcuda-poc -am package -DskipTests` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | GPU-04 | T-07-02 | Local no-device run emits safe `GPU_ABSENT` + `NO-GO` result and never hides errors | smoke | `java -jar bench/jcuda-poc/target/jcuda-poc-jar.jar -Dbench.env=local` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | GPU-04 | T-07-01/T-07-03 | Colab preflight verifies T4 + CUDA 12.6 `libcublas.so.12`; Java must exit zero and the retained log must contain exactly one `result=` line, `result=OK` | integration/manual | Run `notebooks/colab-jcuda-poc.ipynb` on Colab GPU runtime | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | GPU-04 | T-07-02 | Results record Frobenius, speedup, transfer, device, environment, and explicit GO/NO-GO | manual | Copy notebook result into `07-RESULTS.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

- Existing Maven/JUnit infrastructure covers code validation.
- The bench module and Colab notebook are Phase 7 deliverables, not prerequisite test infrastructure.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|---|---|---|---|
| T4 GPU numerical and performance gates | GPU-04 | The local Windows machine has no NVIDIA GPU; Colab is the hardware target | Run the Colab notebook, require a zero Java exit and exactly one `result=OK` line, then retain key=value output in `07-RESULTS.md`. |
| Production API/dependency freeze | GPU-04 | Scope assertion across repo | Confirm `NumJa.java` stays at 61 public static methods, `ArrayOps.java` stays at 34, and `modules/` contains no `org.jcuda`. |

## Validation Sign-Off

- [x] Every implementation task has an automated or manual verification path.
- [x] Sampling continuity is defined for each task and wave.
- [x] No framework setup is required.
- [x] No watch-mode commands are used.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** pending
