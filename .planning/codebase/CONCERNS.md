# Concerns

> Last updated: 2026-08-25

## 1. Source code absent from working tree (CRITICAL for any source work)

`modules/*/src` does not exist on disk — only compiled `modules/*/target/classes`. `modules/` is fully git-ignored, so a fresh clone (or this machine, if sources were cleaned) has **no Java source to build from**, yet [scripts/build_core.ps1](../../scripts/build_core.ps1) requires it and will fail with "No sources found".

**Impact:** Any phase touching library code is blocked until sources are restored locally.
**Detection:** already detected during mapping — verify with the user before planning.

## 2. No automated tests

See [TESTING.md](TESTING.md). "100% tested" claim in MODULES_INVENTORY.md rests on manual example drivers. Every behavioral change ships without regression protection. Highest-leverage fix if quality work is planned.

## 3. Closed-source distribution vs tracked binaries

Obfuscated jars are **committed to git** (`dist/*.jar`, plus `confusion_matrix.png`). Binary-in-repo means merge conflicts on every release, repo bloat, and no diffability. Deliberate trade-off ("clone-and-run"), but worth flagging.

## 4. Stale/contradictory docs

- [docs/BUILD_AND_SETUP.md](../../docs/BUILD_AND_SETUP.md) describes a Python package (`pip install -e .`, `src/numja/tests.py`) that doesn't exist in the tree — leftover from an earlier Python iteration
- README claims "Java 11 through Java 25+" compatibility; build pins `--release 11`; upper bound unverified
- MODULES_INVENTORY.md claims "zero-dependency" while shipping EJML/commons-math3/jfreechart in `dist/libs/`

## 5. No TODO/FIXME markers

Grep across tracked files finds none — either code is clean or (more likely) invisible since source isn't tracked. Cannot assess in-code tech debt.

## 6. Build fragility

Hand-rolled PowerShell build: hardcoded module-name filters in jar selection ([scripts/build_core.ps1:31](../../scripts/build_core.ps1)), ASCII-encoded sources file, no dependency resolution, Windows-only (.bat/.ps1). Cross-platform contributors have no path.

## 7. Security surface

Low: offline library, no network/parsing of untrusted input beyond CSV loading. Obfuscation is IP protection, not security. No secrets handling anywhere.
