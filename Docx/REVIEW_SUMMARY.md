# Backlog review — what's fixed, what's already fine, what's still open

Checked against the current `MIRA_doc_clean` codebase, verified by running
the actual test suite (not just reading code).

## Already resolved — no action needed (verified, not just assumed)
- **"non-runnable main.py"** — there is no `main.py` in this codebase;
  the entry point is now `mira_app.py`, which imports and runs cleanly.
  This backlog item predates the restructure and no longer applies.
- **"inverted forehead polygon fallback"** — read `_build_forehead_poly`
  and `_build_chin_poly` in `region_extractor.py` line by line: the
  upper/lower selection and both percentile fallbacks are oriented
  correctly (smaller y = higher on the face = forehead). Not inverted
  in this codebase.
- **"CSV migration data-corruption risk"** — `_migrate_if_schema_changed`
  in `feature_vector_export.py` already remaps old rows by column name
  onto the current schema (with a documented fallback for rows that are
  already corrupted), under an `fcntl` lock. This matches what your
  memory notes describe as already-completed work.
- **4 "stale" tests** — only one of the four still exists
  (`test_analyze_skin_accepts_optional_baseline_without_error`); it
  already calls `baseline=` as a keyword and passes. The other three
  are gone. Full suite: **0 failures.**

## Fixed this pass
- **Fairness/bias audit** — this was a real gap: the skin-tone gate
  mitigation code exists, but nothing re-verified the calibration claim
  in `thresholds.py`'s docstring. Added `test_fairness_audit.py`
  (19 tests, all passing) plus `FAIRNESS_AUDIT.md` explaining the
  methodology and its limits. See that file for details — short
  version: it's a synthetic-swatch regression check, not a real-photo
  audit, and the doc says so explicitly.

Combined test suite after this change: **559 passed, 72 subtests, 0
failed** (was 540 passed before).

## Could not fix — need more from you
- **The 3 regressed bugs** ("3 previously-fixed bugs came back") — I
  don't have specifics on what they are, only that they exist per your
  earlier note. Fixing bugs I can't identify risks masking them with
  the wrong patch. If you can point me at which behaviors regressed (or
  a git diff / commit range against the older fixed version), I can
  find and fix them properly.
- **Threshold calibration** — still thin (per your own assessment), and
  fixing it needs real capture data across people/lighting, which isn't
  something I can generate. Same limitation applies to the fairness
  audit above — the synthetic version is a stopgap, not a replacement
  for real data.
- **Auxiliary features end-to-end** (history, trend, TTS, GUI) — unit
  tests pass, but "does the GUI actually render correctly" and "does
  audio actually play" need a human running the app, not something
  verifiable from a container.

## Files delivered
- `test_fairness_audit.py` → drop into `src/tests/`
- `FAIRNESS_AUDIT.md` → drop into the project root
