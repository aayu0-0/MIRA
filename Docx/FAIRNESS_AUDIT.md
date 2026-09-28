# Fairness / Skin-Tone Bias Audit

## Status: automated synthetic audit added (this pass). Real-photo audit: not yet done.

## What was missing
`thresholds.py`'s `_SkinToneGate` docstring claimed the Cr/Cb band was
"calibrated against a light-to-dark skin-tone gradient ... then checked
against navy fabric, black hair, white fabric, red fabric, and denim."
That claim existed only as a comment — there was no code that re-ran it,
so a future threshold edit could silently reintroduce a skin-tone bias
with nothing to catch it.

## What was added
`src/tests/test_fairness_audit.py` — runs automatically as part of the
normal test suite (`pytest src/tests`). It:

1. Runs the pipeline's actual gate function (`_tone_gate_mask_flat`) —
   not a reimplementation — against six solid-colour swatches spanning a
   light-to-dark gradient (loosely modeled on Fitzpatrick I-VI).
2. Asserts every swatch in the gradient passes the gate (>95% of pixels
   kept).
3. Asserts the keep-rate spread across the gradient stays under 15
   percentage points, so a change that starts favoring one end of the
   tone range over the other fails the suite immediately.
4. Re-checks the five non-skin materials named in the docstring (navy
   fabric, black hair, white fabric, red fabric, denim) are still
   rejected.
5. Repeats 1–4 for `LIP_TONE_GATE`.

All 19 checks currently pass against the codebase as it stands.

## Limitations — read before treating this as "done"
- **The swatches are synthetic solid colors, not real photos.** They're
  a fast regression check for gross Cr/Cb bias, not a substitute for
  testing against real captured faces across diverse subjects, lighting,
  and cameras. A real audit needs actual reference images.
- The gradient reference values are illustrative approximations of
  Fitzpatrick I–VI, not a validated clinical skin-tone dataset.
- This audit only covers the tone-gate step (Phase 3 region masking). It
  does not audit whether downstream scoring (redness, texture, acne
  detection) performs equivalently across skin tones once pixels pass
  the gate — that needs real images with ground truth, which this
  codebase doesn't have.

## Recommended next step
Before relying on this for anything clinical-facing: collect a small
reference set of real face crops across a range of skin tones and
lighting conditions, and extend this test file with an image-backed
suite alongside the synthetic one (keep the synthetic gradient as the
fast sanity check; add the image-backed suite as the real audit).
