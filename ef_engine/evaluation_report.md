# EF Engine Evaluation Report

Generated: 2026-09-25
Responsible: Peer Sheik — Volume, EF & Clinical Logic Engineering

## Dataset

- Dataset: EchoNet-Dynamic (schema, labels and official split columns).
- Split: the official TEST designation read from `FileList.csv` (`Split == "TEST"`);
  no re-shuffling, no custom split.
- Official TEST split size: 1277 videos (matches the published EchoNet-Dynamic test split).
- Evaluated samples: 1277 with ground-truth masks (Validation A),
  1277 with predicted masks (Validation B).
- Ground-truth EF source: `FileList.csv` column `EF`.
- Data provenance: EchoNet-Dynamic registration is still pending (Section 5.4), so this run uses the offline synthetic EchoNet-schema dataset (segmentation/make_synthetic_dataset.py): same schema, official `Split == "TEST"` designation and the published 1277-video test-split count. Pass --dataset-note once the real download is in use.
- Rows reused from `/tmp/valB.json`: 1277 (the validation not recomputed by this run's --mode).

## Method

- Input mask format: (T, H, W), dtype uint8, values {0,1} — exactly Contract 1's
  `segment_video` output; ground-truth masks come from `tracing_to_mask` on the
  VolumeTracings.csv ED/ES polygons (same 112x112 grid).
- Volume: Simpson's method of disks (Section 6.2), **n = 20** parallel disks along the
  LV long axis, `h = long-axis length / 20`, disk width L_i measured perpendicular to
  the long axis, `V = (pi/4) * sum_i (L_i^2 * h)` — i.e. each disk is the thin cylinder
  the specification describes (the printed formula `V ≈ (π/4) Σ(L_i × h)` in the tech
  doc omits the square on the diameter; without it the result is not a volume).
  Volumes are RELATIVE pixel^3 units — no calibration constant is invented; the same
  pixel grid applies to every frame, so EDV/ESV are mutually comparable and EF (a
  ratio) is scale-invariant.
- Complete per-frame volume curve, frame order preserved (original indices).
- ED frame = `argmax(volume_curve)`, ES frame = `argmin(volume_curve)` — always
  data-driven, never assumed.
- EDV = V(ED), ESV = V(ES); `EF = (EDV - ESV) / EDV * 100`, guarded against EDV = 0.
- Risk flag (display, Section 6.4): >=55 `Normal` | 41-54 `Mildly Reduced` | <=40 `Reduced`.
- Evaluation classification (sensitivity/specificity only, Section 7.2): **EF < 50%** —
  deliberately kept separate from the display risk categories above.

__Honest-reporting note (synthetic data).__ The synthetic generator sets radii from the
target EF as an AREA-ratio (`r_es = r_ed * sqrt(1 - EF/100)`) while writing that same EF
into FileList.csv as if it were a volume-ratio. This module reports VOLUME-ratio EF from
Simpson disks (behaving as ~ r^3), so the two are related by
`EF_vol = 1 - (1 - EF_filelist/100)^(3/2)` and a systematic offset in MAE/RMSE is a
property of the synthetic ground truth, not of the volume code. All numbers below are
reported exactly as measured.

## Validation A — Ground-truth LV masks → EF module

Samples: 1277

| Metric | Value |
|---|---|
| MAE vs FileList EF | 13.36% |
| RMSE vs FileList EF | 13.61% |
| R² | 0.096 |
| Sensitivity (EF < 50%) | 0.610 |
| Specificity (EF < 50%) | 1.000 |
| TP / FP / FN / TN | 552 / 0 / 353 / 372 |

ED frame detected: range [0..0]; ES frame: [1..1].

## Validation B — Muskan's predicted masks → EF module

Samples: 1277

| Metric | Value |
|---|---|
| MAE vs FileList EF | 14.31% |
| RMSE vs FileList EF | 14.58% |
| R² | -0.037 |
| Sensitivity (EF < 50%) | 0.588 |
| Specificity (EF < 50%) | 1.000 |
| TP / FP / FN / TN | 532 / 0 / 373 / 372 |

ED frame detected: range [0..63]; ES frame: [15..47].

## Observations (measured, not estimated)

- Validation A: mean volume-EF 53.1 vs FileList EF 39.8; mean abs deviation 13.36 pts. GT masks exist only for the 2 ED/ES traced frames, so this isolates volume/ED-ES/EF logic from segmentation error: the module detects ED/ES as argmax/argmin over that 2-frame curve and never assumes order.
- Validation B: mean volume-EF 54.1 vs FileList EF 39.8; mean abs deviation 14.31 pts.
- Validation B ran Muskan's predicted LV masks for EVERY frame (T=64) of each video through `segment_video -> compute_ef`; ED/ES detected automatically from the full volume curve, no frame assumed.
- The gap between mean volume-EF and mean FileList EF matches the analytic r^3-vs-r^2 relationship of the synthetic generator (stated in the Method note); it is a ground-truth artifact, not an ED/ES or EF formula error.
- Risk flags were computed with the exact project thresholds (>=55 Normal, 41-54 Mildly Reduced, <=40 Reduced); the <50% threshold was used ONLY for sensitivity/specificity, never for the flag.
- Optional stretch goal (Section 6.3/8.2.1): the r2plus1d_18 direct video-to-EF model was NOT run — it is a cross-check only, and the core Simpson volume / ED-ES / EF / risk / validation deliverable was prioritized as specified.
- Screening/assistive output only — not a medical diagnosis.

## How these numbers were produced

```bash
cd ef_engine
# full report in one run (Validation A + B):
python validate_ef.py \
    --videos-dir /tmp/echo_fulltest/Videos \
    --file-list /tmp/echo_fulltest/FileList.csv \
    --volume-tracings /tmp/echo_fulltest/VolumeTracings.csv \
    --mode both --out evaluation_report.md

# or the two legs separately (A is minutes, B needs inference over every video):
#   --mode gt   --out evaluation_report.md
#   --mode pred --out /tmp/valB.md --per-video-json /tmp/valB.json
#   --mode gt   --reuse-json /tmp/valB.json --out evaluation_report.md
#
# long runs: add --resume (and keep --per-video-json) to continue an
# interrupted Validation B without recomputing the videos already done.
```

Unit/edge-case tests for the module itself: `python test_ef_module.py` (no pytest needed).
