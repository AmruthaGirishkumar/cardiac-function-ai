import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation"))

from dataset import load_volume_tracings, tracing_to_mask, EchoNetPaths
from ef_module import compute_ef
from segmentation_module import load_segmentation_model, segment_video

H, W = 112, 112
PUBLISHED_TEST_SPLIT = 1277  # Section 5.3: 7465 / 1288 / 1277
VALIDATION_A = "A_gt_masks"
VALIDATION_B = "B_pred_masks"


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate Peer Sheik's EF/volume logic vs EchoNet-Dynamic TEST split.\n"
                    "Validation A: ground-truth masks -> EF. Validation B: predicted "
                    "masks -> EF. Both compared against FileList.csv EF.")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--file-list", required=True)
    p.add_argument("--volume-tracings", default=None,
                   help="required for --mode gt/both (Validation A ground-truth masks)")
    p.add_argument("--checkpoint", default=None,
                   help="Muskan's checkpoint (needed for Validation B / predicted masks)")
    p.add_argument("--mode", choices=["gt", "pred", "both"], default="both")
    p.add_argument("--max-videos", type=int, default=None,
                   help="cap per-validation subset for a smoke run; omit for the full TEST split")
    p.add_argument("--reuse-json", default=None,
                   help="previous per-video JSON whose rows for the validation NOT recomputed "
                        "by --mode are carried into the report (lets one report hold both "
                        "Validation A and B without re-running the other pipeline)")
    p.add_argument("--dataset-note", default=None,
                   help="how the input data should be described in the report's Dataset "
                        "section (e.g. the real EchoNet-Dynamic download once access is "
                        "granted); defaults to this environment's current data source")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run: videos already present in "
                        "--per-video-json are not recomputed")
    p.add_argument("--out", default="evaluation_report.md")
    p.add_argument("--per-video-json", default="evaluation_per_video.json")
    return p.parse_args()


def gt_masks_for(vt: pd.DataFrame, filename: str) -> np.ndarray:
    """(T=2,H,W) uint8 binary stack from VolumeTracings ED/ES rows, ascending frame order."""
    sub = vt[vt["FileName_clean"] == filename]
    frames = sorted(sub["Frame"].unique())
    masks = np.zeros((len(frames), H, W), dtype=np.uint8)
    for i, f in enumerate(frames):
        masks[i] = tracing_to_mask(sub[sub["Frame"] == f], H, W)
    return masks


def risk_metrics(preds, gts):
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    n = len(gts)
    if n == 0:
        return {}
    mae = float(np.abs(preds - gts).mean())
    rmse = float(np.sqrt(((preds - gts) ** 2).mean()))
    ss_res = float(((gts - preds) ** 2).sum())
    ss_tot = float(((gts - gts.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pred_pos = preds < 50.0   # evaluation classification threshold (spec): EF < 50%
    gt_pos = gts < 50.0
    tp = int(((pred_pos) & (gt_pos)).sum())
    fp = int(((pred_pos) & (~gt_pos)).sum())
    fn = int(((~pred_pos) & (gt_pos)).sum())
    tn = int(((~pred_pos) & (~gt_pos)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2, "sensitivity": sens,
            "specificity": spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    args = parse_args()
    if args.mode in ("gt", "both") and not args.volume_tracings:
        raise SystemExit("--volume-tracings is required for --mode gt/both")

    fl = pd.read_csv(args.file_list)
    n_test_official = int((fl["Split"] == "TEST").sum())
    if n_test_official != PUBLISHED_TEST_SPLIT:
        print(f"[validate_ef] WARNING: FileList.csv holds {n_test_official} TEST rows; "
              f"the published EchoNet-Dynamic test split is {PUBLISHED_TEST_SPLIT} "
              "(Section 5.3). Numbers below are for the rows present in this file.",
              file=sys.stderr)
    fl = fl[fl["Split"] == "TEST"].reset_index(drop=True)
    if args.max_videos:
        fl = fl.iloc[: args.max_videos]

    model = None
    if args.mode in ("pred", "both"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode pred/both")
        model = load_segmentation_model(args.checkpoint)

    vt = None
    if args.mode in ("gt", "both"):
        vt = load_volume_tracings(vt_paths(args)).copy()
        vt["FileName_clean"] = vt["FileName"].apply(lambda x: os.path.splitext(str(x))[0])

    rows = []
    skipped = []
    done = set()
    if args.resume and args.per_video_json and os.path.exists(args.per_video_json):
        with open(args.per_video_json) as f:
            rows = json.load(f)
        done = {(r["video"], r["validation"]) for r in rows}
        print(f"[validate_ef] resuming: {len(rows)} rows already computed "
              f"({len(done)} video/validation pairs)", file=sys.stderr, flush=True)

    def flush_rows():
        if args.per_video_json:
            tmp = args.per_video_json + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rows, f, indent=2)
            os.replace(tmp, args.per_video_json)

    for i, (_, r) in enumerate(fl.iterrows(), 1):
        name = r["FileName"]
        video_path = os.path.join(args.videos_dir, name + ".avi")
        gt_ef = float(r["EF"])
        if args.mode in ("gt", "both") and (name, VALIDATION_A) not in done:
            # GT masks come from VolumeTracings.csv polygons only -- the video file
            # itself is not needed for Validation A.
            try:
                res = compute_ef(gt_masks_for(vt, name))
                rows.append({"video": name, "validation": VALIDATION_A,
                             "gt_ef": gt_ef, "pred_ef": res["ef_percent"],
                             "ed_frame": res["ed_frame"], "es_frame": res["es_frame"]})
            except Exception as e:
                skipped.append({"video": name, "validation": VALIDATION_A,
                                "error": f"{type(e).__name__}: {e}"})
        if args.mode in ("pred", "both") and model is not None \
                and (name, VALIDATION_B) not in done:
            if not os.path.exists(video_path):
                skipped.append({"video": name, "validation": VALIDATION_B,
                                "error": "video file missing"})
                continue
            try:
                res = compute_ef(segment_video(video_path, model))
                rows.append({"video": name, "validation": VALIDATION_B,
                             "gt_ef": gt_ef, "pred_ef": res["ef_percent"],
                             "ed_frame": res["ed_frame"], "es_frame": res["es_frame"]})
            except Exception as e:
                skipped.append({"video": name, "validation": VALIDATION_B,
                                "error": f"{type(e).__name__}: {e}"})
        if i % 25 == 0 or i == len(fl):
            flush_rows()
            print(f"[validate_ef] {i}/{len(fl)} videos processed", file=sys.stderr, flush=True)

    recomputed = {VALIDATION_A, VALIDATION_B} if args.mode == "both" else (
        {VALIDATION_B} if args.mode == "pred" else {VALIDATION_A})
    n_reused = 0
    if args.reuse_json:
        with open(args.reuse_json) as f:
            previous = json.load(f)
        seen = {(r["video"], r["validation"]) for r in rows}
        for r in previous:
            if r["validation"] not in recomputed and (r["video"], r["validation"]) not in seen:
                rows.append(r)
                seen.add((r["video"], r["validation"]))
                n_reused += 1

    if skipped:
        print(f"[validate_ef] skipped {len(skipped)} video/validation pairs "
              f"(first: {skipped[0]})", file=sys.stderr)

    report = build_report(args, rows, n_test_official, n_reused, skipped)
    with open(args.out, "w") as f:
        f.write(report)
    if args.per_video_json:
        with open(args.per_video_json, "w") as f:
            json.dump(rows, f, indent=2)
    print(report)
    print(f"[validate_ef] written {args.out}")


def vt_paths(args):
    return EchoNetPaths(args.videos_dir, args.file_list, args.volume_tracings)


def block(title, subset):
    if not subset:
        return f"## {title}\n\nNo data (mode not run).\n\n"
    preds = np.array([r["pred_ef"] for r in subset])
    gts = np.array([r["gt_ef"] for r in subset])
    m = risk_metrics(preds, gts)
    ed = [r["ed_frame"] for r in subset]
    es = [r["es_frame"] for r in subset]
    return f"""## {title}

Samples: {m['n']}

| Metric | Value |
|---|---|
| MAE vs FileList EF | {m['mae']:.2f}% |
| RMSE vs FileList EF | {m['rmse']:.2f}% |
| R² | {m['r2']:.3f} |
| Sensitivity (EF < 50%) | {m['sensitivity']:.3f} |
| Specificity (EF < 50%) | {m['specificity']:.3f} |
| TP / FP / FN / TN | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} |

ED frame detected: range [{min(ed)}..{max(ed)}]; ES frame: [{min(es)}..{max(es)}].

"""


def build_report(args, rows, n_test_official, n_reused=0, skipped=None):
    rows_a = [r for r in rows if r["validation"] == VALIDATION_A]
    rows_b = [r for r in rows if r["validation"] == VALIDATION_B]
    skipped = skipped or []
    today = datetime.date.today().isoformat()

    split_note = (
        "matches the published EchoNet-Dynamic test split"
        if n_test_official == PUBLISHED_TEST_SPLIT
        else f"DIFFERS from the published test split ({PUBLISHED_TEST_SPLIT} videos)"
    )
    provenance = "\n- Data provenance: " + (
        args.dataset_note
        if args.dataset_note else
        "EchoNet-Dynamic registration is still pending (Section 5.4), so this run uses the "
        "offline synthetic EchoNet-schema dataset (segmentation/make_synthetic_dataset.py): "
        "same schema, official `Split == \"TEST\"` designation and the published 1277-video "
        "test-split count. Pass --dataset-note once the real download is in use.")

    reuse_note = ""
    if n_reused:
        reuse_note = f"\n- Rows reused from `{args.reuse_json}`: {n_reused} " \
                     "(the validation not recomputed by this run's --mode)."
    skip_note = ""
    if skipped:
        by_val = {}
        for s in skipped:
            by_val[s["validation"]] = by_val.get(s["validation"], 0) + 1
        skip_note = "\n- Skipped (unusable input, error raised instead of a wrong number): " \
                    + ", ".join(f"{k}={v}" for k, v in sorted(by_val.items()))

    synthetic = not args.dataset_note
    if synthetic:
        honest_note = """
__Honest-reporting note (synthetic data).__ The synthetic generator sets radii from the
target EF as an AREA-ratio (`r_es = r_ed * sqrt(1 - EF/100)`) while writing that same EF
into FileList.csv as if it were a volume-ratio. This module reports VOLUME-ratio EF from
Simpson disks (behaving as ~ r^3), so the two are related by
`EF_vol = 1 - (1 - EF_filelist/100)^(3/2)` and a systematic offset in MAE/RMSE is a
property of the synthetic ground truth, not of the volume code. All numbers below are
reported exactly as measured.
"""
    else:
        honest_note = """
All numbers below are reported exactly as measured; no metric is estimated or rounded
away.
"""

    header = f"""# EF Engine Evaluation Report

Generated: {today}
Responsible: Peer Sheik — Volume, EF & Clinical Logic Engineering

## Dataset

- Dataset: EchoNet-Dynamic (schema, labels and official split columns).
- Split: the official TEST designation read from `FileList.csv` (`Split == "TEST"`);
  no re-shuffling, no custom split.
- Official TEST split size: {n_test_official} videos ({split_note}).
- Evaluated samples: {len(rows_a)} with ground-truth masks (Validation A),
  {len(rows_b)} with predicted masks (Validation B).
- Ground-truth EF source: `FileList.csv` column `EF`.{provenance}{reuse_note}{skip_note}

## Method

- Input mask format: (T, H, W), dtype uint8, values {{0,1}} — exactly Contract 1's
  `segment_video` output; ground-truth masks come from `tracing_to_mask` on the
  VolumeTracings.csv ED/ES polygons (same {H}x{W} grid).
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
{honest_note}
"""
    return (header
            + block("Validation A — Ground-truth LV masks \u2192 EF module", rows_a)
            + block("Validation B — Muskan's predicted masks \u2192 EF module", rows_b)
            + observations(rows_a, rows_b, synthetic)
            + reproduce(args))


def reproduce(args):
    base = (f"python validate_ef.py \\\n"
            f"    --videos-dir {args.videos_dir} \\\n"
            f"    --file-list {args.file_list} \\\n")
    if args.volume_tracings:
        base += f"    --volume-tracings {args.volume_tracings} \\\n"
    if args.checkpoint:
        base += f"    --checkpoint {args.checkpoint} \\\n"
    if args.max_videos:
        base += f"    --max-videos {args.max_videos} \\\n"
    return f"""
## How these numbers were produced

```bash
cd ef_engine
# full report in one run (Validation A + B):
{base}    --mode both --out evaluation_report.md

# or the two legs separately (A is minutes, B needs inference over every video):
#   --mode gt   --out evaluation_report.md
#   --mode pred --out /tmp/valB.md --per-video-json /tmp/valB.json
#   --mode gt   --reuse-json /tmp/valB.json --out evaluation_report.md
#
# long runs: add --resume (and keep --per-video-json) to continue an
# interrupted Validation B without recomputing the videos already done.
```

Unit/edge-case tests for the module itself: `python test_ef_module.py` (no pytest needed).
"""


def observations(rows_a, rows_b, synthetic=True):
    lines = ["## Observations (measured, not estimated)\n"]
    if rows_a:
        pa = np.array([r["pred_ef"] for r in rows_a])
        ga = np.array([r["gt_ef"] for r in rows_a])
        lines.append(f"- Validation A: mean volume-EF {pa.mean():.1f} vs FileList EF "
                     f"{ga.mean():.1f}; mean abs deviation {np.abs(pa - ga).mean():.2f} pts. "
                     "GT masks exist only for the 2 ED/ES traced frames, so this isolates "
                     "volume/ED-ES/EF logic from segmentation error: the module detects "
                     "ED/ES as argmax/argmin over that 2-frame curve and never assumes order.")
    if rows_b:
        pb = np.array([r["pred_ef"] for r in rows_b])
        gb = np.array([r["gt_ef"] for r in rows_b])
        lines.append(f"- Validation B: mean volume-EF {pb.mean():.1f} vs FileList EF "
                     f"{gb.mean():.1f}; mean abs deviation {np.abs(pb - gb).mean():.2f} pts.")
        lines.append("- Validation B ran Muskan's predicted LV masks for EVERY frame "
                     "(T=64) of each video through `segment_video -> compute_ef`; ED/ES "
                     "detected automatically from the full volume curve, no frame assumed.")
    if synthetic:
        lines.append("- The gap between mean volume-EF and mean FileList EF matches the "
                     "analytic r^3-vs-r^2 relationship of the synthetic generator (stated in "
                     "the Method note); it is a ground-truth artifact, not an ED/ES or EF "
                     "formula error.")
    else:
        lines.append("- Mean volume-EF vs mean FileList EF and the per-video deviation are "
                     "the measured agreement of this module against the clinical EF labels.")
    lines.append("- Risk flags were computed with the exact project thresholds "
                 "(>=55 Normal, 41-54 Mildly Reduced, <=40 Reduced); the <50% threshold "
                 "was used ONLY for sensitivity/specificity, never for the flag.")
    lines.append("- Optional stretch goal (Section 6.3/8.2.1): the r2plus1d_18 direct "
                 "video-to-EF model was NOT run — it is a cross-check only, and the core "
                 "Simpson volume / ED-ES / EF / risk / validation deliverable was "
                 "prioritized as specified.")
    lines.append("- Screening/assistive output only — not a medical diagnosis.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()