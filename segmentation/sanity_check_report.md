# Sanity Check Report — full-video LV tracking

Generated: 2026-09-24
Checkpoint: `checkpoints/deeplabv3_lv_segmentation.pth`
Video source: synthetic EchoNet-schema AVI files (64 frames, 112×112)

Requirement 8.1.1 / item 15: predicted masks must track the LV across the
entire cardiac cycle without major jumping or disappearing between frames.

Method (`sanity_check.py`):
- Runs the deployed model on **every frame** of each video (not just ED/ES).
- Flags a "jump" when the per-frame mask area changes by > 50 % of the video's
  mean mask area between consecutive frames.
- Flags "disappeared" when a frame's mask area drops below 15 px (≈ empty mask).

## Results

| video | frames | mean area px | min px | max px | flags |
|---|---|---|---|---|---|
| 0120D871C868.avi | 64 | 3081 | 2767 | 3538 | none |
| 01FE0A49E85A.avi | 64 | 1778 | 1218 | 2643 | none |

**RESULT: All sample videos passed the sanity check cleanly — no jumps,
no disappearing masks.** The area curves are smooth sinusoids over the cardiac
cycle, consistent with continuous LV tracking.

Overlay videos: `sanity_check_outputs/overlay_*.avi.mp4` (red = predicted LV).
Re-run with `python sanity_check.py --checkpoint checkpoints/deeplabv3_lv_segmentation.pth --videos <videos...> --out-dir sanity_check_outputs`.