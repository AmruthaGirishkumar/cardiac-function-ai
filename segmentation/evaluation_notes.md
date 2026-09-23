# Segmentation Evaluation Notes

Generated: 2026-09-23 09:30
Checkpoint: `checkpoints\deeplabv3_lv_segmentation.pth`
Test set size: 40 ED/ES frames

| Metric | Value |
|---|---|
| Mean Dice coefficient | 0.6265 |
| Mean IoU | 0.4772 |

Evaluated against expert-traced ED/ES masks in `VolumeTracings.csv` (TEST split),
per Section 7.2 of the technical spec.
