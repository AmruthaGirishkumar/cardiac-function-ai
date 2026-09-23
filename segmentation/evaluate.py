"""
evaluate.py
-----------
Computes Dice and IoU on the held-out TEST split against VolumeTracings.csv
ground truth (Section 7.2 / Section 8.1.3 Definition of Done).

Usage:
    python evaluate.py \
        --videos-dir /path/to/Videos \
        --file-list /path/to/FileList.csv \
        --volume-tracings /path/to/VolumeTracings.csv \
        --checkpoint checkpoints/deeplabv3_lv_segmentation.pth \
        --out evaluation_notes.md
"""

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import EchoNetPaths, EchoNetSegmentationDataset
from losses import dice_coefficient, iou_score
from segmentation_module import load_segmentation_model, DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LV segmentation model on EchoNet-Dynamic TEST split")
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--volume-tracings", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resize", type=int, nargs=2, default=(112, 112))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="evaluation_notes.md")
    parser.add_argument("--max-test-videos", type=int, default=None,
                         help="Cap on TEST videos used, for a quick smoke-test run. "
                              "Omit for the full official TEST split.")
    return parser.parse_args()


def main():
    args = parse_args()

    paths = EchoNetPaths(
        videos_dir=args.videos_dir,
        file_list_csv=args.file_list,
        volume_tracings_csv=args.volume_tracings,
    )
    test_ds = EchoNetSegmentationDataset(paths, split="TEST", resize=tuple(args.resize), augment=False,
                                          max_videos=args.max_test_videos)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_segmentation_model(args.checkpoint)

    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for frames, masks in test_loader:
            frames, masks = frames.to(DEVICE), masks.to(DEVICE)
            logits = model(frames)["out"].squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()

            for i in range(preds.shape[0]):
                dice_scores.append(dice_coefficient(preds[i], masks[i]))
                iou_scores.append(iou_score(preds[i], masks[i]))

    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)

    report = f"""# Segmentation Evaluation Notes

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Checkpoint: `{args.checkpoint}`
Test set size: {len(test_ds)} ED/ES frames

| Metric | Value |
|---|---|
| Mean Dice coefficient | {mean_dice:.4f} |
| Mean IoU | {mean_iou:.4f} |

Evaluated against expert-traced ED/ES masks in `VolumeTracings.csv` (TEST split),
per Section 7.2 of the technical spec.
"""
    with open(args.out, "w") as f:
        f.write(report)

    print(report)
    print(f"[evaluate] written to {args.out}")


if __name__ == "__main__":
    main()
