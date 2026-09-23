"""
train_segmentation.py
----------------------
Fine-tunes DeepLabV3 (ResNet-50) on EchoNet-Dynamic ED/ES frame-mask pairs.

This is NOT part of the Contract 1 deliverable (segmentation_module.py) --
per Section 8.1.1: "do not hand off training code or notebooks as the
deliverable; the deliverable is the clean, callable module." This script's
only job is to produce the checkpoint file that segmentation_module.py's
load_segmentation_model() then loads.

Usage:
    python train_segmentation.py \
        --videos-dir /path/to/EchoNet-Dynamic/Videos \
        --file-list /path/to/FileList.csv \
        --volume-tracings /path/to/VolumeTracings.csv \
        --epochs 15 --batch-size 8 --lr 1e-4
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import EchoNetPaths, EchoNetSegmentationDataset
from losses import BCEDiceLoss, dice_coefficient, iou_score
from segmentation_module import _build_model, DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepLabV3 for LV segmentation on EchoNet-Dynamic")
    parser.add_argument("--videos-dir", required=True, help="Path to EchoNet-Dynamic Videos/ folder")
    parser.add_argument("--file-list", required=True, help="Path to FileList.csv")
    parser.add_argument("--volume-tracings", required=True, help="Path to VolumeTracings.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, nargs=2, default=(112, 112))
    parser.add_argument("--checkpoint-out", default="checkpoints/deeplabv3_lv_segmentation.pth")
    parser.add_argument("--pretrained-checkpoint", default=None,
                         help="Optional: path to an existing EchoNet-pretrained segmentation "
                              "checkpoint to start from, instead of ImageNet-only weights.")
    parser.add_argument("--max-train-videos", type=int, default=None,
                         help="Cap on TRAIN videos used, for a quick smoke-test run. "
                              "Omit for the full official split.")
    parser.add_argument("--max-val-videos", type=int, default=None,
                         help="Cap on VAL videos used, for a quick smoke-test run.")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the loader. optimizer=None means eval mode (no backward pass)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for frames, masks in loader:
            frames, masks = frames.to(DEVICE), masks.to(DEVICE)

            logits = model(frames)["out"].squeeze(1)  # (B, H, W)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_dice += dice_coefficient(preds, masks)
                total_iou += iou_score(preds, masks)

            total_loss += loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    paths = EchoNetPaths(
        videos_dir=args.videos_dir,
        file_list_csv=args.file_list,
        volume_tracings_csv=args.volume_tracings,
    )

    train_ds = EchoNetSegmentationDataset(paths, split="TRAIN", resize=tuple(args.resize), augment=True,
                                           max_videos=args.max_train_videos)
    val_ds = EchoNetSegmentationDataset(paths, split="VAL", resize=tuple(args.resize), augment=False,
                                         max_videos=args.max_val_videos)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[train] {len(train_ds)} train ED/ES frames, {len(val_ds)} val ED/ES frames")

    model = _build_model().to(DEVICE)
    if args.pretrained_checkpoint:
        print(f"[train] loading pretrained checkpoint from {args.pretrained_checkpoint}")
        model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=DEVICE))

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), args.checkpoint_out)
            print(f"[train] new best val Dice {best_val_dice:.4f} -- saved to {args.checkpoint_out}")

    print(f"[train] done. Best val Dice: {best_val_dice:.4f}. Checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
