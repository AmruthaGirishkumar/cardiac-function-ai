"""
sanity_check.py
-----------------
Section 8.1.1: "Run inference on full videos (all frames, not just ED/ES)
and visually sanity-check that the mask tracks the ventricle smoothly
across the cardiac cycle -- flag any video where the mask 'jumps' or
disappears."

This script does two things for each sample video:
  1. Quantitative flagging -- computes LV mask pixel-area per frame and
     flags frames where the area suddenly jumps (large frame-to-frame
     change) or collapses to near-zero (the mask "disappearing").
  2. Visual output -- writes an .mp4 with the predicted mask overlaid in
     translucent red on every frame, so you can actually watch it and
     confirm the boundary tracks the ventricle smoothly.

Run this AFTER training, before writing up Dice/IoU numbers -- a model
that scores fine on Dice but visibly flickers/jumps frame-to-frame is
worth catching here rather than at integration time.

Usage:
    python sanity_check.py \
        --checkpoint checkpoints/deeplabv3_lv_segmentation.pth \
        --videos sample1.avi sample2.avi sample3.avi \
        --out-dir sanity_check_outputs/
"""

import argparse
import os

import cv2
import numpy as np

from segmentation_module import load_segmentation_model, extract_frames, segment_video

# A frame-to-frame area change larger than this fraction of the video's mean
# mask area is flagged as a possible "jump". Tune if you see too many/few flags.
JUMP_THRESHOLD_FRACTION = 0.5
# A mask smaller than this many pixels (out of 112*112 = 12,544) is treated
# as "the mask disappeared" -- an all-background prediction.
DISAPPEAR_PIXEL_THRESHOLD = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Visual + quantitative sanity check for full-video LV segmentation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--videos", nargs="+", required=True, help="One or more sample video paths")
    parser.add_argument("--out-dir", default="sanity_check_outputs")
    return parser.parse_args()


def analyze_mask_curve(masks: np.ndarray, video_name: str):
    """Prints per-video flags for jumps and disappearing masks. Returns flagged frame indices."""
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)  # pixel count per frame
    mean_area = areas.mean()

    flagged = []
    for t in range(1, len(areas)):
        delta = abs(int(areas[t]) - int(areas[t - 1]))
        if mean_area > 0 and delta > JUMP_THRESHOLD_FRACTION * mean_area:
            flagged.append((t, "jump", int(areas[t - 1]), int(areas[t])))
        if areas[t] < DISAPPEAR_PIXEL_THRESHOLD:
            flagged.append((t, "disappeared", int(areas[t]), None))

    print(f"\n[{video_name}] {len(masks)} frames, mean mask area = {mean_area:.0f}px, "
          f"min = {areas.min()}px, max = {areas.max()}px")
    if flagged:
        print(f"  !! {len(flagged)} flagged frame(s):")
        for t, kind, before, after in flagged[:10]:
            if kind == "jump":
                print(f"     frame {t}: JUMP  area {before}px -> {after}px")
            else:
                print(f"     frame {t}: DISAPPEARED  area = {before}px")
        if len(flagged) > 10:
            print(f"     ... and {len(flagged) - 10} more")
    else:
        print("  OK -- no jumps or disappearing masks detected.")
    return flagged


def write_overlay_video(frames: np.ndarray, masks: np.ndarray, out_path: str, fps: float = 15.0):
    """Writes an .mp4 with the predicted mask overlaid in translucent red for visual review."""
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame_rgb, mask in zip(frames, masks):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
        overlay = frame_bgr.copy()
        overlay[mask == 1] = (0, 0, 255)  # red in BGR
        blended = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)
        writer.write(blended)

    writer.release()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_segmentation_model(args.checkpoint)

    all_flags = {}
    for video_path in args.videos:
        video_name = os.path.basename(video_path)
        frames = extract_frames(video_path)
        masks = segment_video(video_path, model)

        flagged = analyze_mask_curve(masks, video_name)
        all_flags[video_name] = flagged

        overlay_path = os.path.join(args.out_dir, f"overlay_{video_name}.mp4")
        write_overlay_video(frames, masks, overlay_path)
        print(f"  overlay saved -> {overlay_path}  (watch this to visually confirm smooth tracking)")

    print("\n" + "=" * 70)
    total_flags = sum(len(f) for f in all_flags.values())
    if total_flags == 0:
        print("RESULT: All sample videos passed the sanity check cleanly.")
    else:
        print(f"RESULT: {total_flags} total flagged frames across {len(args.videos)} video(s). "
              f"Watch the overlay .mp4 files for the flagged videos before finalizing the checkpoint.")
    print("=" * 70)


if __name__ == "__main__":
    main()
