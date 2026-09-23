# Segmentation Track (Muskan) — Track A

Turns a raw echocardiogram video into a per-frame left-ventricle (LV) binary mask.
This is **Contract 1** of the project's interface spec (Section 8.4.1) — the only
thing Peer Shaik's `ef_module.py` and Amrutha's backend need from this folder.

## Files

| File | Section 8.1.1 step it satisfies |
|---|---|
| `inspect_dataset.py` | Step 2 — inspect FileList.csv / VolumeTracings.csv, confirm split BEFORE anything else |
| `segmentation_module.py` | **The deliverable.** `load_segmentation_model()`, `extract_frames()`, `segment_video()` — exactly Contract 1 |
| `dataset.py` | Steps 3 & 4 — frame-extraction (training-side) + mask-construction from VolumeTracings.csv |
| `losses.py` | Step 5 — BCE + Dice loss used in training |
| `train_segmentation.py` | Steps 5 & 6 — fine-tunes DeepLabV3, applies training-only augmentation |
| `sanity_check.py` | Step 7 — full-video inference + visual/quantitative check for mask jumps or disappearance |
| `evaluate.py` | Step 8 — Dice/IoU on TEST split, writes `evaluation_notes.md` |
| `requirements.txt`, `README.md` | Step 9 — packaging for handoff |
| `checkpoints/` | Trained model weights (`.pth`) — keep out of git if large (see `.gitignore` note below) |

## Full Workflow (in spec order)

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Request EchoNet-Dynamic access (MANUAL — do this first, it has a waiting period)
#    -> https://echonet.github.io/dynamic/  (free academic-use registration)
#    Do not proceed past this step until you have the download link and have
#    extracted Videos/, FileList.csv, VolumeTracings.csv locally.

# 2. Inspect the dataset BEFORE doing anything else
python inspect_dataset.py \
    --file-list /path/to/FileList.csv \
    --volume-tracings /path/to/VolumeTracings.csv
#    Confirms split = 7,465 / 1,288 / 1,277 (TRAIN/VAL/TEST) per Section 5.3.
#    Fix/re-download before continuing if this reports a MISMATCH on the full dataset.

# 3. Train (frame extraction + mask construction + fine-tuning + augmentation
#    all happen inside this one command)
python train_segmentation.py \
    --videos-dir /path/to/Videos \
    --file-list /path/to/FileList.csv \
    --volume-tracings /path/to/VolumeTracings.csv \
    --epochs 15 --batch-size 8 --lr 1e-4
#    Saves the best checkpoint to checkpoints/deeplabv3_lv_segmentation.pth

# 4. Sanity-check full-video inference on a few sample videos
python sanity_check.py \
    --checkpoint checkpoints/deeplabv3_lv_segmentation.pth \
    --videos sample1.avi sample2.avi sample3.avi \
    --out-dir sanity_check_outputs/
#    Watch the generated overlay .mp4 files. Investigate/retrain if frames
#    are flagged as "jump" or "disappeared".

# 5. Evaluate on the TEST split
python evaluate.py \
    --videos-dir /path/to/Videos \
    --file-list /path/to/FileList.csv \
    --volume-tracings /path/to/VolumeTracings.csv \
    --checkpoint checkpoints/deeplabv3_lv_segmentation.pth
#    Writes Dice/IoU numbers to evaluation_notes.md

```

