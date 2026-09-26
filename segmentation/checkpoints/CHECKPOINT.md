# Trained Checkpoint

**File:** `segmentation/checkpoints/deeplabv3_lv_segmentation.pth`
**Size:** 168,333,067 bytes (168 MB)
**sha256:** `7f7bdc9f641dfc8d006d88371f156082ed6daac9cc83f8972ea465eb3232dc4c`

## Why it is not in git

GitHub rejects files over 100 MB. This 168 MB checkpoint is therefore **gitignored**
(`.gitignore`: `segmentation/checkpoints/*.pth`) and reproduced locally instead.
The audited commit `666a19d` shared it via a Google Drive link in the same manner.

## How to obtain / reproduce it

```bash
cd segmentation

# 1. small dev dataset (64/16/16 videos) — enough to reproduce below
python make_synthetic_dataset.py --mode demo --out /tmp/echo_demo

# 2. train from scratch (CPU: ~35 s/epoch, 6 epochs)
python train_segmentation.py \
    --videos-dir /tmp/echo_demo/Videos \
    --file-list /tmp/echo_demo/FileList.csv \
    --volume-tracings /tmp/echo_demo/VolumeTracings.csv \
    --epochs 6 --batch-size 8 --lr 1e-4 \
    --checkpoint-out checkpoints/deeplabv3_lv_segmentation.pth \
    --max-train-videos 64 --max-val-videos 16
```

Verify the bytes match the reference above, or just trust `load_segmentation_model()`.

## Loading contract

```python
from segmentation_module import load_segmentation_model, segment_video
model = load_segmentation_model("checkpoints/deeplabv3_lv_segmentation.pth")  # eval mode
masks  = segment_video("some_video.avi", model)   # (T,H,W) uint8, {0,1}, frame order preserved
```

Tested against this checkpoint by `test_segmentation_pipeline.py` (set
`SEG_CKPT=segmentation/checkpoints/deeplabv3_lv_segmentation.pth`).

## Reported quality (see evaluation_notes.md)

- Trained on synthetic EchoNet-schema demo set; **not** real echo.
- Best val Dice 0.8746 (epoch 6). Full (synthetic) TEST split: Dice 0.8618 / IoU 0.7607.
- Integrated EF (ef_engine/validate_ef.py) on 8 synthetic videos: MAE 1.96 %, R² 0.988.