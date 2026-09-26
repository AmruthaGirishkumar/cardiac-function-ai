# Cardiac Function AI

End-to-end cardiac function assessment from echocardiogram videos:

```
ECG video ──► [Contract 1: segmentation_module.segment_video()] ──► (T,H,W) uint8 masks {0,1}
                │
                ▼
              [Contract 2: ef_module.compute_ef(masks)] ──► EF (%)
```

## Tracks

| Track | Owner | Deliverable | Status |
|---|---|---|---|
| A — Data & Segmentation | Muskan | `segmentation/segmentation_module.py` (Contract 1) | Pipeline verified end-to-end |
| B — EF Engine | Peer Shaik | `ef_engine/ef_module.py` (Contract 2) | Implemented, validated |
| C — App | (shared) | `app/` FastAPI backend | Scaffold |

## Contract 1 (Section 8.4.1) — Segmentation

```python
from segmentation_module import load_segmentation_model, segment_video

model = load_segmentation_model("checkpoints/deeplabv3_lv_segmentation.pth")  # only checkpoint path
masks = segment_video("video.avi", model)  # -> np.ndarray (T, H, W), uint8, values {0,1}, frame order preserved
```

See `segmentation/README.md` for the full Track A workflow, dataset setup and
evaluation, and `segmentation/evaluation_notes.md` for recorded Dice/IoU.

The committed checkpoint is **168 MB** (larger than GitHub's 100 MB file limit) so
it is gitignored at `segmentation/checkpoints/*.pth`. Reproduce it with:

```bash
cd segmentation
python make_synthetic_dataset.py --mode testfull --out /tmp/echo_testfull
python make_synthetic_dataset.py --mode demo --out /tmp/echo_demo               # small dev set
python train_segmentation.py --videos-dir /tmp/echo_demo/Videos \
    --file-list /tmp/echo_demo/FileList.csv --volume-tracings /tmp/echo_demo/VolumeTracings.csv \
    --epochs 6 --batch-size 8 --lr 1e-4 --checkpoint-out checkpoints/deeplabv3_lv_segmentation.pth \
    --max-train-videos 64 --max-val-videos 16
```

Fully reproducible verification (no pytest needed):

```bash
cd segmentation
SEG_CKPT=checkpoints/deeplabv3_lv_segmentation.pth python test_segmentation_pipeline.py
```

## Contract 2 — EF

```python
from ef_module import compute_ef

result = compute_ef(masks)   # masks from Contract 1
# {"edv": float, "esv": float, "ef_percent": float,
#  "ed_frame": int, "es_frame": int, "risk_flag": str}
```

See `ef_engine/validate_ef.py` for an end-to-end `segment_video → compute_ef`
validation written to `ef_engine/evaluation_report.md`.

## Quick end-to-end run (synthetic data)

```bash
cd ef_engine
python validate_ef.py --videos-dir /tmp/echo_demo/Videos \
    --file-list /tmp/echo_demo/FileList.csv \
    --volume-tracings /tmp/echo_demo/VolumeTracings.csv \
    --checkpoint ../segmentation/checkpoints/deeplabv3_lv_segmentation.pth \
    --max-videos 8
```