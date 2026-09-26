"""FastAPI backend: echocardiogram video -> EF harness.

Wires Contract 1 (segmentation_module) into Contract 2 (ef_module) and
exposes them over HTTP. Model is loaded once at startup.
"""
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

from fastapi import FastAPI, File, UploadFile
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_BASE)
_ROOT = os.path.dirname(_APP)

sys.path.insert(0, os.path.join(_ROOT, "segmentation"))
sys.path.insert(0, os.path.join(_ROOT, "ef_engine"))

from segmentation_module import load_segmentation_model, segment_video
from ef_module import compute_ef


app = FastAPI(title="Cardiac Function AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
CHECKPOINT = os.environ.get(
    "SEG_CKPT",
    os.path.join(
        _ROOT,
        "segmentation",
        "checkpoints",
        "deeplabv3_lv_segmentation.pth",
    ),
)

_model = None


@app.on_event("startup")
def _load_model():
    global _model

    if os.path.exists(CHECKPOINT):
        _model = load_segmentation_model(CHECKPOINT)
    else:
        _model = None


@app.get("/health")
def health():
    return {
        "model_loaded": _model is not None,
        "checkpoint": CHECKPOINT,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if _model is None:
        return {
            "error": f"no model loaded; checkpoint not found at {CHECKPOINT}"
        }

    tmp = os.path.join(_BASE, "upload.avi")

    with open(tmp, "wb") as f:
        f.write(await file.read())

    masks = segment_video(tmp, _model)
    result = compute_ef(masks)
    areas = masks.sum(axis=(1, 2))
    invalid_frames = np.where(areas == 0)[0].tolist()

    return {
        "video": file.filename,
        "n_frames": int(masks.shape[0]),
        "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype),
        "mask_values": sorted(np.unique(masks).tolist()),
        "edv": result["edv"],
        "esv": result["esv"],
        "ef_percent": round(result["ef_percent"], 2),
        "ed_frame": int(result["ed_frame"]),
        "es_frame": int(result["es_frame"]),
        "risk_flag": result["risk_flag"],
        "invalid_mask_frames": invalid_frames,
"analysis_warning": (
    "Some frames produced empty LV segmentation masks; "
    "EF estimate may be unreliable."
    if invalid_frames
    else None
),
    }