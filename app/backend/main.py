"""FastAPI backend: echocardiogram video -> EF harness.

Wires Contract 1 (segmentation_module) into Contract 2 (ef_module) and
exposes them over HTTP. Model is loaded once at startup.
"""

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
from ef_module import compute_ef_session

app = FastAPI(title="Cardiac Function AI")

CHECKPOINT = os.environ.get(
    "SEG_CKPT",
    os.path.join(_ROOT, "segmentation", "checkpoints", "deeplabv3_lv_segmentation.pth"),
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
    return {"model_loaded": _model is not None, "checkpoint": CHECKPOINT}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if _model is None:
        return {"error": f"no model loaded; checkpoint not found at {CHECKPOINT}"}
    tmp = os.path.join(_BASE, "upload.avi")
    with open(tmp, "wb") as f:
        f.write(await file.read())
    masks = segment_video(tmp, _model)
    session = compute_ef_session(masks)
    return {
        "video": file.filename,
        "n_frames": int(masks.shape[0]),
        "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype),
        "mask_values": sorted(np.unique(masks).tolist()),
        "ef_percent": round(session["ef"], 2),
        "ed_frame": int(session["ed_frame"]),
        "es_frame": int(session["es_frame"]),
    }