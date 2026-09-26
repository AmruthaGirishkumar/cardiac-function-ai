import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds_module
import losses as losses_module
import segmentation_module as sm


class Check:
    def __init__(self, name):
        self.name = name
        self.failures = []

    def ok(self, cond, msg):
        if not cond:
            self.failures.append(msg)

    def done(self):
        if self.failures:
            print(f"  [FAIL] {self.name}")
            for f in self.failures:
                print(f"         - {f}")
            return False
        print(f"  [PASS] {self.name}")
        return True


def make_synthetic_data(tmp, demo=(64, 16, 16)):
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_synthetic_dataset.py"),
         "--mode", "demo", "--out", tmp, "--demo-train", str(demo[0]), "--demo-val", str(demo[1]),
         "--demo-test", str(demo[2])],
        check=True, capture_output=True,
    )
    return os.path.join(tmp, "Videos"), os.path.join(tmp, "FileList.csv"), os.path.join(tmp, "VolumeTracings.csv")


def main():
    tmp = tempfile.mkdtemp(prefix="segtest_")
    videos_dir, fl_csv, vt_csv = make_synthetic_data(tmp)
    paths = ds_module.EchoNetPaths(videos_dir, fl_csv, vt_csv)

    print("== Contract 1 (Section 8.4.1) + Track A requirements ==")
    all_ok = True

    # 4/5/6 -- extract_frames
    c = Check("extract_frames -> (T,H,W,3) uint8 @112x112 (Req 4,5,6)")
    video = os.path.join(videos_dir, os.listdir(videos_dir)[0])
    frames = sm.extract_frames(video)
    c.ok(frames.ndim == 4, f"expected 4D, got {frames.ndim}D")
    c.ok(frames.shape[1] == 112 and frames.shape[2] == 112, f"expected 112x112, got {frames.shape[1:3]}")
    c.ok(frames.shape[3] == 3, f"expected 3 channels, got {frames.shape[3]}")
    c.ok(frames.dtype == np.uint8, f"expected uint8, got {frames.dtype}")
    all_ok &= c.done()

    # 9/10/20 -- model build + load by path only
    c = Check("DeepLabV3-R50 + load_segmentation_model(path) (Req 9,10,20,22)")
    model = sm._build_model()
    c.ok(isinstance(model, torch.nn.Module), "model not a nn.Module")
    ckpt = os.path.join(tmp, "trained.pth")
    torch.save(model.state_dict(), ckpt)
    loaded = sm.load_segmentation_model(ckpt)
    c.ok(not loaded.training, "model not in eval mode after load")
    all_ok &= c.done()

    # 11 -- BCE+Dice
    c = Check("BCE+Dice loss computes (Req 11)")
    crit = losses_module.BCEDiceLoss()
    logits = torch.randn(4, 112, 112)
    tgt = (torch.rand(4, 112, 112) > 0.8).float()
    val = float(crit(logits, tgt))
    c.ok(np.isfinite(val), f"loss non-finite: {val}")
    all_ok &= c.done()

    # 13/14/23 -- segment_video contract
    c = Check("segment_video -> (T,H,W) uint8 {0,1}, order matches (Req 13,14,23)")
    model2 = sm._build_model()
    torch.save(model2.state_dict(), os.path.join(tmp, "t2.pth"))
    m2 = sm.load_segmentation_model(os.path.join(tmp, "t2.pth"))
    masks = sm.segment_video(video, m2)
    c.ok(masks.ndim == 3, f"expected 3D, got {masks.ndim}D")
    c.ok(masks.dtype == np.uint8, f"expected uint8, got {masks.dtype}")
    c.ok(set(np.unique(masks).tolist()) <= {0, 1}, f"values not in {{0,1}}: {np.unique(masks)}")
    c.ok(masks.shape[0] == frames.shape[0], f"T mismatch: {masks.shape[0]} vs {frames.shape[0]}")
    c.ok(masks.shape[1] == 112 and masks.shape[2] == 112, f"H,W not 112: {masks.shape[1:]}")
    all_ok &= c.done()

    # 7/8 -- GT masks ED/ES only + no crash (tracing_to_mask)
    c = Check("GT masks from VolumeTracings, ED/ES only (Req 7,8)")
    vt = ds_module.load_volume_tracings(paths)
    fname = vt["FileName"].iloc[0]
    per_frame = vt[vt.FileName == fname]["Frame"].nunique()
    c.ok(per_frame == 2, f"expected exactly 2 ED/ES frames, got {per_frame}")
    sub = vt[vt.FileName == fname]
    m = ds_module.tracing_to_mask(sub, 112, 112)
    c.ok(set(np.unique(m).tolist()) <= {0, 1}, "mask not binary")
    c.ok(int(m.sum()) > 0, "empty mask from valid tracing")
    d0 = ds_module.EchoNetSegmentationDataset(paths, split="TRAIN")
    c.ok(len(d0) == 2 * 64, f"expected 2*64 ED/ES items, got {len(d0)}")
    all_ok &= c.done()

    # 12 -- augmentation params
    c = Check("augmentation: ±10deg rot, no hflip, mild contrast/brightness (Req 12)")
    import inspect
    src = inspect.getsource(ds_module.EchoNetSegmentationDataset._augment)
    c.ok("uniform(-10, 10)" in src or "uniform(-10,10)" in src, "rotation range not ±10")
    c.ok("cv2.flip" not in src and "np.flip" not in src, "horizontal flip applied (must be absent)")
    c.ok("alpha" in src and "beta" in src, "brightness/contrast jitter missing")
    all_ok &= c.done()

    # 24/25/26 -- packaging
    c = Check("requirements.txt + usage example + importable module (Req 24,25,26)")
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    c.ok(os.path.exists(req_path), "requirements.txt missing")
    c.ok(os.path.getsize(req_path) > 0, "requirements.txt empty")
    c.ok(sm.__file__.endswith("segmentation_module.py"), "deliverable not a module file")
    all_ok &= c.done()

    # 19/20/21/22 -- checkpoint exists + full pipeline run (trained ckpt path)
    c = Check("trained checkpoint load + end-to-end run (Req 19,21)")
    trained = os.environ.get("SEG_CKPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "checkpoints", "deeplabv3_lv_segmentation.pth"))
    if os.path.exists(trained):
        tmodel = sm.load_segmentation_model(trained)
        tmasks = sm.segment_video(video, tmodel)
        c.ok(tmasks.shape[0] == frames.shape[0] and tmasks.dtype == np.uint8, "trained-model output contract broken")
    else:
        c.ok(False, f"trained checkpoint not found at {trained} (set SEG_CKPT)")
    all_ok &= c.done()

    print()
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())