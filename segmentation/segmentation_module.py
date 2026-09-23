import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model() -> nn.Module:
    """DeepLabV3 with a ResNet-50 backbone, head replaced for 1-class (LV) output.

    Section 6.1: DeepLabV3 (ResNet-50) is the primary architecture; U-Net is
    an acceptable fallback only if training time runs long. We start from
    torchvision's pretrained weights (COCO/ImageNet-pretrained backbone) and
    fine-tune, rather than training from random weights.
    """
    model = deeplabv3_resnet50(weights="DEFAULT")
    # Replace the classifier head's final conv to output 1 channel (LV vs background)
    # instead of torchvision's default 21 COCO classes.
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    # DeepLabV3 also has an auxiliary classifier head; either drop it or match
    # its output channels too so a checkpoint saved with aux_loss=True loads cleanly.
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model


def load_segmentation_model(checkpoint_path: str):
    """Loads the fine-tuned DeepLabV3 model. Returns a ready-to-use model object.

    Contract 1 signature -- takes only a checkpoint path, returns a model
    object ready for segment_video(). Handles CPU-only environments
    automatically (falls back from CUDA if unavailable).
    """
    model = _build_model()
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def extract_frames(video_path: str, resize=(112, 112)) -> np.ndarray:
    """Returns a NumPy array of shape (T, H, W, 3), dtype uint8.

    T = number of frames, in original video order, frame 0 = first frame.

    Uses OpenCV VideoCapture per Section 4.1 Stage 2. Frames are converted
    BGR -> RGB (OpenCV's native read order is BGR, but the model and every
    downstream consumer expects RGB) and resized to match the training
    resolution (112x112 by default, matching the pretrained checkpoints,
    per Section 5.3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        raise IOError(f"No frames could be read from video: {video_path}")

    return np.stack(frames, axis=0).astype(np.uint8)


def _normalize_batch(frames_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float32 tensor, ImageNet-normalized."""
    frames = frames_uint8.astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    frames = frames.transpose(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
    return torch.from_numpy(frames).float()


def segment_video(video_path: str, model, resize=(112, 112), batch_size: int = 16) -> np.ndarray:
    """Returns a NumPy array of shape (T, H, W), dtype uint8, values in {0, 1}.

    1 = left-ventricle pixel, 0 = background.
    T and frame order MUST exactly match extract_frames() on the same video
    -- this function calls extract_frames() internally with the same resize,
    so that guarantee holds by construction.

    Runs inference frame-by-frame (batched internally for speed) across the
    ENTIRE video, not just the ED/ES frames -- ground truth only exists for
    ED/ES (Section 5.2), but the deployed model must generalize across the
    full cardiac cycle, per Section 8.1.1.
    """
    frames = extract_frames(video_path, resize=resize)  # (T, H, W, 3) uint8
    num_frames = frames.shape[0]

    all_masks = []
    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            batch = frames[start:start + batch_size]
            batch_tensor = _normalize_batch(batch).to(DEVICE)

            output = model(batch_tensor)["out"]  # (B, 1, H, W) logits
            probs = torch.sigmoid(output)
            binary_masks = (probs > 0.5).squeeze(1).byte()  # (B, H, W) in {0,1}

            all_masks.append(binary_masks.cpu().numpy())

    masks = np.concatenate(all_masks, axis=0).astype(np.uint8)
    assert masks.shape[0] == num_frames, (
        f"Frame/mask count mismatch: {num_frames} frames vs {masks.shape[0]} masks"
    )
    return masks


# ---------------------------------------------------------------------------
# Usage example (per Contract 1: "a two-line usage example" deliverable)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = load_segmentation_model("checkpoints/deeplabv3_lv_segmentation.pth")
    masks = segment_video("sample_echo.avi", model)
    print(f"masks shape: {masks.shape}, dtype: {masks.dtype}, unique values: {np.unique(masks)}")
