import numpy as np

N_DISKS = 20
RISK_NORMAL = "Normal"
RISK_MILD = "Mildly Reduced"
RISK_REDUCED = "Reduced"


def _mask_volume(mask: np.ndarray, n_disks: int = N_DISKS) -> float:
    """Simpson's method-of-disks volume (relative pixel^3 units) for one binary mask.

    The LV long axis is the mask's principal (largest-eigenvalue) direction. The
    measured length L along that axis is divided into ``n_disks`` equal intervals of
    height h = L / n; for each interval the LV width L_i (extent perpendicular to the
    long axis) is measured from the mask, and the chamber volume is estimated as

        V ≈ (pi/4) * sum_i (L_i^2 * h)

    This is the standard method-of-disks (Simpson) approximation used for 2D-echo LV
    volumes. Because the mask is a 2D image, the volumes are RELATIVE (pixel^3) units,
    not millilitres; no absolute calibration constant is invented (see tech doc). The
    same pixel grid is used for every frame, so EDV/ESV are mutually comparable and the
    EF ratio is scale-invariant.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0  # degenerate / nearly-empty mask
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center

    cov = centered.T @ centered / len(centered)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return 0.0
    axis /= norm

    t = centered @ axis
    t_min, t_max = float(t.min()), float(t.max())
    length = t_max - t_min
    if length <= 0.0:
        return 0.0
    h = length / n_disks

    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    u = centered @ perp
    volume = 0.0
    for i in range(n_disks):
        lo = t_min + i * h
        hi = lo + h
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 1:
            continue
        width = float(u[sel].max() - u[sel].min())
        volume += (np.pi / 4.0) * (width ** 2) * h
    return float(volume)


def compute_volume_curve(masks: np.ndarray) -> np.ndarray:
    """Per-frame Simpson volume for every frame, in original frame order.

    Returns np.ndarray of shape (T,) (float64). Mirrors the frame ordering of
    ``segmentation_module.segment_video`` output so that index i always
    corresponds to frame i of the source video.
    """
    masks = _validate(masks)
    curve = np.array([_mask_volume(m) for m in masks], dtype=np.float64)
    return curve


def _validate(masks) -> np.ndarray:
    if masks is None:
        raise ValueError("masks must be a NumPy array, got None")
    arr = np.asarray(masks)
    if arr.ndim != 3:
        raise ValueError(f"expected exactly 3 dimensions (T, H, W), got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("mask sequence has no frames (T must be >= 1)")
    if arr.size > 0:
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, (0, 1))):
            raise ValueError(f"mask values must be binary in {{0, 1}}, got {uniq.tolist()}")
    return arr.astype(np.uint8)


def _classify(ef_percent: float) -> str:
    if ef_percent >= 55.0:
        return RISK_NORMAL
    if ef_percent >= 41.0:
        return RISK_MILD
    return RISK_REDUCED


def compute_ef(masks: np.ndarray) -> dict:
    """Compute EDV, ESV, EF% and a risk flag from a full-cardiac-cycle mask stack.

    Args:
        masks: np.ndarray shape (T, H, W), dtype uint8, values in {0, 1}, in original
               frame order (exactly the output of Contract 1's ``segment_video``).
               1 = LV pixel, 0 = background.

    Returns (exact contract, JSON-serializable):
        {
            "edv": float,          # Simpson volume at ED (relative pixel^3 units)
            "esv": float,          # Simpson volume at ES
            "ef_percent": float,   # (edv - esv) / edv * 100
            "ed_frame": int,       # argmax of the full volume curve (ED)
            "es_frame": int,       # argmin of the full volume curve (ES)
            "risk_flag": str       # "Normal" | "Mildly Reduced" | "Reduced"
        }

    ED/ES are detected automatically from the complete volume curve
    (ED = maximum-volume frame, ES = minimum-volume frame), never assumed.

    Raises:
        ValueError if the input is None, not 3-D, has no frames, holds non-binary
        values, or if the detected EDV is zero (nothing can be computed safely).
    """
    masks = _validate(masks)
    curve = compute_volume_curve(masks)

    ed_frame = int(np.argmax(curve))
    es_frame = int(np.argmin(curve))
    edv = float(curve[ed_frame])
    esv = float(curve[es_frame])

    if edv == 0.0:
        raise ValueError("EDV is zero (no LV volume detected in any frame); "
                         "cannot compute a meaningful EF")

    ef_percent = (edv - esv) / edv * 100.0  # never divides by zero (guarded above)
    return {
        "edv": edv,
        "esv": esv,
        "ef_percent": float(ef_percent),
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "risk_flag": _classify(ef_percent),
    }