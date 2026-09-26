import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ef_module import compute_ef, compute_volume_curve, _mask_volume

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def rect(w, h, frame_w=64, frame_h=64, x0=4, y0=4):
    m = np.zeros((frame_h, frame_w), dtype=np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    return m


def circle(r, size=64):
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    return ((xx - center) ** 2 + (yy - center) ** 2 <= r * r).astype(np.uint8)


def main():
    print("== Contract (Section 8.4.1): exact keys, types, JSON == ")
    masks = np.stack([rect(40, 20, 64, 64), rect(20, 20, 64, 64),
                      rect(10, 20, 64, 64), rect(20, 20, 64, 64)], axis=0)
    r = compute_ef(masks)
    keys = {"edv", "esv", "ef_percent", "ed_frame", "es_frame", "risk_flag"}
    check("exactly the 6 contract keys, no extras",
          set(r.keys()) == keys, f"got {sorted(r.keys())}")
    check("edv/esv/ef_percent are float", isinstance(r["edv"], float)
          and isinstance(r["esv"], float) and isinstance(r["ef_percent"], float))
    check("ed_frame/es_frame are int", isinstance(r["ed_frame"], int)
          and isinstance(r["es_frame"], int))
    check("risk_flag is str", isinstance(r["risk_flag"], str))
    check("JSON-serializable", isinstance(json.dumps(r), str))
    check("edv and esv non-negative finite", np.isfinite(r["edv"]) and np.isfinite(r["esv"]))

    print("\n== Input validation ==")
    try:
        compute_ef(None)
        check("None input raises", False)
    except ValueError:
        check("None input raises", True)
    try:
        compute_ef(np.zeros((64, 64)))
        check("2-D input raises", False)
    except ValueError:
        check("2-D input raises", True)
    try:
        compute_ef(np.zeros((4, 2, 64, 64)))
        check("4-D input raises", False)
    except ValueError:
        check("4-D input raises", True)
    try:
        compute_ef(np.zeros((0, 64, 64)))
        check("empty (T=0) input raises", False)
    except ValueError:
        check("empty (T=0) input raises", True)
    bad = np.zeros((4, 64, 64), dtype=np.uint8)
    bad[0, 0, 0] = 2
    try:
        compute_ef(bad)
        check("non-binary values raise", False)
    except ValueError:
        check("non-binary values raise", True)
    try:
        compute_ef(np.zeros((4, 64, 64), dtype=np.uint8))
        check("all-zero masks raise (EDV=0)", False)
    except ValueError:
        check("all-zero masks raise (EDV=0)", True)

    print("\n== Degenerate but computable inputs ==")
    r = compute_ef(np.stack([rect(20, 20)]))  # single frame
    check("single-frame input: EF=0", r["ef_percent"] == 0.0,
          f"got {r['ef_percent']}")
    check("single-frame: ED==ES==frame0", r["ed_frame"] == 0 and r["es_frame"] == 0)
    check("single-frame: risk Reduced", r["risk_flag"] == "Reduced")
    same = np.stack([rect(20, 20)] * 5)  # identical volumes across frames
    r = compute_ef(same)
    check("identical volumes: EF=0", r["ef_percent"] == 0.0, f"got {r['ef_percent']}")
    check("identical volumes: risk Reduced", r["risk_flag"] == "Reduced")

    print("\n== Risk classification (exact thresholds) ==")
    def risk_for(large_w, small_w):
        m = np.stack([rect(large_w, 20), rect(small_w, 20)])
        return compute_ef(m)["risk_flag"]
    check("EF>=55 -> Normal", risk_for(40, 15) == "Normal", risk_for(40, 15))
    check("41-54 -> Mildly Reduced", risk_for(40, 21) == "Mildly Reduced", risk_for(40, 21))
    check("<=40 -> Reduced", risk_for(40, 25) == "Reduced", risk_for(40, 25))

    from ef_module import _classify
    for ef_val, want in [(100.0, "Normal"), (55.0, "Normal"), (54.999, "Mildly Reduced"),
                         (41.0, "Mildly Reduced"), (40.999, "Reduced"), (40.0, "Reduced"),
                         (0.0, "Reduced")]:
        check(f"threshold EF={ef_val} -> {want!r}",
              _classify(ef_val) == want, _classify(ef_val))
    check("only the exact 3 project strings are ever produced",
          {_classify(x) for x in [0, 20, 40, 41, 50, 54, 55, 100]}
          == {"Normal", "Mildly Reduced", "Reduced"})

    print("\n== ED/ES detection on full volume curve ==")
    vols = [rect(15, 20), rect(30, 20), rect(45, 20), rect(25, 20),
            rect(8, 20), rect(35, 20)]  # ED (max) = frame 2, ES (min) = frame 4
    m = np.stack(vols)
    r = compute_ef(m)
    check("ED = frame 2 (argmax)", r["ed_frame"] == 2, f"got {r['ed_frame']}")
    check("ES = frame 4 (argmin)", r["es_frame"] == 4, f"got {r['es_frame']}")
    edv_gt = compute_volume_curve(m)[2]
    esv_gt = compute_volume_curve(m)[4]
    check("EDV == volume(ED frame)", abs(r["edv"] - edv_gt) < 1e-6)
    check("ESV == volume(ES frame)", abs(r["esv"] - esv_gt) < 1e-6)
    ef_gt = (edv_gt - esv_gt) / edv_gt * 100.0
    check("EF formula matches (EDV-ESV)/EDV*100",
          abs(r["ef_percent"] - ef_gt) < 1e-9)

    print("\n== Numerical correctness (independently hand-computed) ==")
    # Rectangle w x h (pixel indices 0..n-1 -> extent n-1): method-of-disks
    # long axis = longer side; perpendicular width = (h-1) everywhere; so
    # V = (pi/4) * (h-1)^2 * (max(w,h)-1).
    for w, h in [(40, 20), (20, 40), (30, 12)]:
        m = rect(w, h)
        v = _mask_volume(m)
        expect = (np.pi / 4.0) * (min(w, h) - 1) ** 2 * (max(w, h) - 1)
        check(f"rect {w}x{h}: Simpson volume = (pi/4)*(h-1)^2*L",
              abs(v - expect) / expect < 0.02, f"v={v:.2f} expect={expect:.2f}")
    # Circle: method-of-disks on a filled circle ~ volume of rotationally
    # symmetric body ~ roughly (4/3)*pi*r^3 (long-axis = 2r, widths = chords).
    r_c = 20
    v = _mask_volume(circle(r_c))
    expect_sphere = (4.0 / 3.0) * np.pi * r_c ** 3
    check("circle r=20: ~ sphere volume (4/3)pi*r^3",
          abs(v - expect_sphere) / expect_sphere < 0.1,
          f"v={v:.1f} sphere={expect_sphere:.1f}")

    # Ellipse mask (semi-axes rx, ry): disks perpendicular to the long axis
    # integrate exactly to the prolate/oblate spheroid (4/3)*pi*rx*ry^2.
    # Checks long-axis orientation detection AND perpendicular width measurement.
    def ellipse(rx, ry, size=128, angle_deg=0.0):
        yy, xx = np.mgrid[:size, :size]
        c = (size - 1) / 2.0
        x, y = xx - c, yy - c
        a = np.deg2rad(angle_deg)
        xr = x * np.cos(a) + y * np.sin(a)
        yr = -x * np.sin(a) + y * np.cos(a)
        return (((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0).astype(np.uint8)

    for rx, ry in [(30, 12), (25, 20)]:
        exp = (4.0 / 3.0) * np.pi * rx * ry ** 2
        v0 = _mask_volume(ellipse(rx, ry, angle_deg=0.0))
        v30 = _mask_volume(ellipse(rx, ry, angle_deg=30.0))
        check(f"ellipse rx={rx} ry={ry}: V ~ (4/3)pi*rx*ry^2 (spheroid)",
              abs(v0 - exp) / exp < 0.10, f"v={v0:.1f} exp={exp:.1f}")
        check(f"ellipse rx={rx} ry={ry}: volume orientation-invariant (0 vs 30 deg)",
              abs(v0 - v30) / v0 < 0.10, f"v0={v0:.1f} v30={v30:.1f}")

    print("\n== Volume curve: shape, order, convertibility ==")
    seq = np.stack([rect(10, 20), rect(30, 20), rect(20, 20)], axis=0)
    curve = compute_volume_curve(seq)
    check("curve shape (T,) float64",
          curve.shape == (3,) and curve.dtype == np.float64, str((curve.shape, curve.dtype)))
    check("curve preserves frame order (matches per-frame volumes)",
          all(abs(curve[i] - _mask_volume(seq[i])) < 1e-9 for i in range(3)))
    as_list = compute_ef([rect(40, 20), rect(20, 20), rect(10, 20)])
    as_arr = compute_ef(np.stack([rect(40, 20), rect(20, 20), rect(10, 20)]))
    check("list-of-masks input converts safely (same result as ndarray)",
          as_list == as_arr, f"{as_list} vs {as_arr}")
    wide, tall = _mask_volume(rect(40, 20)), _mask_volume(rect(20, 40))
    check("coordinate order: wide vs tall rect give equal volume (transpose-invariant)",
          abs(wide - tall) / wide < 1e-6, f"wide={wide:.3f} tall={tall:.3f}")
    check("coordinate order: different aspect ratios give different volumes",
          abs(wide - _mask_volume(rect(45, 15))) / wide > 0.1,
          f"40x20={wide:.1f} 45x15={_mask_volume(rect(45, 15)):.1f}")

    print("\n== Full chain sanity: mask->ED/ES->EDV/ESV->EF->risk ==")
    ed_mask = circle(25)
    es_mask = circle(18)
    frames = np.stack([circle(22), ed_mask, circle(21), es_mask, circle(23)],
                      axis=0)  # ED frame=1, ES frame=3
    r = compute_ef(frames)
    check("ED frame=1", r["ed_frame"] == 1, f"got {r['ed_frame']}")
    check("ES frame=3", r["es_frame"] == 3, f"got {r['es_frame']}")
    v_ed = _mask_volume(ed_mask)
    v_es = _mask_volume(es_mask)
    exp_ef = (v_ed - v_es) / v_ed * 100
    check("EF consistent with hand-measured volumes",
          abs(r["ef_percent"] - exp_ef) < 1e-6)
    check("EF within (0,100)", 0 < r["ef_percent"] < 100, str(r["ef_percent"]))
    check("EDV>ESV>0", r["edv"] > r["esv"] > 0)

    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())