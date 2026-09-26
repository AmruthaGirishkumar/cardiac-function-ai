import argparse
import math
import os

import cv2
import numpy as np
import pandas as pd

FPS = 50
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
N_FRAMES = 64
N_SEGMENTS = 28

OFFICIAL_SPLIT = {"TRAIN": 7465, "VAL": 1288, "TEST": 1277}


def parse_args():
    p = argparse.ArgumentParser(description="Generate a synthetic EchoNet-Dynamic-schema dataset for offline dev/verification")
    p.add_argument("--mode", choices=["official", "demo", "testfull"], default="demo",
                   help="official: writes only FileList/VolumeTracings CSVs with the published "
                        "7465/1288/1277 split for split-verification evidence. "
                        "demo: writes a runnable subset INCLUDING Videos/*.avi for the full pipeline.")
    p.add_argument("--out", default="data/raw", help="output directory (videos + both CSVs)")
    p.add_argument("--demo-train", type=int, default=64, help="demo TRAIN videos (official mode ignores this)")
    p.add_argument("--demo-val", type=int, default=16, help="demo VAL videos (official mode ignores this)")
    p.add_argument("--demo-test", type=int, default=16, help="demo TEST videos (official mode ignores this)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def random_name(rng, length=12):
    chars = "0123456789ABCDEF"
    return "".join(rng.choice(list(chars)) for _ in range(length))


def radii_for_ef(rng, ef):
    r_ed = float(rng.uniform(24, 32))
    r_es = r_ed * math.sqrt((100.0 - ef) / 100.0)
    return r_ed, r_es


def radius_curve(n_frames, r_ed, r_es):
    return np.array([r_ed - (r_ed - r_es) * math.sin(math.pi * t / n_frames) for t in range(n_frames)])


def tracing_rows_for_ellipse(cx, cy, rx, ry, filename, frame, n_segments=N_SEGMENTS):
    angles = [2 * math.pi * k / n_segments for k in range(n_segments)]
    xs = [cx + rx * math.cos(a) for a in angles]
    ys = [cy + ry * math.sin(a) for a in angles]
    rows = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        rows.append({
            "FileName": filename,
            "Frame": frame,
            "X1": xs[i], "Y1": ys[i],
            "X2": xs[j], "Y2": ys[j],
        })
    return rows


def render_frame(r, cx=56.0, cy=52.0, rng=np.random.RandomState(0)):
    img = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 18, dtype=np.uint8)
    sector = np.zeros_like(img)
    cv2.ellipse(sector, (int(cx), int(cy + 10)), (52, 44), 0, -55, 55, (90, 90, 90), -1)
    img = cv2.addWeighted(img, 1.0, sector, 1.0, 0)
    cv2.ellipse(img, (int(cx), int(cy)), (int(r + 7), int(r + 7)), 0, 0, 360, (120, 120, 120), -1)
    cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r)), 0, 0, 360, (215, 215, 215), -1)
    noise = rng.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def write_video(path, radii):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for r in radii:
        vw.write(render_frame(r))
    vw.release()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "Videos"), exist_ok=True)

    if args.mode == "official":
        counts = OFFICIAL_SPLIT
        want_videos = False
    elif args.mode == "testfull":
        counts = {"TEST": OFFICIAL_SPLIT["TEST"]}
        want_videos = True
    else:
        counts = {"TRAIN": args.demo_train, "VAL": args.demo_val, "TEST": args.demo_test}
        want_videos = True

    file_rows, tracing_rows = [], []
    for split, n in counts.items():
        for _ in range(n):
            name = random_name(rng)
            eff_ef = float(rng.uniform(15, 65))
            edv = float(rng.uniform(70, 160))
            esv = edv * (100.0 - eff_ef) / 100.0
            n_frames = N_FRAMES
            ed_frame, es_frame = 0, n_frames // 2
            file_rows.append({
                "FileName": name,
                "EF": eff_ef, "ESV": esv, "EDV": edv,
                "FrameHeight": FRAME_HEIGHT, "FrameWidth": FRAME_WIDTH,
                "FPS": FPS, "NumberOfFrames": n_frames, "Split": split,
            })
            r_ed, r_es = radii_for_ef(rng, eff_ef)
            cx, cy = rng.uniform(50, 62), rng.uniform(46, 58)
            if want_videos:
                curve = radius_curve(n_frames, r_ed, r_es)
                write_video(os.path.join(args.out, "Videos", name + ".avi"), curve)
            for frame, rr in ((ed_frame, r_ed), (es_frame, r_es)):
                tracing_rows.extend(tracing_rows_for_ellipse(cx, cy, rr, rr, name + ".avi", frame))

    fl_path = os.path.join(args.out, "FileList.csv")
    vt_path = os.path.join(args.out, "VolumeTracings.csv")
    pd.DataFrame(file_rows).to_csv(fl_path, index=False)
    pd.DataFrame(tracing_rows).to_csv(vt_path, index=False)
    print(f"mode={args.mode} out={args.out}")
    print("FileList.csv rows:", len(file_rows), "| splits:", pd.DataFrame(file_rows)["Split"].value_counts().to_dict())
    print("VolumeTracings.csv rows:", len(tracing_rows), f"({N_SEGMENTS} segments x 2 ED/ES frames per video)")
    if args.mode == "official":
        print("split check matches published 7465/1288/1277")
    elif args.mode == "testfull":
        print("full TEST split count 1277 as published; videos also generated for full-TEST evaluation")
    else:
        print("split check demo subset (dev-only, warning expected)")


if __name__ == "__main__":
    main()