import argparse
import sys

import pandas as pd

EXPECTED_SPLIT = {"TRAIN": 7465, "VAL": 1288, "TEST": 1277}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect EchoNet-Dynamic FileList.csv and VolumeTracings.csv")
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--volume-tracings", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("EchoNet-Dynamic Dataset Inspection")
    print("=" * 70)

    # ---- FileList.csv ----
    file_list = pd.read_csv(args.file_list)
    print(f"\nFileList.csv: {len(file_list)} rows")
    print(f"Columns: {list(file_list.columns)}")

    required_cols = {"FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth",
                      "FPS", "NumberOfFrames", "Split"}
    missing_cols = required_cols - set(file_list.columns)
    if missing_cols:
        print(f"  !! MISSING EXPECTED COLUMNS: {missing_cols}")
    else:
        print("  All expected columns present.")

    print("\nSplit counts:")
    actual_counts = file_list["Split"].value_counts().to_dict()
    all_ok = True
    for split, expected_count in EXPECTED_SPLIT.items():
        actual = actual_counts.get(split, 0)
        status = "OK" if actual == expected_count else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  {split:6s}: {actual:5d}  (expected {expected_count})  [{status}]")

    print(f"\nEF ground-truth range: {file_list['EF'].min():.1f}% - {file_list['EF'].max():.1f}%")
    print(f"Null values in FileList.csv: {file_list.isnull().sum().sum()}")

    # ---- VolumeTracings.csv ----
    tracings = pd.read_csv(args.volume_tracings)
    print(f"\nVolumeTracings.csv: {len(tracings)} rows (line segments)")
    print(f"Columns: {list(tracings.columns)}")

    videos_with_tracings = tracings["FileName"].nunique()
    print(f"Unique videos with tracings: {videos_with_tracings}")

    # Each video should have exactly 2 labeled frames (ED + ES)
    frames_per_video = tracings.groupby("FileName")["Frame"].nunique()
    wrong_frame_count = frames_per_video[frames_per_video != 2]
    if len(wrong_frame_count) > 0:
        print(f"  !! {len(wrong_frame_count)} videos do NOT have exactly 2 labeled frames (ED+ES):")
        print(f"     {wrong_frame_count.head(10).to_dict()}")
    else:
        print("  Every video has exactly 2 labeled frames (ED + ES). OK.")

    # ---- Cross-check: do FileList.csv and VolumeTracings.csv filenames actually match? ----
    # KNOWN QUIRK of the real EchoNet-Dynamic release: FileList.csv's FileName
    # has NO .avi extension, but VolumeTracings.csv's FileName DOES. If these
    # aren't reconciled, every downstream join silently returns zero matches.
    import os
    file_list_names = set(file_list["FileName"].astype(str))
    tracing_names_raw = set(tracings["FileName"].astype(str))
    tracing_names_stripped = set(os.path.splitext(str(n))[0] for n in tracing_names_raw)

    raw_overlap = len(file_list_names & tracing_names_raw)
    stripped_overlap = len(file_list_names & tracing_names_stripped)

    print(f"\nFilename cross-check (FileList.csv vs VolumeTracings.csv):")
    print(f"  Direct match (no extension handling): {raw_overlap} videos overlap")
    print(f"  Match after stripping .avi from VolumeTracings names: {stripped_overlap} videos overlap")
    if raw_overlap == 0 and stripped_overlap > 0:
        print("  -> Extension mismatch confirmed (expected quirk). dataset.py already handles this "
              "by stripping the extension before matching -- no action needed.")
    elif stripped_overlap == 0:
        print("  !! WARNING: filenames don't match even after stripping extensions. "
              "Something else is different between the two files -- inspect manually before training.")

    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: Split counts match Section 5.3 exactly. Safe to proceed.")
    else:
        print("RESULT: Split counts DO NOT match the published split.")
        print("  If you're using the full official download, stop and re-download --")
        print("  do not re-shuffle the split yourself, since published benchmarks")
        print("  rely on this exact split for comparability.")
        print("  If you're intentionally using a small sample subset for early")
        print("  pipeline testing (per Section 11's mitigation for delayed access),")
        print("  this mismatch is expected and fine to proceed with for now.")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
