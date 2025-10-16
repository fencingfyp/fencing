import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare OCR readings against ground truth (with confidence filtering)")
    parser.add_argument("ocr_csv", help="Path to OCR results CSV (frame_id,left_score,right_score,left_confidence,right_confidence)")
    parser.add_argument("gt_csv", help="Path to ground truth CSV (frame_id,left_score,right_score, sparse changes only)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Exclude frames where either side's confidence is below this threshold (default=0.0)")
    args = parser.parse_args()

    # Load CSVs
    ocr_df = pd.read_csv(args.ocr_csv)
    gt_df = pd.read_csv(args.gt_csv)

    # Prepare ground truth: expand to all frames via forward-fill
    gt_df = gt_df.sort_values("frame_id").set_index("frame_id")
    full_index = pd.RangeIndex(start=0, stop=int(ocr_df["frame_id"].max()) + 1)
    gt_full = gt_df.reindex(full_index, method="ffill").ffill().reset_index()
    gt_full = gt_full.rename(columns={"index": "frame_id"})

    # count statistics in the ocr data
    ocr_stats = {
        "total_frames": len(ocr_df),
        "nan_left": ocr_df["left_score"].isna().sum(),
        "nan_right": ocr_df["right_score"].isna().sum(),
        "nan_both": ((ocr_df["left_score"].isna()) & (ocr_df["right_score"].isna())).sum(),
    }
    print("OCR DataFrame statistics:")
    for k, v in ocr_stats.items():
        print(f"  {k}: {v}")
    
    # print the first few rows of each dataframe for debugging
    # print("OCR DataFrame sample:")
    # print(ocr_df.head())
    # print("\nGround Truth DataFrame sample:")
    # print(gt_full.head())

    # Merge expanded GT with OCR readings
    merged = pd.merge(ocr_df, gt_full, on="frame_id", suffixes=("_ocr", "_gt"))
    if merged.empty:
        print("No overlapping frame IDs after merge.")
        return

    # print the first few rows of the merged dataframe for debugging
    # print("\nMerged DataFrame sample:")
    # print(merged.head())
    # print(f"\nTotal merged frames: {len(merged)}")

    # Filter by confidence threshold
    conf_mask = (merged["left_confidence"] >= args.confidence_threshold) & \
                (merged["right_confidence"] >= args.confidence_threshold)
    filtered = merged[conf_mask]

    if filtered.empty:
        print(f"No frames passed confidence threshold {args.confidence_threshold}.")
        return

    # Compute correctness only for filtered frames
    # Apply confidence threshold safely
    filtered = merged[
        (merged["left_confidence"] >= args.confidence_threshold) &
        (merged["right_confidence"] >= args.confidence_threshold)
    ].copy()

    # Compute correctness using .loc
    filtered.loc[:, "left_correct"] = filtered["left_score_ocr"] == filtered["left_score_gt"]
    filtered.loc[:, "right_correct"] = filtered["right_score_ocr"] == filtered["right_score_gt"]

    # Statistics
    total_frames = len(merged)
    used_frames = len(filtered)
    left_acc = filtered["left_correct"].mean() * 100
    right_acc = filtered["right_correct"].mean() * 100
    both_acc = ((filtered["left_correct"] & filtered["right_correct"]).mean()) * 100

    avg_left_conf = filtered["left_confidence"].mean()
    avg_right_conf = filtered["right_confidence"].mean()

    print(f"Total frames available: {total_frames}")
    print(f"Frames above confidence {args.confidence_threshold}: {used_frames} ({used_frames / total_frames * 100:.2f}%)\n")

    print(f"Left score accuracy: {left_acc:.2f}%")
    print(f"Right score accuracy: {right_acc:.2f}%")
    print(f"Both-sides-correct accuracy: {both_acc:.2f}%")
    print(f"Avg left confidence (used): {avg_left_conf:.3f}")
    print(f"Avg right confidence (used): {avg_right_conf:.3f}")

    # Mismatch samples
    mismatches = filtered[(~filtered["left_correct"]) | (~filtered["right_correct"])]
    if not mismatches.empty:
        print("\nSample mismatches (first 10):")
        print(mismatches[["frame_id", "left_score_ocr", "left_score_gt", "right_score_ocr", "right_score_gt"]].head(10))


if __name__ == "__main__":
    main()
