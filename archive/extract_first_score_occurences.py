import argparse
import pandas as pd
from collections import Counter

def extract_first_score_occurrences(csv_path, window=50, conf_threshold=0.5, majority_fraction=0.5):
    """
    Reads a CSV and returns the first frame_id where each score is confidently displayed for both players.
    Uses a sliding window and requires a score to appear in >= majority_fraction of the window to count.

    Args:
        csv_path: CSV path with columns [frame_id, left_score, right_score, left_confidence, right_confidence]
        window: number of frames in sliding window
        conf_threshold: minimum confidence to consider OCR reading
        majority_fraction: fraction of frames in window that must match to accept the score

    Returns:
        {
            "left": {score: first_frame_id or None},
            "right": {score: first_frame_id or None}
        }
    """
    df = pd.read_csv(csv_path)
    frame_ids = df["frame_id"].tolist()
    left_scores = df["left_score"].astype(str).tolist()
    right_scores = df["right_score"].astype(str).tolist()
    left_conf = df["left_confidence"].tolist()
    right_conf = df["right_confidence"].tolist()

    first_occurrences = {"left": {}, "right": {}}
    seen_scores = {"left": set(), "right": set()}

    for i in range(len(frame_ids)):
        window_end = min(i + window, len(frame_ids))

        for side, scores, confs in [
            ("left", left_scores, left_conf),
            ("right", right_scores, right_conf),
        ]:
            chunk_scores = scores[i:window_end]
            chunk_confs = confs[i:window_end]
            chunk_frame_ids = frame_ids[i:window_end]

            # filter confident scores
            confident_scores = [
                s for s, c in zip(chunk_scores, chunk_confs)
                if c >= conf_threshold and s != "nan"
            ]
            if not confident_scores:
                continue

            counts = Counter(confident_scores)
            most_common, count = counts.most_common(1)[0]

            try:
                score_val = int(float(most_common))
            except ValueError:
                continue

            # accept only if majority of window matches
            if most_common == scores[i] and count / len(chunk_scores) >= majority_fraction:
                if score_val not in seen_scores[side]:
                    first_occurrences[side][score_val] = chunk_frame_ids[0]
                    seen_scores[side].add(score_val)

    # fill missing scores with None
    for side in ["left", "right"]:
        all_scores = set()
        try:
            all_scores |= set(int(float(s)) for s in left_scores if s != "nan")
            all_scores |= set(int(float(s)) for s in right_scores if s != "nan")
        except Exception:
            pass
        for s in all_scores:
            if s not in first_occurrences[side]:
                first_occurrences[side][s] = None
    
    # delete scores > 15
    for side in ["left", "right"]:
        to_delete = [s for s in first_occurrences[side] if s > 15]
        for s in to_delete:
            del first_occurrences[side][s]

    # add a +window buffer to each occurrence (to account for delays)
    # for side in ["left", "right"]:
    #     for s in first_occurrences[side]:
    #         if first_occurrences[side][s] is not None:
    #             first_occurrences[side][s] += window * 2

    return first_occurrences

def get_header_row() -> list[str]:
    return [
        "fencer_direction",  # "left" or "right"
        *[f"score_{i+1}" for i in range(15)]
    ]

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guess score timings from a list of readings")
    parser.add_argument("csv_path", help="Path to CSV with score readings")
    args = parser.parse_args()
    file_name = args.csv_path

    first_occurrences = extract_first_score_occurrences(file_name)
    for side in ["left", "right"]:
        print(f"{side.capitalize()} first occurrences (score → frame_id):")
        for score, frame_id in sorted(first_occurrences[side].items()):
            print(score, frame_id)
    # output_csv_path = "outputs/foil_2_full_first_occurrences.csv"
    # with open(output_csv_path, "w") as f:
    #     f.write(",".join(get_header_row()) + "\n")
    #     for side in ["left", "right"]:
    #         row = [side] + [str(first_occurrences[side].get(i + 1, "")) for i in range(15)]
    #         f.write(",".join(row) + "\n")
