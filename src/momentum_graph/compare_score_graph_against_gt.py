import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.momentum_graph.process_scores import process_scores

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and smooth score predictions.")
    parser.add_argument('gt', type=str, help='Path to the ground truth CSV file.')
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    args = parser.parse_args()
    return args.gt, args.folder

def main():
    gt, folder = parse_arguments()
    # Load both CSVs
    pred = pd.read_csv(f'{folder}/raw_scores.csv')
    gt = pd.read_csv(gt)

    pred = process_scores(pred)
    gt = process_scores(gt, smoothen=False, total_length=len(pred))

    # Rewrite the predictions CSV with cleaned data in this format: frame_id,left_score,right_score,left_confidence,right_confidence. set confidence to 1.0
    # frame_ids = np.arange(len(pred))
    # pred_df = pd.DataFrame({'frame_id': frame_ids, 'left_score': pred[:, 0], 'right_score': pred[:, 1], 'left_confidence': 1.0, 'right_confidence': 1.0})
    # pred_df.to_csv('outputs/foo_' + video_name + '_gray_binarised_score_est_cleaned.csv', index=False)

    # ---- Step 4: Plot both predictions and ground truth ----
    # --- Left ---
    plt.figure("Left", figsize=(12, 6))
    plt.plot(pred[:, 0], label='Pred Left (Smoothed)', color='blue', alpha=0.8)
    plt.plot(gt[:, 0], '--', color='red', label='GT Left')
    plt.title('Smoothed Predicted vs Ground Truth Scores (Left)')
    plt.xlabel('Frame ID')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Right ---
    plt.figure("Right", figsize=(12, 6))
    plt.plot(pred[:, 1], label='Pred Right (Smoothed)', color='blue', alpha=0.8)
    plt.plot(gt[:, 1], '--', color='red', label='GT Right')
    plt.title('Smoothed Predicted vs Ground Truth Scores (Right)')
    plt.xlabel('Frame ID')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show both figures
    plt.show()

if __name__ == "__main__":
    main()