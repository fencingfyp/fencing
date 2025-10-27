import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.momentum_graph.util.extract_score_increases import _retroactive_flatten
from src.momentum_graph.process_scores import process_scores
from src.util import setup_input_video_io

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and smooth score predictions.")
    parser.add_argument('gt', type=str, help='Path to the ground truth CSV file.')
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    parser.add_argument('--flatten', action='store_true', help='Flatten temporary score increases due to incorrect point awarding.')
    args = parser.parse_args()
    return args.gt, args.folder, args.flatten

def frame_diffs(gt, pred):
    """
    gt and pred: np arrays of shape (n_frames, 2)
    Returns: dict with keys 'left' and 'right', each containing a list of frame differences per score increase
    """
    diffs = {'left': [0], 'right': [0]}
    for fencer in [0, 1]:
        gt_scores = gt[:, fencer]
        pred_scores = pred[:, fencer]
        max_score = max(gt_scores.max(), pred_scores.max())
        
        for score in range(1, int(max_score)+1):
            # first frame where score is reached
            gt_frame = np.argmax(gt_scores >= score)
            pred_frame = np.argmax(pred_scores >= score)
            diffs['left' if fencer == 0 else 'right'].append(abs(pred_frame - gt_frame))
            
    return diffs

def main():
    gt, folder, flatten = parse_arguments()
    # Load both CSVs
    pred = pd.read_csv(f'{folder}/raw_scores.csv')
    gt = pd.read_csv(gt)

    cap, fps, _, _, _ = setup_input_video_io(f'{folder}/cropped_scoreboard.mp4')
    cap.release()

    pred = process_scores(pred, window_median=int(fps*7))
    gt = process_scores(gt, smoothen=False, total_length=len(pred))

    # Optionally flatten both predictions and ground truth    
    if flatten:
        # check if pred is monotonic non-decreasing
        if not np.all(np.diff(pred[:, 0]) >= 0):
            print("Warning: Predictions are not monotonic non-decreasing on the left side.")
        if not np.all(np.diff(pred[:, 1]) >= 0):
            print("Warning: Predictions are not monotonic non-decreasing on the right side.")

        smoothed_pred = pd.DataFrame({'frame_id': np.arange(len(pred)), 'left_score': _retroactive_flatten(pred[:, 0].tolist()), 'right_score': _retroactive_flatten(pred[:, 1].tolist())
        })
        smoothed_gt = pd.DataFrame({'frame_id': np.arange(len(gt)), 'left_score': _retroactive_flatten(gt[:, 0].tolist()), 'right_score': _retroactive_flatten(gt[:, 1].tolist())
        })
        pred = process_scores(smoothed_pred, smoothen=False)
        gt = process_scores(smoothed_gt, smoothen=False)
        

    diffs = frame_diffs(gt, pred)
    diffs['left'] = [d / fps for d in diffs['left']]  # convert to seconds
    diffs['right'] = [d / fps for d in diffs['right']]

    # combined statistics
    total = diffs['left'] + diffs['right']
    print("MAE Total Score Increases (seconds):", np.mean(total))
    print("RMSE Total Score Increases (seconds):", np.sqrt(np.mean(np.array(total)**2)))
    print("Max deviation (seconds):", np.max(total))

    # diffs['left'] = [d / fps for d in diffs['left']]  # convert to seconds
    # diffs['right'] = [d / fps for d in diffs['right']]  # convert to seconds
    # print("MAE Left Score Increases (seconds):", np.mean(diffs['left']))
    # print("MAE Right Score Increases (seconds):", np.mean(diffs['right']))
    # print("RMSE Left Score Increases (seconds):", np.sqrt(np.mean(np.array(diffs['left'])**2)))
    # print("RMSE Right Score Increases (seconds):", np.sqrt(np.mean(np.array(diffs['right'])**2)))

    # plot diffs bar charts on separate figures, divide by fps to get seconds
    plt.figure("Differences (Left)", figsize=(12, 6))
    plt.bar(range(len(diffs['left'])), np.array(diffs['left']), label='Left Score Increases', color='blue', alpha=0.8)
    plt.title('Differences for Left Score Increases')
    plt.xlabel('Point ID')
    plt.ylabel('Difference (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure("Differences (Right)", figsize=(12, 6))
    plt.bar(range(len(diffs['right'])), np.array(diffs['right']), label='Right Score Increases', color='red', alpha=0.8)
    plt.title('Differences for Right Score Increases')
    plt.xlabel('Point ID')
    plt.ylabel('Difference (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # pred = pred[50000:]
    # gt = gt[50000:]

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