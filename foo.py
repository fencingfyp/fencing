"""
view_keypoints.py
-----------------
Draws SuperPoint keypoints on video frames using batched inference.
Processes `batch_size` frames per forward pass, unpacking with the
canonical HF pattern (torch.nonzero on the mask).

Usage:
    python view_keypoints.py path/to/video.mp4 [batch_size]
    python view_keypoints.py path/to/image.jpg

Press any key to advance, Q to quit.
"""

import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    sys.exit("Usage: python view_keypoints.py <video_or_image> [batch_size]")

path = sys.argv[1]
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}  batch_size: {batch_size}")

print("Loading model...")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = (
    SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    .to(device)
    .eval()
)
print("Model ready.\n")

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def frames_to_pil(frames: list[np.ndarray]) -> list[Image.Image]:
    """BGR uint8 frames -> grayscale PIL images with CLAHE."""
    pils = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        pils.append(Image.fromarray(gray))
    return pils


def extract_batch(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Run one SuperPoint forward pass on a batch of BGR frames.
    Returns list of (N_i, 2) float32 pixel-space xy arrays, one per frame.

    Unpacking follows the canonical HF pattern:
        image_indices = torch.nonzero(mask[i]).squeeze()
        keypoints     = outputs.keypoints[i][image_indices]
    Keypoints are in relative [0,1] space; multiply by [w, h] for pixels.
    """
    pils = frames_to_pil(frames)
    inputs = processor(pils, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]

        image_mask = outputs.mask[i]
        image_indices = torch.nonzero(image_mask).squeeze()

        # squeeze() collapses to scalar when only one keypoint — guard against it
        if image_indices.dim() == 0:
            image_indices = image_indices.unsqueeze(0)

        image_keypoints = outputs.keypoints[i][image_indices]  # (N, 2) in [0,1]

        scale = torch.tensor([w, h], dtype=torch.float32, device=image_keypoints.device)
        kp_px = (image_keypoints * scale).cpu().numpy().astype(np.float32)
        results.append(kp_px)

    return results


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_keypoints(frame: np.ndarray, kp: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for x, y in kp:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.putText(
        out,
        f"{len(kp)} keypoints",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    is_video = path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

    if is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"Could not open: {path}")

        print("Press any key to advance, Q to quit.")
        buf: list[np.ndarray] = []

        def show_buf(buf: list[np.ndarray]) -> bool:
            """Draw and show each frame in buf. Returns True if user quit."""
            for kp, fr in zip(extract_batch(buf), buf):
                cv2.imshow("SuperPoint keypoints", draw_keypoints(fr, kp))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return True
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                if buf:
                    show_buf(buf)
                print("End of video.")
                break

            buf.append(frame)

            if len(buf) == batch_size:
                if show_buf(buf):
                    break
                buf.clear()

        cap.release()

    else:
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"Could not read: {path}")

        kp_list = extract_batch([frame])
        vis = draw_keypoints(frame, kp_list[0])
        print(f"{len(kp_list[0])} keypoints. Press any key to close.")
        cv2.imshow("SuperPoint keypoints", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    cProfile.run("main()", "profile.stats")
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(10)
