#!/usr/bin/env python3
import torch
from ultralytics import YOLO

def main():
    print("🔍 Checking PyTorch backend...")
    print("PyTorch version:", torch.__version__)

    if torch.backends.mps.is_available():
        print("✅ MPS (Apple GPU) is available.")
    else:
        print("❌ MPS not available, using CPU.")

    print("\n🚀 Loading YOLO model...")
    model = YOLO("yolo11m-pose.pt")  # smallest model for quick test

    # Run a quick inference on a sample image (downloaded automatically)
    results = model.predict(source="https://ultralytics.com/images/bus.jpg", device="mps" if torch.backends.mps.is_available() else "cpu")

    print("\n✅ YOLO inference complete. Results:")
    for r in results:
        print(r.summary())

if __name__ == "__main__":
    main()
