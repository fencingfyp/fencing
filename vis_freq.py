import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.util.utils import setup_input_video_io


def fft_spectrum(img):
    # ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1e-8)  # log scale for visibility
    return mag


def plot_frequency(img1, img2=None, label1="A", label2="B"):
    if img2 is None:
        spec = fft_spectrum(img1)
        plt.figure(figsize=(5, 5))
        plt.title(f"Spectrum {label1}")
        plt.imshow(spec, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    # match sizes
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    s1 = fft_spectrum(img1)
    s2 = fft_spectrum(img2)

    # signed difference
    diff = s1 - s2
    vmax = np.max(np.abs(diff))  # symmetrical colour range

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Spectrum {label1}")
    plt.imshow(s1, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Spectrum {label2}")
    plt.imshow(s2, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Signed Difference (A - B)")
    plt.imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder1 = "sabre_1_full"
    file1 = f"outputs/{folder1}/original.mp4"
    cap1, _, _, _, _ = setup_input_video_io(file1)
    folder2 = "sabre_1_full"
    file2 = f"outputs/{folder2}/bout.mp4"
    cap2, _, _, _, _ = setup_input_video_io(file2)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        print("Comparing frequencies")
        plot_frequency(frame1, frame2, folder1, folder2)
    else:
        if ret1:
            print(f"Processing frequency for {folder1}")
            plot_frequency(frame1)

        if ret2:
            print(f"Processing frequency for {folder2}")
            plot_frequency(frame2)

    cap1.release()
    cap2.release()
