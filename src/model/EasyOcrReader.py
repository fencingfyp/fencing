import cv2
import easyocr
import numpy as np


class EasyOcrReader:
    def __init__(self, device, seven_segment=False):
        self.reader = easyocr.Reader(["ch_sim"], gpu=device)
        self.seven_segment = seven_segment

    def read(self, image, max_digits=2, height_tol=0.8, pad=20):
        """
        Reads numeric content from an image.

        Args:
            image: Input (BGR or grayscale)
            seven_segment: If True, segment digits by contours before OCR
            max_digits: Max number of digits to extract (for seven-segment mode)
            height_tol: Relative height threshold (fraction of tallest contour)

        Returns:
            (score, conf): recognized string and confidence
        """

        # Helper: simple contour-based segmentation
        def segment_digits(roi):
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            _, bin_img = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return []

            boxes = [cv2.boundingRect(c) for c in contours]
            boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
            tallest_h = boxes[0][3]
            filtered = [b for b in boxes if b[3] >= tallest_h * height_tol]
            filtered = sorted(filtered[:max_digits], key=lambda b: b[0])
            return filtered

        # --- Normal OCR mode ---
        if not self.seven_segment:
            results = self.reader.recognize(image, allowlist="0123456789")
            score, conf = "", 0
            if results:
                for bbox, text, prob in sorted(
                    results, key=lambda x: x[2], reverse=True
                ):
                    if bbox is None or len(bbox) != 4 or not text:
                        continue
                    score = text
                    conf = prob
                    break
            return score, conf

        # --- Seven-segment mode ---
        boxes = segment_digits(image)
        if not boxes:
            return "", 0.0

        digits, confidences = [], []

        for x, y, w, h in boxes:
            # Crop strictly within the box (no pre-padding)
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(image.shape[1], x + w)
            y1 = min(image.shape[0], y + h)

            crop = image[y0:y1, x0:x1]

            crop = cv2.copyMakeBorder(
                crop,
                top=pad,
                bottom=pad,
                left=pad,
                right=pad,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            results = self.reader.recognize(crop, allowlist="0123456789")
            if results:
                best = max(results, key=lambda x: x[2])  # highest confidence
                _, text, prob = best
                digits.append(text)
                confidences.append(prob)

        if digits:
            final_score = "".join(digits)
            avg_conf = np.mean(confidences) if confidences else 0
            return final_score, float(avg_conf)
        else:
            return "", 0.0
