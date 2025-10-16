import pytesseract
import cv2

# Path to tesseract executable (needed if it's not in PATH)
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image_path = "outputs/bw_12.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Invert: make text black on white
img = cv2.bitwise_not(img)

# scale 3
img = cv2.resize(img, None, fx=0.10, fy=0.10, interpolation=cv2.INTER_CUBIC)

# add padding around the image
pad = 100
# img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)

# OCR with digit whitelist
cv2.imshow("Processed Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
config = "--psm 7 -c tessedit_char_whitelist=0123456789"
text = pytesseract.image_to_data(img, config=config)


print("Extracted Text:")
print(text)
