import os
import zipfile
from setuptools import setup
import subprocess

# 1. Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# 2. Extract zip contents
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# 3. Find the first mp4 file
data_folder = "data"
mp4_file = None
for root, _, files in os.walk(data_folder):
    for file in files:
        if file.lower().endswith(".mp4"):
            mp4_file = os.path.join(root, file)
            break
    if mp4_file:
        break

if mp4_file:
    # 4. Create video_image folder and subfolder with the mp4 name
    video_image_folder = os.path.join(data_folder,"video_image", os.path.splitext(os.path.basename(mp4_file))[0])
    os.makedirs(video_image_folder, exist_ok=True)

    # 5. Run the conversion script from scripts folder
    subprocess.run([
        "python3",
        os.path.join("scripts", "convert_video_to_image.py"),
        mp4_file,
        video_image_folder
    ])
else:
    print("No mp4 file found in the data folder.")

