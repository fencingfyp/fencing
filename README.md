# Fencing Video Analysis Tool (FVAT)

## Installation Guide

---

### 1. Install Conda (Mamba Recommended)

Install **Miniforge** (recommended, includes conda and supports mamba):

- macOS (Apple Silicon / Intel):  
  https://github.com/conda-forge/miniforge

- Windows: download the installer and follow the setup wizard  
- Linux: use the `.sh` installer provided on the release page  

After installation, open a new terminal.

---

### 2. Create Environment

Using mamba (recommended):

```bash
mamba create -n fvat python=3.13
mamba activate fvat
```

If you do not have mamba:

```bash
conda create -n fvat python=3.13
conda activate fvat
```

---
### 3. Install Python Dependencies
Inside the activated environment:
```bash
pip install \
pyside6 \
opencv-python \
numpy \
pandas \
ultralytics \
easyocr \
```

---
### 4. Run the Application
Navigate to the project root directory:
```bash
cd path/to/FVAT
```
Then run:
```bash
python -m scripts.app
```

---
### Dependencies Summary
#### Conda
python 3.13

#### Pip
pyside6
opencv-python
numpy
pandas
ultralytics
easyocr

---

## Useful commands

Downloading a video from YouTube
```
yt-dlp -f "bv*" "video_url"
```

Cropping
```
ffmpeg -ss 00:12:00 -to 00:18:30 -i input.mp4 -c copy bout.mp4
```

Converting from H.264 to MPEG-4 codec (check the codec first)
```
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k output.mp4
```