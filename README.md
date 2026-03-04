# Fencing Video Analysis Tool (FVAT)

## Installation Guide

---

## 1. Install Conda (Mamba Recommended)

Install **Miniforge** (recommended, includes conda and supports mamba):

- macOS (Apple Silicon / Intel):  
  https://github.com/conda-forge/miniforge

- Windows: download the installer and follow the setup wizard  
- Linux: use the `.sh` installer provided on the release page  

After installation, open a new terminal.

---

## 2. Create Environment

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
## 3. Install Graphviz (System Dependency)

Graphviz must be installed on your system.

### macOS
```bash
brew install graphviz
```

### Ubuntu / Debian
```bash
sudo apt install graphviz
```

### Windows
Download and install from:
https://graphviz.org/download/

After installation, verify it is available:
```bash
dot -V
```

---
## 4. Install Python Dependencies
Inside the activated environment:
```bash
pip install \
pyside6 \
opencv-python \
numpy \
pandas \
ultralytics \
easyocr \
graphviz
```

---
## 5. Run the Application
Navigate to the project root directory:
```bash
cd path/to/FVAT
```
Then run:
```bash
python -m scripts.app
```

---
## Dependencies Summary
### Conda
python 3.13

### Pip
pyside6
opencv-python
numpy
pandas
ultralytics
easyocr
graphviz

### System
Graphviz 