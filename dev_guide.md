# Developer Guide

## Installation
Refer to the main `README.md` for installation instructions.

## Quick Start
**Input**: `.mp4` video  
**Output**: A `.data/` folder containing intermediate results and visualisations
1. Download the video from this link: https://www.youtube.com/watch?v=IdjZ97NRqHQ
2. Crop the video into a shorter `.mp4` format (using `ffmpeg` or some other video editor).
3. Run `python -m scripts.app` to start the app.
4. Using the app, select the cropped video obtained in step 2.
5. Run the momentum graph and heat map workflows. Their outputs should look similar to the images in `sample_outputs/`.
   - It is expected that the pose tracking task will take ~10-15 mins to complete.
   - If there are any unclear areas, take note of them. This can be used to improve the UX of the application.

## Project Structure
The project is organised into two main directories:
- `scripts/`
  - Contains utility scripts
  - Includes the main application entry point (`app.py`)
  - Houses OpenCV-based workflow prototypes
  - Note: Some scripts use an old version of data storage where the video file is stored in the same folder as its processed outputs. To run those scripts, copy the video into the data folder and rename it to `ORIGINAL_VIDEO_NAME` as defined in this codebase.
- `src/`
  - Contains all core application logic
  - All new development should go here
  - Exception: standalone OpenCV prototype scripts may remain in `scripts/`

## Development Guidelines
### OpenCV vs PySide6 (GUI)
For rapid prototyping, OpenCV is recommended:
- Minimal boilerplate
- Faster iteration and debugging
- Easier parameter tweaking

For production UI, PySide6 should be used:
- Significantly better user experience
- More flexible and scalable UI architecture

⚠️ Note: OpenCV and PySide6 use fundamentally different rendering models. Code written for one will require non-trivial adaptation to work with the other.


### Data Storage
Data for each video is stored in the same folder as the original video in a sidecar manner, as requested by HPSI. They share the same name as the original video, with a `.data` suffix.

### Pipeline Design Philosophy

The system is designed as a sequence of loosely coupled stages. Each stage:

- Consumes structured input (e.g. video frames or prior outputs)
- Produces intermediate outputs that are persisted to disk
- Intermediate outputs should be treated as the primary interface between stages.

These intermediate outputs serve two purposes:
- **Visual validation**: Many stages (e.g. cropping, OCR, light detection) can be verified visually. Persisting outputs allows quick inspection and debugging.
- **Modularity**: Stages can be developed, tested, and iterated independently without rerunning the entire pipeline.

#### Development vs Production Workflows

- **Development workflows** tend to expose more intermediate steps and outputs for inspection.
- **Production workflows** may combine multiple stages into a single step (e.g. read + process) to reduce overhead and user interaction.

As a result, the exact pipeline structure may evolve, but the core principle remains:
> Each stage should produce inspectable outputs and be composable with other stages.

### Workflow Components
#### 1. Region Cropping
- Typically one of the first stages in a pipeline
- Location: `src/pyside_pipelines/multi_region_cropper`
- Responsible for defining and tracking quadrilateral regions of interest (ROI)
- Can be reused for extracting structured regions (e.g. timers, scoreboards)

#### 2. OCR
Used after ROIs have been defined. Two OCR pipelines are implemented:

- **General OCR**
  - Location: `src/model/reader/EasyOcrReader`
  - Uses EasyOCR for generic text detection
- **Seven-Segment OCR**
  - Location: `src/model/reader/SevenSegmentReader`
  - Uses a custom MobileNetV2-based model
  - Training code: `src/seven_segment`

#### 3. Score Light Detection
Used for detecting scoring events from scoreboard lights. Also used after ROIs have been defined.
Location: `src/model/AutoPatchLightDetector`

Current approach:
- Requires reference OFF and ON images
- Detects valid hits (red/green lights) vs default (off state)

Limitations:
- Does not currently support invalid hits (white light)

Possible extensions:
- Add reference images for additional states (e.g. white light)
- Replace or extend Mahalanobis distance with:
  - Earth Mover’s Distance (Wasserstein-1)
  - Bhattacharyya distance
  - Jensen–Shannon divergence

#### 4. Raw Pose Extraction
A fundamental step in certain pipelines.
Location: `src/gui/heat_map/track_poses_widget`
Notes:
- Uses multiprocessing to accelerate pose data extraction
- TODO: Benchmark effectiveness
- May require refactoring or removal depending on performance validation


### Stability Guidelines

- **Stable (change carefully)**:
  - Data storage format (`.data` sidecar)
  - Core models and detectors (`src/model/*`)

- **Evolving**:
  - PySide workflows and pipeline structure

- **Experimental (safe to modify freely)**:
  - `scripts/` and OpenCV prototypes
  - Performance-related components (e.g. multiprocessing)

#### GUI-specific notes
- Avoid modifying `navbar` and `task_graph` unless necessary
- `base_task_widget` is widely used; changes may have broad impact