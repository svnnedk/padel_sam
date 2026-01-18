# Padel Match Analysis with SAM3 and Deep Learning

Analyze padel match videos to extract player statistics, ball tracking, and match insights using state-of-the-art computer vision models.

## Features

- **Player Detection & Tracking**: YOLOv8 for detection, ByteTrack for consistent tracking
- **Precise Segmentation**: SAM2/SAM3 for pixel-accurate player masks
- **Ball Detection**: Hybrid color + motion detection for accurate ball tracking
- **Court Mapping**: Homography transformation for bird's-eye view analysis
- **Rally Detection**: Automatic point/rally detection and scoring
- **Statistics**: Distance covered, speed, position heatmaps, zone analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd padel_sam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download SAM2 checkpoint (optional, for segmentation)
# See: https://github.com/facebookresearch/segment-anything-2
mkdir -p checkpoints
# Download sam2_hiera_tiny.pt to checkpoints/
```

## Quick Start

```bash
# Basic analysis
python analyze.py match_video.mp4 --output analyzed.mp4

# Fast mode (no SAM segmentation)
python analyze.py match_video.mp4 --no-sam --output analyzed.mp4

# Preview mode
python analyze.py match_video.mp4 --preview

# With manual court corner selection
python analyze.py match_video.mp4 --preview --output analyzed.mp4
```

## Architecture

```
padel_sam/
├── analyze.py              # Main entry point
├── requirements.txt        # Dependencies
├── configs/
│   └── default.yaml        # Default configuration
├── src/
│   ├── pipeline.py         # Main analysis pipeline
│   ├── court/
│   │   ├── detector.py     # Court line detection
│   │   └── homography.py   # Coordinate transformation
│   ├── detection/
│   │   ├── player_detector.py  # YOLO-based player detection
│   │   └── ball_detector.py    # Ball detection (color/motion/ML)
│   ├── tracking/
│   │   └── tracker.py      # ByteTrack multi-object tracking
│   ├── segmentation/
│   │   └── sam_segmenter.py    # SAM2/SAM3 segmentation
│   ├── analysis/
│   │   ├── statistics.py   # Match statistics extraction
│   │   └── rally_detector.py   # Rally/point detection
│   └── utils/
│       ├── video.py        # Video I/O utilities
│       └── visualization.py    # Drawing and visualization
└── outputs/                # Analysis outputs
```

## Pipeline Flow

1. **Court Detection**: Detect court lines and compute homography matrix
2. **Per-Frame Processing**:
   - Player detection (YOLOv8)
   - Team classification (near/far based on position)
   - Player tracking (ByteTrack)
   - SAM segmentation (optional, for precise masks)
   - Ball detection (hybrid method)
   - Court position mapping (homography)
3. **Rally Detection**: Track ball movement to detect points
4. **Statistics**: Aggregate player movements, positions, speeds
5. **Output**: Annotated video, JSON statistics, heatmaps

## Output Statistics

The analysis produces:

- **Per-player stats**:
  - Total distance covered (meters)
  - Average/max speed
  - Sprint count
  - Time in different court zones
  - Position heatmap

- **Match stats**:
  - Total rallies
  - Average rally duration
  - Shots per rally
  - Team points

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Use larger SAM model for better accuracy
segmentation:
  model_type: "sam2_hiera_large"

# Adjust ball detection sensitivity
ball_detection:
  hsv_lower: [15, 80, 80]  # More lenient yellow detection
```

## API Usage

```python
from src.pipeline import PadelAnalysisPipeline

# Initialize pipeline
pipeline = PadelAnalysisPipeline(
    use_sam=True,
    sam_model='sam2_hiera_tiny',
    device='cuda'
)

# Analyze video
results = pipeline.analyze_video(
    'match.mp4',
    output_video='analyzed.mp4',
    show_preview=True
)

# Access statistics
print(results['statistics'])
print(results['rally_stats'])
```

## Single Frame Analysis

```python
import cv2
from src.pipeline import PadelAnalysisPipeline

pipeline = PadelAnalysisPipeline(use_sam=True)

# Analyze single frame
frame = cv2.imread('frame.jpg')
result = pipeline.analyze_frame(frame)

print(f"Detected {len(result['players'])} players")
if result['ball']:
    print(f"Ball at {result['ball']['center']}")
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for real-time processing)

## Models Used

- **YOLOv8**: Player detection (ultralytics)
- **SAM2/SAM3**: Precise segmentation (segment-anything-2)
- **ByteTrack**: Multi-object tracking (supervision)

## License

MIT License
