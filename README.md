# Near-Miss Traffic Analytics

Smart City road safety system that watches traffic surveillance video and automatically identifies near-miss events — moments where a collision between road users nearly occurred.

Built with Ultralytics YOLO26 for object detection and ByteTrack for multi-object tracking. Written in Python using OpenCV for video I/O and annotation rendering.

## Demo

[![Near-Miss Traffic Analytics](https://img.youtube.com/vi/BdNePB-CfK8/maxresdefault.jpg)](https://www.youtube.com/watch?v=BdNePB-CfK8)

## Pipeline

```
Input Video → YOLO Detection → ByteTrack Tracking
  → Proximity (edge-dist + expanded IoU)
  → Risk Scoring (distance + speed + type danger)
  → Near-Miss Event FSM (Approaching → Critical → Recovery)
  → Visual Dashboard + Color-Coded Boxes
  → Output Video + JSON Safety Report
```

All 6 stages and 4 bonuses implemented in a single file: `traffic_near_miss.py`.

## Requirements

- Python ≥ 3.9
- `ultralytics` (includes YOLO and ByteTrack)
- `opencv-python`

```
pip install ultralytics opencv-python
```

## Usage

```bash
python traffic_near_miss.py --video traffic.mp4 --model yolo26m.pt --show
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Path to input traffic video |
| `--model` | `yolo26m.pt` | Use `yolo26n.pt` for CPU, `yolo26m.pt` for GPU |
| `--conf` | `0.4` | Detection confidence threshold |
| `--output` | `output_annotated.mp4` | Path for annotated output video |
| `--show` | off | Show a live preview window while processing |

### Model Selection

- **CPU**: `yolo26n.pt` (nano — faster, lighter)
- **GPU**: `yolo26m.pt` (medium — better accuracy)

Detected classes (COCO): `person(0)`, `bicycle(1)`, `car(2)`, `motorcycle(3)`, `bus(5)`, `truck(7)`

## Output

An annotated MP4 video with:
- **Colored bounding boxes:** green (normal), yellow (interacting), orange (risk > 65), red (near-miss alert)
- **Track IDs**, class labels, confidence, and estimated speed (km/h) on each object
- **Distance lines** between interacting pairs with gradient colour and pixel-distance label
- **"⚠ NEAR-MISS" overlay** on active near-miss events
- **Dashboard** (top-left): near-miss count, current risk level, vehicle count, and 100-frame risk timeline
- **Trajectory arrows** (15-frame prediction) on all objects; magenta dashed lines for predicted intersections
- **Danger heatmap** (semi-transparent Jet overlay) accumulated over near-miss locations
- **JSON safety report** saved alongside the output video (`*_safety_report.json`)

## Design

All thresholds, Designer's Choice decisions, and design rationale are documented in [`DESIGN.md`](DESIGN.md).

**Risk formula** (three components, 0–100):
1. **Distance score** (0–40) — nearest-edge distance / combined box diagonal, linearly mapped
2. **Approach speed score** (0–35) — closing velocity projected along inter-object vector
3. **Type danger score** (0–25) — person+vehicle = 25, bicycle/motorcycle+vehicle = 20, vehicle+vehicle = 10, other = 5

**Near-miss definition** (per-pair FSM):
- APPROACHING: risk > 65 for 8 consecutive frames
- CRITICAL: risk > 85 for 3 consecutive frames → alert triggered
- Recovery: risk < 40 for 5 consecutive frames → event ends
- Cooldown: 120 frames before same pair can retrigger
- Stationary filter: both objects must not be stationary
