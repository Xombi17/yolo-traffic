# Near-Miss Traffic Analytics

Smart City road safety system that watches traffic surveillance video and automatically identifies near-miss events — moments where a collision between road users nearly occurred.

Built with Ultralytics YOLO26 for object detection and ByteTrack for multi-object tracking.

## Pipeline

```
Video → Detection (YOLO) → Tracking (ByteTrack) → Proximity → Risk Scoring → Near-Miss Events → Dashboard
```

| Stage | Component | Status |
|-------|-----------|--------|
| 1 | Object detection on road users (person, bicycle, car, motorcycle, bus, truck) | ✅ Done |
| 2 | Multi-object tracking with persistent IDs via ByteTrack | ✅ Done |
| 3 | Proximity detection — nearest-edge distance + expanded IOU | 🔲 Planned |
| 4 | Risk scoring (0–100) — distance, approach speed, object type | 🔲 Planned |
| 5 | Near-miss event detection — sustained high risk with cooldown | 🔲 Planned |
| 6 | Visual dashboard — colored boxes, distance lines, overlay counters | 🔲 Planned |
| B1–B4 | Heatmap, speed (km/h), trajectory prediction, safety report | 🔲 Planned |

## Requirements

- Python ≥ 3.9
- `ultralytics` — YOLO model loading and inference
- `opencv-python` — video I/O and annotation rendering
- `byteTrack` — bundled with ultralytics (`bytetrack.yaml`)

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

Detected classes (COCO): `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`

## Output

An annotated MP4 video with:
- Colored bounding boxes (green = normal, yellow = interacting, orange = high risk, red = near-miss)
- Track IDs and class labels on each object
- Distance lines between interacting pairs
- Top-left dashboard overlay (near-miss count, risk level, vehicle count)

## Design

Risk is computed from three factors:
1. **Distance** (0–40): proximity of nearest bounding-box edges, scaled by object size
2. **Approach speed** (0–35): closing velocity projected along the vector between objects
3. **Type danger** (0–25): weighted by the road-user combination (e.g. person+vehicle = highest)

A near-miss is a sustained interaction exceeding risk thresholds over consecutive frames, with recovery detection and per-pair cooldown to avoid duplicates.

Full thresholds and design rationale: [`AGENTS.md`](AGENTS.md)
