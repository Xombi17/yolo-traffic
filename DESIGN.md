# Near-Miss Traffic Analytics — Project Report

## System Overview

```
Input Video → YOLO Detection → ByteTrack Tracking
  → Proximity Detection (Stage 3) → Risk Scoring (Stage 4)
  → Near-Miss Event Detection (Stage 5)
  → Visual Dashboard (Stage 6) + Bonuses B1–B4
  → Output Video + JSON Report
```

A single-file pipeline (`traffic_near_miss.py`) that processes traffic surveillance video frame by frame. Each frame goes through all stages sequentially. The architecture is monolithic by design — YOLO inference dominates per-frame latency, so any microservice split would add IPC overhead without throughput gains.

---

## Stage 1 — Object Detection

**Model:** `yolo26n.pt` (or `yolo26m.pt` on GPU) via Ultralytics.

**Classes detected:** person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7). Only these six COCO road-user classes are used — irrelevant classes (animals, traffic lights, signs) are filtered at inference time using YOLO's `classes` parameter.

**Confidence threshold:** 0.4 — balances false positives vs. missed detections at typical traffic-camera distances.

---

## Stage 2 — Object Tracking

**Tracker:** ByteTrack (`bytetrack.yaml`), built into YOLO.

Each detected object receives a persistent `track_id`. For every track, the last 30 center positions are stored in a `deque(maxlen=30)` — old positions drop off automatically, bounding memory and ensuring velocity estimates only reflect recent motion.

Track class is stored on first assignment and never updated — prevents class-ID flicker when YOLO occasionally changes its prediction frame to frame.

---

## Stage 3 — Proximity Detection

### 🎨 Designer's Choice: How Do You Measure 'Too Close'?

**Selected approach: Option C (nearest-edge distance) combined with Option B (expanded-box IoU).**

Both metrics must pass — not OR. This is a deliberate choice:

- **Edge distance alone (Option C)** would flag objects stacked vertically (e.g., one above another in different lanes) as close, even if they are separated horizontally by metres.
- **Expanded IoU alone (Option B)** would miss fast-moving objects whose boxes barely overlap in a single frame, even if their edges are nearly touching.
- Requiring both **reduces false positives** — a pair is only flagged when they are both near in edge distance *and* their expanded areas overlap.

**Scale normalisation (the perspective problem):** Objects farther from the camera appear smaller. A fixed pixel threshold would flag small distant objects as "close" when they are actually metres apart in the real world. To compensate, the edge-distance threshold is scaled by box height:

```
edge_threshold = min(h1, h2) × 1.5
```

This works because height is a rough perspective proxy — objects higher in the frame are farther away and have smaller boxes, so the threshold shrinks. Similarly, the box expansion for IoU is applied proportionally (1.5× width and height around the original center).

**Person-person pairs:** Excluded. Two pedestrians on the same footpath or crossing in front of each other are normal traffic behaviour, not a near-miss. At least one object in the pair must be a vehicle (car, bus, truck, motorcycle, or bicycle).

---

## Stage 4 — Risk Scoring

### 🎨 Designer's Choice: Design Your Risk Formula

**Output format:** Numerical score 0–100, mapped to three levels:
- **Low:** [0, 30)
- **Medium:** [30, 65)
- **High:** [65, 100]

A numerical score was chosen over discrete levels because it allows the near-miss state machine (Stage 5) to use precise thresholds (65 and 85) and supports fine-grained risk tracking (e.g., the dashboard timeline).

**Formula:**
```
risk = distance_score(0–40) + approach_speed_score(0–35) + type_danger_score(0–25)
```

Each component and its justification:

#### 1. Distance Score (0–40 points)

**Why this weight?** Distance is the most direct indicator of danger — if two objects are far apart, nothing else matters. 40/100 reflects this primacy.

**How it scales:** Based on nearest-edge distance relative to the combined box diagonal (diag_a + diag_b):

```
ratio = edge_distance / (diag_a + diag_b)
```

- ratio < 0.5 → 40 (maximum score)
- ratio > 2.0 → 0
- Between → linearly interpolated

**Why combined diagonal as the reference?** It normalises for object size. A 50px gap between two cars (with large diagonals) is proportionally smaller than the same 50px gap between a car and a pedestrian. Using a fixed pixel threshold would unfairly penalise small-object combinations. The combined diagonal naturally proportions the distance metric to the scale of the objects involved.

**Why linear scaling (not exponential)?** At the scale of traffic intersections, the relationship between pixel distance and real danger is roughly linear — a gap of 10px is twice as concerning as 20px. Exponential scaling would compress the meaningful range too aggressively.

#### 2. Approach Speed Score (0–35 points)

**Why this weight?** Speed adds crucial temporal context — two objects close but stationary have less risk than two approaching each other rapidly. 35/100 reflects that speed nearly doubles the risk potential.

**How it works:**
- Estimate each object's velocity vector from its last 10 center positions (`dx_avg`, `dy_avg`).
- Project each velocity onto the direction from that object toward the other:
  - `proj_a`: how fast A is moving toward B (negative means away)
  - `proj_b`: how fast B is moving toward A
- `closing_speed = proj_a + proj_b` — positive means they are approaching
- Score = `min(closing_speed × cap_factor, 35)`, where cap_factor = 2.0

**Why projection instead of absolute speed?** Two parallel-moving cars at 60 km/h in adjacent lanes have high absolute speed but zero closing speed — they aren't approaching each other. Projection isolates the dangerous component of motion.

**Why only positive closing speed scores?** If objects are moving apart, the situation is de-escalating by itself. Score = 0 for separating pairs (or one moving away from a stationary object).

**Why 10 frames for velocity estimation?** Short enough to capture recent motion trends, long enough to smooth out tracker jitter. At 30 fps this is ~330ms of history.

#### 3. Type Danger Score (0–25 points)

**Why this weight?** The combination of road-user types fundamentally changes the consequences of a collision. 25/100 reflects that type danger is a ceiling modifier — person vs vehicle should always score higher than vehicle vs vehicle at the same distance/speed.

**Mapping:**

| Combination | Score | Rationale |
|-------------|-------|-----------|
| Person + vehicle | 25 | A pedestrian hit by a car is the worst outcome — highest weight |
| Bicycle + vehicle | 20 | Cyclists are vulnerable but more manoeuvrable and aware than pedestrians |
| Motorcycle + vehicle | 20 | Equivalent risk to bicycles — rider is exposed |
| Vehicle + vehicle | 10 | Both parties have structural protection — lower consequence |
| Other / fallback | 5 | Covers edge cases (bicycle + person, etc.) |

**Why hardcoded and not learned?** Type danger is a safety policy decision, not a statistical pattern. Explicit mapping produces predictable, auditable scores that can be reviewed by city safety officials. A learned model would encode the biases of the training data and could not be easily adjusted per city policy.

### Risk Level Boundary Justification

- **Low [0, 30):** 0–30 covers objects that are close but stationary or separating, or distant but moving fast. These are non-actionable.
- **Medium [30, 65):** 30–65 indicates observable proximity worth noting but not alarming. Triggers yellow visual highlighting.
- **High [65, 100]:** 65+ is where active monitoring and alerts begin. 65 was chosen because it requires meaningful contributions from at least two of the three components (e.g., moderate distance + moderate speed + any type danger). 85+ (used in Stage 5 as the critical trigger) requires all three components to contribute significantly.

---

## Stage 5 — Near-Miss Event Detection

### 🎨 Designer's Choice: What Is a Near-Miss?

A near-miss is defined as a **sustained, high-risk interaction between two tracked road users** where:
1. Both objects are in physical proximity,
2. At least one is moving toward the other, and
3. The risk score stays critically high for a meaningful duration.

The definition is implemented as a per-pair finite state machine:

```
IDLE → APPROACHING (risk > 65 for 8 consecutive frames)
     → CRITICAL / ALERT (risk > 85 for 3 consecutive frames)
     → IDLE (risk < 40 for 5 consecutive frames → cooldown ends)
```

#### Phase 1 — Approaching (risk > 65 for 8 frames)

8 consecutive frames (~267ms at 30fps) with risk > 65. This filters out:
- Transient tracker jitter that briefly inflates edge distance
- Objects that pass each other at speed in a single frame
- Detection flicker from occlusion

**Why 8?** Fewer than 5 would be too sensitive to noise. More than 15 would miss brief but real near-misses (e.g., a car swerving around a pedestrian). 8 balances noise immunity with temporal coverage.

#### Phase 2 — Critical / Alert (risk > 85 for 3 frames)

3 consecutive frames (~100ms) with risk > 85 triggers the actual near-miss alert. The 3-frame buffer still provides noise immunity but allows real-time response. Risk > 85 requires all three formula components to contribute — very close proximity, high closing speed, and a dangerous object-type combination.

#### Recovery (risk < 40 for 5 frames)

Risk must drop below 40 for 5 consecutive frames to end the event. This hysteresis prevents rapid on/off toggling when risk hovers near the threshold. The 5-frame debounce means brief dips in risk (e.g., one frame of occlusion) don't prematurely end the event.

#### Cooldown (120 frames ≈ 4 seconds)

After an event ends, the same `(track_id_a, track_id_b)` pair enters a 120-frame cooldown. Without this, a stalled scenario (e.g., a car slowly creeping toward a pedestrian) would trigger dozens of near-miss events per minute.

#### Stationary Filter

Any object with average speed < 0.5 px/frame over the last 30 frames is tagged `stationary`. If both objects in a pair are stationary, no near-miss is triggered — two parked cars or a parked car next to a standing pedestrian don't constitute a near-miss regardless of proximity.

**Why both must be stationary?** If a moving car approaches a stationary pedestrian, that IS a near-miss. Only pairs where nothing is moving are filtered.

#### Direction Consideration

Is proximity alone enough, or must both objects be moving toward each other? The system uses **proximity + approach speed** in the risk formula, so direction is already factored into the score. If both objects move toward each other, the speed score contributes significantly, making phase transitions easier. If one approaches a stationary object, the speed score is halved (only one projection is positive), raising the bar for triggering but still allowing it at closer distances.

---

## Stage 6 — Visual Dashboard

### Box Color Coding

| Box Color | Condition | Meaning |
|-----------|-----------|---------|
| Green | Default state | Normal, no interaction detected |
| Yellow | Part of an interacting pair (Stage 3) | Proximity detected |
| Orange | Risk score > 65 | High risk |
| Red | Near-miss alert active (Stage 5) | ⚠ Near-miss triggered |

Orange as an intermediate color between yellow and red provides a visual gradient — the operator can see risk escalating before it becomes critical.

### Distance Line

A coloured line drawn between centers of every interacting pair. The line colour transitions yellow → orange → red based on the risk score, giving an at-a-glance indication of which interactions are most dangerous. Pixel distance is labelled at the midpoint.

### Dashboard Overlay (Top-Left)

A semi-transparent black panel with:
1. **Near-Miss Count:** Lifetime count of triggered events.
2. **Risk Level:** Current highest risk among all pairs (Low/Medium/High + peak numerical score).
3. **Vehicles:** Number of currently tracked objects.
4. **Timeline Bar:** Last 100 frames of peak risk plotted as colour-coded vertical strips. This provides ~3.3 seconds of risk history — long enough to perceive trends, short enough for real-time situation awareness.

---

## Bonus B1 — Danger Heatmap

A float32 heatmap array sized to the frame, accumulated over the video duration. On each near-miss frame, a Gaussian blob (σ = 20px) is added at the midpoint of the interacting pair using `np.maximum` — blobs stack rather than sum, preventing unbounded growth at busy intersections.

Heatmap decays by 0.5% per frame (`* 0.995`, half-life ≈ 138 frames / 4.6s), so hotspots fade over time. Rendered as a Jet-colormap overlay at α = 0.3 on the output frame.

**Why max instead of additive?** If the same intersection sees 50 near-misses, additive accumulation saturates and washes out spatial discrimination. Max preserves the "hottest" zone at each pixel independently, showing which exact positions near-misses cluster around.

**Why 0.5% decay?** Hotspots should persist long enough to be visible across multiple passes (~5 seconds) but not accumulate across the entire video duration. Higher decay would make them invisible too quickly; lower decay would retain hotspots from the first minute through the entire video.

---

## Bonus B2 — Speed Estimation (km/h)

Converts pixel displacement to real-world speed using a perspective camera model:

**Assumptions (configurable constants):**
- Camera height: 8m (typical traffic-gantry height)
- Vertical FOV: 50° (standard traffic lens)
- Camera tilt: 30° downward from horizontal

**Model:** For each object at y-coordinate `y` in a frame of height `H`:

```
φ(y) = tilt − FOV/2 + (y / H) × FOV    (ray angle from horizontal, in radians)
m_per_px = (h_camera × FOV_rad) / (H × sin²(φ))
```

Speed in km/h: `displacement_px × m_per_px × fps × 3.6`

**Why a sin² model?** A simple linear scale (more pixels = farther = larger m/pixel) is incorrect for tilted cameras — the horizon approaches infinite m/pixel while the area near the camera has very small m/pixel. The sin²(φ) model accounts for perspective foreshortening. At the bottom of the frame (near the camera), φ is small, sin²(φ) is small, so m/pixel is small. Near the horizon, φ → 90°, sin²(φ) → 1, so m/pixel approaches the maximum computed from camera height.

**Limitation:** This assumes all objects are on the same ground plane. A truck bed and a pedestrian on a sidewalk sit at different heights, introducing speed bias. This is an inherent monocular-camera limitation.

---

## Bonus B3 — Trajectory Prediction

**Approach:** Linear velocity extrapolation from the last 5 center positions.

For each tracked object, the mean velocity vector (dx, dy) is computed from the past 5 frames. A dashed cyan arrow is drawn from the current center position 15 frames into the future.

**For high-risk pairs (risk > 65):** The predicted positions of both objects (15 frames ahead) are compared. If they are within 30px of each other, a dashed magenta line is drawn between the predicted positions with a `PREDICTED INTERSECTION` label.

**Why linear?** Traffic motion is approximately linear over short horizons (0.5s at 30fps). Higher-order models (quadratic extrapolation, Kalman filters) add complexity and can overfit to noise in the small (5-point) window.

**Why 15 frames?** At 30fps this is 0.5 seconds — long enough to provide actionable warning, short enough that the linear assumption still holds. Beyond ~1 second, turns and braking make linear predictions unreliable.

**Why 5 frames for velocity?** Short enough to capture the current direction of motion, long enough to smooth out tracker jitter in the center-position estimate.

**Predicted-intersection threshold (30px):** At typical intersection scales (using the m/pixel model above), 30px corresponds to roughly 2–5 metres in the real world — close enough that corrective action is needed.

---

## Bonus B4 — Safety Report

At video end, a JSON report is printed to console and saved to `*_safety_report.json`.

| Field | Computation | Why This Metric |
|-------|------------|-----------------|
| `intersection_risk_score` | Mean of the top 5% highest per-frame peak risks | Overall mean is dominated by long idle periods (risk=0). Top 5% isolates the genuinely dangerous moments. |
| `most_dangerous_30s_window` | Sliding window (30 × fps frames) with the highest mean risk. Includes start/end frame numbers and wall-clock timestamps. | 30 seconds is short enough to isolate specific dangerous periods (a bus platoon, a pedestrian crossing, etc.) but long enough to avoid frame-level noise. |
| `object_type_involvement` | Dict mapping class name → count of near-miss participations | Answers "which road users are most frequently in near-misses?" for targeted safety interventions. |
| `near_miss_timestamps` | List of `(frame_number, pair_ids, peak_risk)` per event | Provides frame-level traceability — safety officials can jump to the exact moment of each near-miss in the output video. |

---

## All Thresholds — Summary & Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Confidence threshold | 0.4 | Standard YOLO detection threshold — good precision/recall trade-off for road users |
| Track history length | 30 frames (~1s) | Long enough for robust velocity estimation; short enough to respond to direction changes |
| **Stage 3 — Proximity** | | |
| Edge-distance threshold | `min(h1,h2) × 1.5` | Scale-invariant perspective proxy — object height correlates with distance from camera |
| IoU expansion factor | 1.5× centered | Close-enough-to-matter without requiring actual box overlap |
| **Stage 4 — Risk** | | |
| Distance max ratio | 0.5× combined diagonal | Objects closer than half their total extent deserve max distance score |
| Distance min ratio | 2.0× combined diagonal | Beyond twice the combined extent, distance contributes nothing |
| Velocity window | 10 frames (~330ms) | Recent enough for direction, long enough for smoothing |
| Speed cap factor | 2.0 | Determined empirically — maps typical intersection closing speeds to the 0–35 range |
| **Stage 5 — Events** | | |
| Phase 1 threshold | > 65 | Requires non-trivial contributions from ≥2 formula components |
| Phase 1 duration | 8 frames (~267ms) | Filters transient noise without missing real events |
| Phase 2 threshold | > 85 | Requires significant contributions from all 3 components |
| Phase 2 duration | 3 frames (~100ms) | Quick trigger but still noise-immune |
| Recovery threshold | < 40 | Below medium risk — situation has genuinely de-escalated |
| Recovery duration | 5 frames (~167ms) | Debounce against brief threshold crossings |
| Cooldown | 120 frames (~4s) | Prevents event storms; typical vehicle takes ≥4s to re-approach |
| Stationary threshold | < 0.5 px/frame (~0.1–0.3 km/h) | Below walking speed — effectively parked/stopped |
| **Bonus B3 — Trajectory** | | |
| Trajectory horizon | 15 frames (0.5s) | Actionable warning without exceeding linear-motion validity |
| Velocity window (traj.) | 5 frames | Smoothing vs. responsiveness trade-off |
| Predicted intersect. | 30px | Corresponds to ~2–5m real-world distance |
| **Bonus B1 — Heatmap** | | |
| Gaussian σ | 20px | Localised to the immediate vicinity of the near-miss |
| Decay rate | 0.995×/frame (half-life ≈ 138 frames) | Hotspots visible for ~5s then fade |
| Overlay α | 0.3 | Visible without obscuring scene |
| **Bonus B2 — Speed** | | |
| Camera height | 8m | Typical traffic gantry/pole |
| Vertical FOV | 50° | Standard traffic camera |
| Camera tilt | 30° | Typical downward angle |

---

## Deliverables Checklist

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Annotated output video with colour-coded bounding boxes | ✅ Green / Yellow / Orange / Red |
| 2 | Track IDs displayed on all detected road users | ✅ Shown with class name |
| 3 | Risk score/level shown on interacting pairs | ✅ Numerical score + level label |
| 4 | Near-miss event counter + visual alert | ✅ Dashboard counter + red box + "⚠ NEAR-MISS" overlay |
| 5 | Written explanation of risk formula and near-miss definition | ✅ This document |

## Bonus Checklist

| Bonus | Requirement | Status |
|-------|-------------|--------|
| B1 | Live danger heatmap | ✅ Gaussian accum., Jet overlay, decay |
| B2 | Speed in km/h | ✅ Perspective camera model (sin²) |
| B3 | Trajectory prediction + predicted intersection | ✅ Linear extrapolation, 15-frame horizon |
| B4 | Safety report (JSON) | ✅ Top 5%, 30s window, type involvement, timestamps |

---

## Limitations & Known Issues

1. **Monocular speed dependency:** Speed estimation relies on fixed camera parameters (height = 8m, VFOV = 50°, tilt = 30°). If the test video uses a different camera angle, absolute speeds will be inaccurate. Calibrating these parameters per video would improve accuracy.

2. **Ground-plane assumption:** The speed and distance models assume all objects are on the same ground plane. Trucks (tall) and pedestrians (short) at the same y-coordinate get different real-world positions. This is inherent to monocular vision without depth estimation.

3. **No occlusion handling:** ByteTrack can swap IDs or lose tracks when objects overlap in the frame. A re-identification module would add robustness.

4. **Single-camera range limits:** Objects beyond ~50m have tiny pixel footprints, making detection confidence low and box positions noisy.

5. **No weather/scene adaptation:** Thresholds are static. Rain, fog, darkness, or different intersection geometries would benefit from adaptive parameters.

6. **Linear trajectory only:** Turns, braking, and swerving are not modelled. A Kalman filter with constant-acceleration would handle curved trajectories better at the cost of parameter tuning per intersection.
