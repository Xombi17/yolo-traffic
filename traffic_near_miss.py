"""
Near-Miss Traffic Analytics — Stage 1 & 2
Object Detection + Tracking (ByteTrack)

Usage:
    python traffic_near_miss.py --video traffic.mp4 --model yolo26m.pt --show
"""

import argparse
import json
import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class ids we care about for road users
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
ROAD_USER_CLASSES = list(CLASS_NAMES.keys())

HISTORY_LEN = 30  # frames of center-position history kept per track (used later for velocity)
DISPLAY_MAX_DIM = 1000  # preview window is capped to this many pixels on its longest side

# Stage 3 — Proximity Detection Constants
EDGE_DIST_SCALE = 1.5
IOU_EXPAND_SCALE = 1.5
IOU_THRESHOLD = 0.0

# Bonus B2 — Speed Estimation Constants
CAMERA_HEIGHT_M = 8.0
CAMERA_VFOV_DEG = 50.0
CAMERA_TILT_DEG = 30.0  # camera tilt from horizontal (0=horizontal, 90=straight down)

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
PERSON_CLASS = "person"

# Bonus B3 — Trajectory Prediction Constants
TRAJECTORY_HORIZON = 15
PREDICTED_INTERSECTION_THRESHOLD = 30


def nearest_edge_distance(box_a, box_b):
    """Compute nearest-edge distance between two boxes [x1, y1, x2, y2]."""
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    dx = max(0.0, max(x1b - x2a, x1a - x2b))
    dy = max(0.0, max(y1b - y2a, y1a - y2b))
    return math.hypot(dx, dy)


def expanded_iou(box_a, box_b, scale=IOU_EXPAND_SCALE):
    """Compute IoU of two boxes expanded by `scale` around their centers."""
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    cx_a, cy_a = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx_b, cy_b = (x1b + x2b) / 2, (y1b + y2b) / 2

    w_a, h_a = (x2a - x1a) * scale, (y2a - y1a) * scale
    w_b, h_b = (x2b - x1b) * scale, (y2b - y1b) * scale

    ea_x1, ea_y1 = cx_a - w_a / 2, cy_a - h_a / 2
    ea_x2, ea_y2 = cx_a + w_a / 2, cy_a + h_a / 2
    eb_x1, eb_y1 = cx_b - w_b / 2, cy_b - h_b / 2
    eb_x2, eb_y2 = cx_b + w_b / 2, cy_b + h_b / 2

    ix1 = max(ea_x1, eb_x1)
    iy1 = max(ea_y1, eb_y1)
    ix2 = min(ea_x2, eb_x2)
    iy2 = min(ea_y2, eb_y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_interacting_pairs(boxes, ids, cls_ids, track_history):
    """
    Find interacting pairs based on nearest-edge distance and expanded IoU.
    Returns list of dicts: {id_a, id_b, box_a, box_b, cls_a, cls_b, dist, iou, center_a, center_b}
    """
    pairs = []
    n = len(boxes)
    if n < 2:
        return pairs

    for i in range(n):
        for j in range(i + 1, n):
            id_a, id_b = int(ids[i]), int(ids[j])
            cls_a = CLASS_NAMES.get(int(cls_ids[i]), str(cls_ids[i]))
            cls_b = CLASS_NAMES.get(int(cls_ids[j]), str(cls_ids[j]))

            # Filter: ignore person-person pairs
            if cls_a == PERSON_CLASS and cls_b == PERSON_CLASS:
                continue

            # Filter: at least one must be a vehicle
            if cls_a not in VEHICLE_CLASSES and cls_b not in VEHICLE_CLASSES:
                continue

            box_a = boxes[i]
            box_b = boxes[j]

            h_a = box_a[3] - box_a[1]
            h_b = box_b[3] - box_b[1]
            edge_thresh = min(h_a, h_b) * EDGE_DIST_SCALE

            dist = nearest_edge_distance(box_a, box_b)
            iou = expanded_iou(box_a, box_b)

            # Both must pass for high-confidence; edge alone for borderline
            if dist <= edge_thresh and iou > IOU_THRESHOLD:
                cx_a, cy_a = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
                cx_b, cy_b = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
                pairs.append({
                    "id_a": id_a, "id_b": id_b,
                    "box_a": box_a, "box_b": box_b,
                    "cls_a": cls_a, "cls_b": cls_b,
                    "dist": dist, "iou": iou,
                    "center_a": (cx_a, cy_a), "center_b": (cx_b, cy_b),
                })
    return pairs


# ===================== Stage 4: Risk Scoring =====================

# Type danger mapping
TYPE_DANGER = {
    frozenset(("person", "car")): 25,
    frozenset(("person", "bus")): 25,
    frozenset(("person", "truck")): 25,
    frozenset(("person", "motorcycle")): 25,
    frozenset(("person", "bicycle")): 25,
    frozenset(("bicycle", "car")): 20,
    frozenset(("bicycle", "bus")): 20,
    frozenset(("bicycle", "truck")): 20,
    frozenset(("bicycle", "motorcycle")): 20,
    frozenset(("motorcycle", "car")): 20,
    frozenset(("motorcycle", "bus")): 20,
    frozenset(("motorcycle", "truck")): 20,
    frozenset(("car", "bus")): 10,
    frozenset(("car", "truck")): 10,
    frozenset(("bus", "truck")): 10,
    frozenset(("car", "car")): 10,
    frozenset(("bus", "bus")): 10,
    frozenset(("truck", "truck")): 10,
    frozenset(("motorcycle", "motorcycle")): 10,
    frozenset(("bicycle", "bicycle")): 10,
    frozenset(("person", "person")): 5,
}

def type_danger_score(cls_a, cls_b):
    key = frozenset((cls_a, cls_b))
    return TYPE_DANGER.get(key, 5)


def estimate_velocity(track_history, track_id, frames=10):
    """Estimate velocity vector (vx, vy) from last `frames` positions."""
    hist = track_history.get(track_id, [])
    if len(hist) < 2:
        return 0.0, 0.0
    n = min(frames, len(hist))
    # Use last n positions
    recent = list(hist)[-n:]
    vx = recent[-1][0] - recent[0][0]
    vy = recent[-1][1] - recent[0][1]
    vx /= (n - 1)
    vy /= (n - 1)
    return vx, vy


def risk_score(pair, track_history):
    """Compute risk score 0-100 for an interacting pair."""
    cls_a = pair["cls_a"]
    cls_b = pair["cls_b"]
    dist = pair["dist"]
    center_a = pair["center_a"]
    center_b = pair["center_b"]
    box_a = pair["box_a"]
    box_b = pair["box_b"]

    # --- Distance score (0-40) ---
    # combined diagonal
    w_a, h_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
    w_b, h_b = box_b[2] - box_b[0], box_b[3] - box_b[1]
    diag_a = (w_a ** 2 + h_a ** 2) ** 0.5
    diag_b = (w_b ** 2 + h_b ** 2) ** 0.5
    combined_diag = diag_a + diag_b

    if dist < 0.5 * combined_diag:
        dist_score = 40
    elif dist > 2.0 * combined_diag:
        dist_score = 0
    else:
        # linear interpolation
        dist_score = 40 * (2.0 * combined_diag - dist) / (1.5 * combined_diag)

    # --- Approach speed score (0-35) ---
    id_a = pair["id_a"]
    id_b = pair["id_b"]
    vx_a, vy_a = estimate_velocity(track_history, id_a)
    vx_b, vy_b = estimate_velocity(track_history, id_b)

    # Direction from A to B
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    d_ab = (dx ** 2 + dy ** 2) ** 0.5

    if d_ab > 0:
        # Unit vector from A to B
        ux, uy = dx / d_ab, dy / d_ab
        # Project A's velocity onto A->B direction (positive = moving toward B)
        proj_a = vx_a * ux + vy_a * uy
        # Project B's velocity onto B->A direction (positive = moving toward A)
        proj_b = vx_b * (-ux) + vy_b * (-uy)
        closing_speed = proj_a + proj_b  # positive = approaching
    else:
        closing_speed = 0.0

    if closing_speed > 0:
        # Cap at 35, scale proportionally
        approach_score = min(35, closing_speed * 2.0)  # tune factor
    else:
        approach_score = 0.0

    # --- Type danger score (0-25) ---
    type_score = type_danger_score(cls_a, cls_b)

    total = dist_score + approach_score + type_score
    return min(100, max(0, total))


def risk_level(score):
    """Map 0-100 score to risk level string."""
    if score >= 65:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


# ===================== Stage 5: Near-Miss Event Detection =====================

# Event state tracking
# pair_key -> {"phase1_frames": int, "phase2_frames": int, "recovery_frames": int, 
#              "cooldown": int, "triggered": bool, "peak_risk": float}
near_miss_events = {}

PHASE1_FRAMES = 8
PHASE1_THRESHOLD = 65
PHASE2_FRAMES = 3
PHASE2_THRESHOLD = 85
RECOVERY_FRAMES = 5
RECOVERY_THRESHOLD = 40
COOLDOWN_FRAMES = 120
STATIONARY_SPEED_THRESH = 0.5
STATIONARY_HISTORY_FRAMES = 30


def is_stationary(track_id, track_history):
    """Check if object is stationary (avg speed < 0.5 px/frame over last 30 frames)."""
    hist = track_history.get(track_id, [])
    if len(hist) < STATIONARY_HISTORY_FRAMES:
        return False
    recent = list(hist[-STATIONARY_HISTORY_FRAMES:])
    total_dist = 0.0
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i-1][0]
        dy = recent[i][1] - recent[i-1][1]
        total_dist += (dx ** 2 + dy ** 2) ** 0.5
    avg_speed = total_dist / (len(recent) - 1)
    return avg_speed < STATIONARY_SPEED_THRESH


# ===================== Bonus B2: Speed Estimation =====================

def estimate_speed(bbox, frame_height, fps):
    """
    Estimate speed in km/h from bounding box position.
    Uses tilted camera perspective projection.
    Camera height = 8m, vertical FOV = 50°, tilt = 30° from horizontal.
    """
    vfov_rad = math.radians(CAMERA_VFOV_DEG)
    tilt_rad = math.radians(CAMERA_TILT_DEG)

    y_center = (bbox[1] + bbox[3]) / 2
    y_norm = y_center / frame_height

    # Ray angle from horizontal (positive = downward).
    # y=0 (top)  → tilt - FOV/2  (closer to horizon → farther ground)
    # y=H (bottom) → tilt + FOV/2 (closer to camera → nearer ground)
    phi = tilt_rad - vfov_rad / 2 + y_norm * vfov_rad
    phi = max(phi, math.radians(2.0))  # clamp to avoid blow-up near horizon

    # Ground distance per pixel: m_per_px = (h * α) / (H * sin²(φ))
    sin_phi = math.sin(phi)
    m_per_px = (CAMERA_HEIGHT_M * vfov_rad) / (frame_height * sin_phi * sin_phi)

    return m_per_px


def compute_speed_kmh(track_id, track_history, bbox, frame_height, fps):
    """
    Compute speed in km/h from track history.
    Uses last 10 frames for velocity estimation.
    """
    hist = track_history.get(track_id, [])
    if len(hist) < 2:
        return 0.0
    
    # Use last 10 frames for velocity
    n = min(10, len(hist))
    recent = list(hist)[-n:]
    
    # Total pixel displacement
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    px_dist = (dx ** 2 + dy ** 2) ** 0.5
    
    # Meters per pixel at current position
    m_per_px = estimate_speed(bbox, frame_height, fps)
    
    # Meters traveled over (n-1) frames
    meters = px_dist * m_per_px
    
    # Speed in m/s
    time_seconds = (n - 1) / fps
    if time_seconds <= 0:
        return 0.0
    speed_ms = meters / time_seconds
    
    # Convert to km/h
    speed_kmh = speed_ms * 3.6
    
    return speed_kmh


# --- Bonus B3 — Trajectory Prediction ---

def predict_position(track_id, track_history, horizon=15):
    """Predict future position using linear velocity from last 5 center positions."""
    if track_id not in track_history:
        return None
    history = track_history[track_id]
    if len(history) < 2:
        return None
    recent = history[-5:] if len(history) >= 5 else history
    if len(recent) < 2:
        return None
    recent_centers = [(int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)) for b in recent]
    mean_vx = 0.0
    mean_vy = 0.0
    for i in range(1, len(recent_centers)):
        mean_vx += recent_centers[i][0] - recent_centers[i - 1][0]
        mean_vy += recent_centers[i][1] - recent_centers[i - 1][1]
    n = len(recent_centers) - 1
    mean_vx /= n
    mean_vy /= n
    last_cx, last_cy = recent_centers[-1]
    pred_cx = int(last_cx + mean_vx * horizon)
    pred_cy = int(last_cy + mean_vy * horizon)
    return (pred_cx, pred_cy)


def draw_trajectory_arrow(frame, current_box, predicted_pos, color=(0, 255, 255)):
    """Draw dashed arrow from current center to predicted position."""
    if predicted_pos is None:
        return
    cx = int((current_box[0] + current_box[2]) / 2)
    cy = int((current_box[1] + current_box[3]) / 2)
    dx = predicted_pos[0] - cx
    dy = predicted_pos[1] - cy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 5:
        return
    # Draw dashed line with arrowhead
    steps = max(int(dist / 10), 1)
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min((i + 1) / steps, 1.0)
        x1 = int(cx + dx * t1)
        y1 = int(cy + dy * t1)
        x2 = int(cx + dx * t2)
        y2 = int(cy + dy * t2)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    # Arrowhead
    arrow_len = 8
    angle = np.arctan2(dy, dx)
    tip = (predicted_pos[0], predicted_pos[1])
    left = (int(tip[0] - arrow_len * np.cos(angle - 0.5)),
            int(tip[1] - arrow_len * np.sin(angle - 0.5)))
    right = (int(tip[0] - arrow_len * np.cos(angle + 0.5)),
             int(tip[1] - arrow_len * np.sin(angle + 0.5)))
    cv2.line(frame, tip, left, color, 1)
    cv2.line(frame, tip, right, color, 1)


def draw_predicted_intersection(frame, box_a, box_b, pred_a, pred_b):
    """Draw magenta line and label if predicted positions are close."""
    if pred_a is None or pred_b is None:
        return
    dx = pred_a[0] - pred_b[0]
    dy = pred_a[1] - pred_b[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > PREDICTED_INTERSECTION_THRESHOLD:
        return
    # Draw dashed magenta line between predicted positions
    color = (255, 0, 255)
    ddx = pred_b[0] - pred_a[0]
    ddy = pred_b[1] - pred_a[1]
    d = (ddx * ddx + ddy * ddy) ** 0.5
    if d < 1:
        return
    steps = max(int(d / 10), 1)
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min((i + 1) / steps, 1.0)
        x1 = int(pred_a[0] + ddx * t1)
        y1 = int(pred_a[1] + ddy * t1)
        x2 = int(pred_a[0] + ddx * t2)
        y2 = int(pred_a[1] + ddy * t2)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    mid_x = int((pred_a[0] + pred_b[0]) / 2)
    mid_y = int((pred_a[1] + pred_b[1]) / 2)
    cv2.putText(frame, "PREDICTED INTERSECTION", (mid_x - 80, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def update_near_miss_events(pair_key, risk, frame_idx):
    """
    Update near-miss event state for a pair.
    Returns (event_triggered, event_state_dict)
    """
    global near_miss_events

    # Decrease cooldown for all pairs
    for k in list(near_miss_events.keys()):
        near_miss_events[k]["cooldown"] = max(0, near_miss_events[k]["cooldown"] - 1)

    if pair_key not in near_miss_events:
        near_miss_events[pair_key] = {
            "phase1_frames": 0,
            "phase2_frames": 0,
            "recovery_frames": 0,
            "cooldown": 0,
            "triggered": False,
            "peak_risk": 0.0,
            "start_frame": frame_idx,
        }

    state = near_miss_events[pair_key]
    state["peak_risk"] = max(state["peak_risk"], risk)
    event_triggered = False

    # If in cooldown, skip
    if state["cooldown"] > 0:
        return False, state

    # Phase 1: Approaching (risk > 65 for 8+ frames)
    if risk > PHASE1_THRESHOLD:
        state["phase1_frames"] += 1
        state["phase2_frames"] = 0
        state["recovery_frames"] = 0
    else:
        state["phase1_frames"] = 0

    # Phase 2: Critical (risk > 85 for 3+ frames) - only if phase1 was satisfied
    if state["phase1_frames"] >= PHASE1_FRAMES and risk > PHASE2_THRESHOLD:
        state["phase2_frames"] += 1
    else:
        if state["phase2_frames"] > 0 and not state["triggered"]:
            state["phase2_frames"] = 0

    # Trigger near-miss event
    if state["phase2_frames"] >= PHASE2_FRAMES and not state["triggered"]:
        event_triggered = True
        state["triggered"] = True
        state["cooldown"] = COOLDOWN_FRAMES
        state["event_frame"] = frame_idx

    # Recovery: risk < 40 for 5+ frames to end event
    if state["triggered"]:
        if risk < RECOVERY_THRESHOLD:
            state["recovery_frames"] += 1
        else:
            state["recovery_frames"] = 0

        if state["recovery_frames"] >= RECOVERY_FRAMES:
            # Event ended, reset for next detection (but keep cooldown)
            state["triggered"] = False
            state["phase1_frames"] = 0
            state["phase2_frames"] = 0
            state["recovery_frames"] = 0

    return event_triggered, state


# ===================== Stage 6: Visuals & Dashboard =====================

# Color constants (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)

# ===================== Bonus B1: Danger Heatmap =====================

HEATMAP_SIGMA = 20.0
HEATMAP_ALPHA = 0.3
HEATMAP_DECAY = 0.995

heatmap = None

def init_heatmap(frame_shape):
    """Initialize float32 heatmap array sized to frame."""
    global heatmap
    h, w = frame_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

def gaussian_2d(x, y, cx, cy, sigma):
    """2D Gaussian function."""
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

def update_heatmap(near_miss_pairs, frame_shape):
    """Add Gaussian blobs at midpoints of near-miss pairs and decay."""
    global heatmap
    if heatmap is None:
        init_heatmap(frame_shape)
    
    h, w = frame_shape[:2]
    
    # Decay existing heatmap
    heatmap *= HEATMAP_DECAY
    
    # Add Gaussian blobs for each near-miss pair
    for p in near_miss_pairs:
        cx_a, cy_a = p["center_a"]
        cx_b, cy_b = p["center_b"]
        mid_x = int((cx_a + cx_b) / 2)
        mid_y = int((cy_a + cy_b) / 2)
        
        if 0 <= mid_x < w and 0 <= mid_y < h:
            # Create Gaussian blob (local region for efficiency)
            radius = int(3 * HEATMAP_SIGMA)
            y_min = max(0, mid_y - radius)
            y_max = min(h, mid_y + radius + 1)
            x_min = max(0, mid_x - radius)
            x_max = min(w, mid_x + radius + 1)
            
            if y_min < y_max and x_min < x_max:
                yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                blob = gaussian_2d(xx, yy, mid_x, mid_y, HEATMAP_SIGMA)
                heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                    heatmap[y_min:y_max, x_min:x_max], blob
                )

def draw_heatmap_overlay(frame):
    """Blend Jet-colormap heatmap onto frame with alpha."""
    global heatmap
    if heatmap is None:
        return
    
    # Normalize heatmap to 0-255
    hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Jet colormap
    hm_colored = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    
    # Blend with alpha
    cv2.addWeighted(hm_colored, HEATMAP_ALPHA, frame, 1.0 - HEATMAP_ALPHA, 0, frame)


def get_box_color(track_id, interacting_pairs, near_miss_pairs):
    """Determine box color based on interaction/risk/near-miss status."""
    # Check if in near-miss pair
    for p in near_miss_pairs:
        if track_id == p["id_a"] or track_id == p["id_b"]:
            return COLOR_RED

    # Check if in high-risk interacting pair (risk > 65)
    for p in interacting_pairs:
        if track_id == p["id_a"] or track_id == p["id_b"]:
            risk = p.get("risk", 0)
            if risk > 65:
                return COLOR_ORANGE
            return COLOR_YELLOW

    return COLOR_GREEN


def draw_distance_line(frame, box_a, box_b, color, risk):
    """Draw line between centers with distance label."""
    cx_a, cy_a = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    cx_b, cy_b = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    cv2.line(frame, (int(cx_a), int(cy_a)), (int(cx_b), int(cy_b)), color, 2)
    mid_x, mid_y = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)
    dist = nearest_edge_distance(box_a, box_b)
    cv2.putText(frame, f"{dist:.0f}px", (mid_x, mid_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_dashboard(frame, near_miss_count, peak_risk, vehicle_count, risk_history):
    """Draw dashboard overlay in top-left corner."""
    h, w = frame.shape[:2]

    # Dashboard background
    dash_h = 140
    dash_w = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + dash_w, 10 + dash_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Text lines
    y = 35
    cv2.putText(frame, f"Near-Miss Count: {near_miss_count}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30
    level = risk_level(peak_risk)
    cv2.putText(frame, f"Risk Level: {level} (peak={peak_risk:.0f})", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30

    # Timeline bar: last 100 frames of peak risk
    timeline_w = dash_w - 20
    timeline_h = 30
    tx, ty = 20, y
    cv2.rectangle(frame, (tx, ty), (tx + timeline_w, ty + timeline_h), (50, 50, 50), -1)

    if risk_history:
        n = min(100, len(risk_history))
        recent = risk_history[-n:]
        for i, r in enumerate(recent):
            x = tx + int(i * timeline_w / max(1, n - 1))
            bar_h = int(r / 100 * timeline_h)
            # Color based on risk level
            if r >= 85:
                bar_color = COLOR_RED
            elif r >= 65:
                bar_color = COLOR_ORANGE
            elif r >= 30:
                bar_color = COLOR_YELLOW
            else:
                bar_color = COLOR_GREEN
            cv2.line(frame, (x, ty + timeline_h), (x, ty + timeline_h - bar_h), bar_color, 2)

    # Border
    cv2.rectangle(frame, (tx, ty), (tx + timeline_w, ty + timeline_h), (100, 100, 100), 1)


def draw_near_miss_overlay(frame, pair):
    """Draw NEAR-MISS alert overlay on the pair."""
    box_a, box_b = pair["box_a"], pair["box_b"]
    cx_a, cy_a = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    cx_b, cy_b = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    mid_x, mid_y = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)

    # Red box around the pair
    x1 = int(min(box_a[0], box_b[0]))
    y1 = int(min(box_a[1], box_b[1]))
    x2 = int(max(box_a[2], box_b[2]))
    y2 = int(max(box_a[3], box_b[3]))
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 3)

    # Alert text
    text = "⚠ NEAR-MISS"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text, (mid_x - tw // 2, mid_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="yolo26m.pt", help="yolo26n.pt (CPU) or yolo26m.pt (GPU)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--output", default="output_annotated.mp4")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing")
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # track_id -> deque of (cx, cy) center positions, oldest first, newest last
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    # track_id -> class name, assigned once and kept stable
    track_class = {}

    # Stage 5: Near-miss event state per pair
    near_miss_states = defaultdict(lambda: {
        "phase1_frames": 0, "phase2_frames": 0, "recovery_frames": 0,
        "triggered": False, "cooldown": 0, "event_frame": -1,
        "peak_risk": 0
    })
    near_miss_count = 0
    risk_history = []  # per-frame peak risk for timeline
    near_miss_events = []  # for bonus B4 report

    if args.show:
        cv2.namedWindow("Near-Miss Traffic Analytics", cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            classes=ROAD_USER_CLASSES,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        interacting_pairs = []
        near_miss_pairs = []
        frame_peak_risk = 0

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            # Stage 3: Compute interacting pairs
            interacting_pairs = compute_interacting_pairs(xyxy, ids, cls_ids, track_history)

            # Stage 4: Compute risk for each interacting pair
            for p in interacting_pairs:
                risk = risk_score(p, track_history)
                p["risk"] = risk
                frame_peak_risk = max(frame_peak_risk, risk)

                # Stage 5: Update near-miss state machine
                pair_key = (min(p["id_a"], p["id_b"]), max(p["id_a"], p["id_b"]))
                event_triggered, state = update_near_miss_events(pair_key, risk, frame_idx)
                p["near_miss_state"] = state

                if event_triggered:
                    near_miss_count += 1
                    p["near_miss_triggered"] = True
                    near_miss_pairs.append(p)
                    # Record event for bonus B4
                    near_miss_events.append({
                        "frame": frame_idx,
                        "pair_ids": [p["id_a"], p["id_b"]],
                        "peak_risk": risk,
                        "cls_a": p["cls_a"],
                        "cls_b": p["cls_b"],
                    })
                elif state["triggered"]:
                    near_miss_pairs.append(p)

            risk_history.append(frame_peak_risk)
            if len(risk_history) > 100:
                risk_history.pop(0)

            # Create sets for quick lookup
            interacting_ids = set()
            near_miss_ids = set()
            for p in interacting_pairs:
                interacting_ids.add(p["id_a"])
                interacting_ids.add(p["id_b"])
            for p in near_miss_pairs:
                near_miss_ids.add(p["id_a"])
                near_miss_ids.add(p["id_b"])

            for box, track_id, cls_id, conf in zip(xyxy, ids, cls_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                track_history[track_id].append((cx, cy))
                track_class[track_id] = CLASS_NAMES.get(int(cls_id), str(cls_id))

                # Stage 6: Determine box color
                color = get_box_color(track_id, interacting_pairs, near_miss_pairs)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"#{track_id} {track_class[track_id]} {conf:.2f}"
                cv2.putText(
                    frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

                # Bonus B2: Display speed in km/h
                speed_kmh = compute_speed_kmh(track_id, track_history, box, height, fps)
                speed_label = f"{speed_kmh:.1f} km/h"
                cv2.putText(
                    frame, speed_label, (int(x1), int(y2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

            # Bonus B3: Draw trajectory arrows for all tracked objects
            for tid, box in zip(track_ids, boxes):
                pred_pos = predict_position(tid, track_history, TRAJECTORY_HORIZON)
                draw_trajectory_arrow(frame, box, pred_pos, (0, 255, 255))

            # Stage 3/6: Draw distance lines with risk-based color
            for p in interacting_pairs:
                risk = p.get("risk", 0)
                if risk >= 85:
                    line_color = COLOR_RED
                elif risk >= 65:
                    line_color = COLOR_ORANGE
                else:
                    line_color = COLOR_YELLOW
                draw_distance_line(frame, p["box_a"], p["box_b"], line_color, risk)

                # Bonus B3: Check predicted intersection for high-risk pairs
                if risk >= 65:
                    pred_a = predict_position(p["id_a"], track_history, TRAJECTORY_HORIZON)
                    pred_b = predict_position(p["id_b"], track_history, TRAJECTORY_HORIZON)
                    draw_predicted_intersection(frame, p["box_a"], p["box_b"], pred_a, pred_b)

            # Stage 6: Draw near-miss overlay
            for p in near_miss_pairs:
                draw_near_miss_overlay(frame, p)

            # Bonus B1: Update and draw danger heatmap
            update_heatmap(near_miss_pairs, frame.shape)
            draw_heatmap_overlay(frame)

        else:
            risk_history.append(0)
            if len(risk_history) > 100:
                risk_history.pop(0)

        # Stage 6: Draw dashboard
        vehicle_count = sum(1 for c in track_class.values() if c in VEHICLE_CLASSES)
        draw_dashboard(frame, near_miss_count, frame_peak_risk, vehicle_count, risk_history)

        writer.write(frame)

        if frame_idx % 30 == 0:
            if total_frames:
                pct = frame_idx / total_frames * 100
                print(f"  ...frame {frame_idx}/{total_frames} ({pct:.1f}%)")
            else:
                print(f"  ...frame {frame_idx} (total unknown)")

        if args.show:
            h, w = frame.shape[:2]
            scale = min(DISPLAY_MAX_DIM / max(h, w), 1.0)
            preview = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            cv2.imshow("Near-Miss Traffic Analytics", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. {frame_idx} frames processed. Output saved to {args.output}")
    print(f"Total Near-Miss Events: {near_miss_count}")


if __name__ == "__main__":
    main()