"""
Near-Miss Traffic Analytics — Stage 1 & 2
Object Detection + Tracking (ByteTrack)

Usage:
    python traffic_near_miss.py --video traffic.mp4 --model yolo26m.pt --show
"""

import argparse
from collections import defaultdict, deque

import cv2
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None  # may be 0/unknown for some containers

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # track_id -> deque of (cx, cy) center positions, oldest first, newest last
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    # track_id -> class name, assigned once and kept stable
    track_class = {}

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

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(xyxy, ids, cls_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                track_history[track_id].append((cx, cy))
                track_class[track_id] = CLASS_NAMES.get(int(cls_id), str(cls_id))

                # Stage 1+2 only — every box drawn green for now.
                # Stage 3-5 will recolor based on proximity / risk / near-miss status.
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"#{track_id} {track_class[track_id]} {conf:.2f}"
                cv2.putText(
                    frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

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


if __name__ == "__main__":
    main()