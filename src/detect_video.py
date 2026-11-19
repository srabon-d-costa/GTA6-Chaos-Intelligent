
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "sample_videos" / "Game.mp4"
OUTPUT_DIR = ROOT / "outputs" / "videos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "Game_detected.mp4"
OUTPUT_CSV = OUTPUT_DIR / "detection_results.csv"

# =====================================
# 2. LOAD YOLO MODEL
# =====================================

print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

# COCO classes:
# 0 person, 1 bicycle, 2 car,
# 3 motorcycle, 5 bus, 7 truck

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

print("Model loaded successfully!")

# =====================================
# 3. OPEN VIDEO
# =====================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30

print(f"Video FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Total frames: {total_frames}")

# =====================================
# 4. VIDEO WRITER
# =====================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video!")

# =====================================
# 5. DETECTION AND TRACKING
# =====================================

frame_number = 0
unique_tracks = {}

class_names = model.names

with open(OUTPUT_CSV, "w", newline="") as csvfile:

    csv_writer = csv.writer(csvfile)

    csv_writer.writerow([
        "frame",
        "timestamp",
        "class",
        "track_id",
        "confidence",
        "x1", "y1", "x2", "y2"
    ])

    print("\nStarting video analysis...\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Run YOLO detection + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Draw detections
        annotated_frame = result.plot(
            line_width=2,
            font_size=20
        )

        # Extract detection information
        if result.boxes is not None:

            boxes = result.boxes

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = class_names[class_id]

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                    unique_tracks.setdefault(
                        class_name, set()
                    ).add(track_id)

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                timestamp = frame_number / fps

                csv_writer.writerow([
                    frame_number,
                    round(timestamp, 2),
                    class_name,
                    track_id,
                    round(confidence, 3),
                    x1, y1, x2, y2
                ])

        # Save annotated frame
        writer.write(annotated_frame)

        if frame_number % 30 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

# =====================================
# 6. RELEASE RESOURCES
# =====================================

cap.release()
writer.release()

# =====================================
# 7. FINAL SUMMARY
# =====================================

print("\n========== ANALYSIS COMPLETE ==========")

print(f"Processed frames: {frame_number}")

print("\nUnique tracked objects:")

for class_name, ids in unique_tracks.items():
    print(f"{class_name}: {len(ids)}")

print(f"\nProcessed video: {OUTPUT_VIDEO}")
print(f"Detection CSV: {OUTPUT_CSV}")

print("\nVICE CITY VIDEO ANALYSIS FINISHED!")
