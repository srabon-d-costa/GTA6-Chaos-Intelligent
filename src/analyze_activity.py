
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "sample_videos" / "Game.mp4"

DETECTION_CSV = (
    ROOT / "outputs" / "videos" / "detection_results.csv"
)

HEATMAP_DIR = ROOT / "outputs" / "heatmaps"
REPORT_DIR = ROOT / "outputs" / "reports"

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Analyze every 5th frame for motion
FRAME_STEP = 5

# Resize frames for faster optical flow
ANALYSIS_WIDTH = 640

VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
]

# ==========================================
# 2. CHECK INPUT FILES
# ==========================================

print("\nVICE CITY ACTIVITY ANALYZER")
print("=" * 45)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not DETECTION_CSV.exists():
    raise FileNotFoundError(
        f"Detection CSV not found: {DETECTION_CSV}"
    )

detections = pd.read_csv(DETECTION_CSV)

if detections.empty:
    raise ValueError("Detection CSV contains no detections.")

required_columns = [
    "frame", "class", "x1", "y1", "x2", "y2"
]

for column in required_columns:
    if column not in detections.columns:
        raise ValueError(f"Missing CSV column: {column}")

detections["frame"] = pd.to_numeric(
    detections["frame"], errors="coerce"
)

detections = detections.dropna(subset=["frame"])

detections["frame"] = detections["frame"].astype(int)

detections["class"] = (
    detections["class"].astype(str).str.lower()
)

print("Detection data loaded successfully!")
print("Detection records:", len(detections))

# ==========================================
# 3. OPEN ORIGINAL VIDEO
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video resolution:", width, "x", height)
print("Video FPS:", fps)

analysis_height = int(
    height * ANALYSIS_WIDTH / width
)

# ==========================================
# 4. OPTICAL FLOW ANALYSIS
# ==========================================

print("\nStarting motion analysis...")

frame_numbers = []
motion_values = []

previous_gray = None
frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Sample every FRAME_STEP frames
    if (frame_number - 1) % FRAME_STEP != 0:
        continue

    resized = cv2.resize(
        frame,
        (ANALYSIS_WIDTH, analysis_height)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    if previous_gray is None:

        motion_magnitude = 0.0

    else:

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        motion_magnitude = float(
            np.mean(magnitude)
        )

    frame_numbers.append(frame_number)
    motion_values.append(motion_magnitude)

    previous_gray = gray

    if frame_number % 150 == 1:
        print("Motion analyzed:", frame_number)

cap.release()

print("Motion analysis completed!")

# ==========================================
# 5. CROWD AND VEHICLE ANALYSIS
# ==========================================

print("\nAnalyzing crowd and vehicles...")

# Count detected objects in each frame
count_table = (
    detections.groupby(["frame", "class"])
    .size()
    .unstack(fill_value=0)
)

# Ensure all expected columns exist
EXPECTED_CLASSES = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
]

for class_name in EXPECTED_CLASSES:

    if class_name not in count_table.columns:
        count_table[class_name] = 0

# Align counts with sampled video frames
sampled_counts = count_table.reindex(
    frame_numbers,
    fill_value=0
)

# Pedestrian counts
person_counts = sampled_counts["person"].values

# Vehicle counts
vehicle_counts = (
    sampled_counts[VEHICLE_CLASSES]
    .sum(axis=1)
    .values
)

# Total detected objects
total_counts = person_counts + vehicle_counts

# ==========================================
# 6. NORMALIZE METRICS
# ==========================================

def normalize_score(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.zeros(0)

    reference = np.percentile(values, 95)

    if reference <= 0:
        return np.zeros_like(values)

    return np.clip(
        (values / reference) * 100,
        0,
        100
    )


motion_scores = normalize_score(motion_values)

crowd_scores = normalize_score(person_counts)

vehicle_scores = normalize_score(vehicle_counts)

# ==========================================
# 7. CREATE ACTIVITY DATAFRAME
# ==========================================

metrics = pd.DataFrame({

    "frame": frame_numbers,

    "timestamp_sec": [
        (f - 1) / fps for f in frame_numbers
    ],

    "motion_magnitude": motion_values,

    "motion_score": motion_scores,

    "person_count": person_counts,

    "crowd_score": crowd_scores,

    "vehicle_count": vehicle_counts,

    "vehicle_score": vehicle_scores,

    "total_objects": total_counts

})

# Save CSV
METRICS_CSV = REPORT_DIR / "activity_metrics.csv"

metrics.to_csv(METRICS_CSV, index=False)

print("Activity metrics saved!")

# ==========================================
# 8. GENERATE CROWD AND VEHICLE HEATMAPS
# ==========================================

print("\nGenerating heatmaps...")

person_heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

vehicle_heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

for _, row in detections.iterrows():

    try:

        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cx = np.clip(cx, 0, width - 1)
        cy = np.clip(cy, 0, height - 1)

        class_name = row["class"]

        if class_name == "person":

            cv2.circle(
                person_heatmap,
                (cx, cy),
                15,
                1,
                -1
            )

        elif class_name in VEHICLE_CLASSES:

            cv2.circle(
                vehicle_heatmap,
                (cx, cy),
                15,
                1,
                -1
            )

    except (ValueError, TypeError):
        continue


def save_heatmap(data, output_path, title):

    blurred = cv2.GaussianBlur(
        data,
        (0, 0),
        sigmaX=18
    )

    if np.max(blurred) > 0:

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    else:

        normalized = np.zeros(
            data.shape,
            dtype=np.uint8
        )

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO
    )

    cv2.putText(
        colored,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imwrite(str(output_path), colored)


save_heatmap(
    person_heatmap,
    HEATMAP_DIR / "crowd_heatmap.png",
    "CROWD DENSITY HEATMAP"
)

save_heatmap(
    vehicle_heatmap,
    HEATMAP_DIR / "vehicle_heatmap.png",
    "VEHICLE ACTIVITY HEATMAP"
)

print("Heatmaps generated successfully!")

# ==========================================
# 9. ACTIVITY OVER TIME GRAPH
# ==========================================

print("\nGenerating activity graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    metrics["timestamp_sec"],
    metrics["motion_score"],
    label="Motion Intensity",
    linewidth=2
)

plt.plot(
    metrics["timestamp_sec"],
    metrics["crowd_score"],
    label="Crowd Activity",
    linewidth=2
)

plt.plot(
    metrics["timestamp_sec"],
    metrics["vehicle_score"],
    label="Vehicle Activity",
    linewidth=2
)

plt.xlabel("Video Time (seconds)")

plt.ylabel("Relative Activity Score (0-100)")

plt.title("Vice City - Activity Analysis Over Time")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

GRAPH_PATH = REPORT_DIR / "activity_over_time.png"

plt.savefig(
    GRAPH_PATH,
    dpi=160
)

plt.close()

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

peak_motion = metrics.loc[
    metrics["motion_score"].idxmax()
]

print("\n" + "=" * 45)

print("VICE CITY ACTIVITY ANALYSIS COMPLETE!")

print("=" * 45)

print(
    "Average detected pedestrians:",
    round(metrics["person_count"].mean(), 2)
)

print(
    "Average detected vehicles:",
    round(metrics["vehicle_count"].mean(), 2)
)

print(
    "Average motion score:",
    round(metrics["motion_score"].mean(), 2)
)

print(
    "Peak motion timestamp:",
    round(peak_motion["timestamp_sec"], 2),
    "seconds"
)

print("\nOUTPUT FILES:")

print("Metrics:", METRICS_CSV)

print("Crowd heatmap:", HEATMAP_DIR / "crowd_heatmap.png")

print("Vehicle heatmap:", HEATMAP_DIR / "vehicle_heatmap.png")

print("Activity graph:", GRAPH_PATH)

print("\nVICE CITY INTELLIGENCE MODULE FINISHED!")
