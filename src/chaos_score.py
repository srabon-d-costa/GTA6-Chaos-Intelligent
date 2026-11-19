
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = ROOT / "outputs/reports/activity_metrics.csv"
DETECTION_PATH = ROOT / "outputs/videos/detection_results.csv"

OUTPUT_DIR = ROOT / "outputs/reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD EXISTING ANALYSIS DATA
# ==========================================

print("\nVICE CITY CHAOS SCORE ENGINE")
print("=" * 45)

metrics = pd.read_csv(METRICS_PATH)
detections = pd.read_csv(DETECTION_PATH)

required_columns = [
    "frame",
    "motion_score",
    "crowd_score",
    "vehicle_score",
    "timestamp_sec"
]

for col in required_columns:
    if col not in metrics.columns:
        raise ValueError(f"Missing column: {col}")

metrics["frame"] = metrics["frame"].astype(int)
detections["frame"] = detections["frame"].astype(int)

detections["class"] = (
    detections["class"].astype(str).str.lower()
)

print("Activity metrics loaded:", len(metrics))
print("Detection records loaded:", len(detections))

# ==========================================
# 3. SCENE COMPLEXITY ANALYSIS
# ==========================================

SUPPORTED_CLASSES = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
]

# Count unique object categories per frame

class_diversity = (
    detections.groupby("frame")["class"]
    .nunique()
)

metrics["class_diversity"] = (
    metrics["frame"]
    .map(class_diversity)
    .fillna(0)
)

# Convert category diversity into 0-100 score

metrics["scene_complexity_score"] = (
    metrics["class_diversity"]
    / len(SUPPORTED_CLASSES)
    * 100
).clip(0, 100)

# ==========================================
# 4. PREPARE NORMALIZED COMPONENTS
# ==========================================

components = [
    "motion_score",
    "crowd_score",
    "vehicle_score",
    "scene_complexity_score"
]

for col in components:

    metrics[col] = pd.to_numeric(
        metrics[col],
        errors="coerce"
    ).fillna(0).clip(0, 100)

# ==========================================
# 5. CUSTOM CHAOS SCORE ALGORITHM
# ==========================================

print("\nCalculating chaos scores...")

metrics["chaos_score_raw"] = (

    0.35 * metrics["motion_score"]

    + 0.30 * metrics["crowd_score"]

    + 0.25 * metrics["vehicle_score"]

    + 0.10 * metrics["scene_complexity_score"]

)

# Smooth short-term fluctuations

metrics["chaos_score"] = (
    metrics["chaos_score_raw"]
    .rolling(
        window=5,
        min_periods=1,
        center=True
    )
    .mean()
    .clip(0, 100)
)

# ==========================================
# 6. CHAOS LEVEL CLASSIFICATION
# ==========================================

def classify_chaos(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Very High"


metrics["chaos_level"] = (
    metrics["chaos_score"]
    .apply(classify_chaos)
)

# ==========================================
# 7. SAVE FRAME-LEVEL RESULTS
# ==========================================

OUTPUT_CSV = OUTPUT_DIR / "chaos_scores.csv"

metrics.to_csv(OUTPUT_CSV, index=False)

# ==========================================
# 8. FIND PEAK CHAOS MOMENTS
# ==========================================

peak_index = metrics["chaos_score"].idxmax()

peak_row = metrics.loc[peak_index]

average_chaos = metrics["chaos_score"].mean()

maximum_chaos = metrics["chaos_score"].max()

minimum_chaos = metrics["chaos_score"].min()

# Save top 5 highest activity moments

top_moments = (
    metrics.sort_values(
        "chaos_score",
        ascending=False
    )
    .head(5)
    [[
        "timestamp_sec",
        "chaos_score",
        "chaos_level",
        "motion_score",
        "crowd_score",
        "vehicle_score"
    ]]
)

top_moments.to_csv(
    OUTPUT_DIR / "peak_chaos_moments.csv",
    index=False
)

# ==========================================
# 9. CREATE CHAOS SCORE GRAPH
# ==========================================

plt.figure(figsize=(13, 6))

plt.plot(
    metrics["timestamp_sec"],
    metrics["chaos_score"],
    linewidth=2,
    label="Chaos Score"
)

plt.axhline(
    25,
    linestyle="--",
    alpha=0.5,
    label="Low / Moderate"
)

plt.axhline(
    50,
    linestyle="--",
    alpha=0.5,
    label="Moderate / High"
)

plt.axhline(
    75,
    linestyle="--",
    alpha=0.5,
    label="High / Very High"
)

plt.scatter(
    peak_row["timestamp_sec"],
    peak_row["chaos_score"],
    s=90,
    label="Peak Chaos"
)

plt.xlabel("Video Time (seconds)")
plt.ylabel("Chaos Score (0-100)")

plt.title("VICE CITY - CHAOS INTELLIGENCE ANALYSIS")

plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

GRAPH_PATH = OUTPUT_DIR / "chaos_score_over_time.png"

plt.savefig(GRAPH_PATH, dpi=180)
plt.close()

# ==========================================
# 10. GENERATE SUMMARY REPORT
# ==========================================

summary = f"""
========================================
VICE CITY CHAOS INTELLIGENCE REPORT
========================================

Total analyzed intervals: {len(metrics)}

Average Chaos Score:
{average_chaos:.2f}/100

Maximum Chaos Score:
{maximum_chaos:.2f}/100

Minimum Chaos Score:
{minimum_chaos:.2f}/100

Peak Chaos Timestamp:
{peak_row['timestamp_sec']:.2f} seconds

Peak Chaos Level:
{peak_row['chaos_level']}

Average Motion Score:
{metrics['motion_score'].mean():.2f}

Average Crowd Score:
{metrics['crowd_score'].mean():.2f}

Average Vehicle Score:
{metrics['vehicle_score'].mean():.2f}

========================================
ANALYSIS COMPLETED
========================================
"""

SUMMARY_PATH = OUTPUT_DIR / "chaos_summary.txt"

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    file.write(summary)

print(summary)

print("OUTPUT FILES:")
print("Chaos CSV:", OUTPUT_CSV)
print("Chaos graph:", GRAPH_PATH)
print("Peak moments:", OUTPUT_DIR / "peak_chaos_moments.csv")
print("Summary:", SUMMARY_PATH)

print("\nVICE CITY CHAOS ENGINE FINISHED!")
