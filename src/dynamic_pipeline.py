
import cv2
import numpy as np
import pandas as pd
import subprocess
import imageio_ffmpeg
import torch

from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

VEHICLE_CLASSES = {
    "car", "motorcycle", "bus", "truck", "bicycle"
}

MODEL_NAME = "yolo11n.pt"


# ==========================================
# 2. NORMALIZATION
# ==========================================

def normalize_score(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.zeros(0)

    reference = np.percentile(values, 95)

    if reference <= 0:
        return np.zeros_like(values)

    return np.clip(
        values / reference * 100,
        0,
        100
    )


def classify_chaos(score):

    if score < 25:
        return "Low"
    elif score < 50:
        return "Moderate"
    elif score < 75:
        return "High"
    return "Very High"


# ==========================================
# 3. HEATMAP GENERATOR
# ==========================================

def save_heatmap(data, path, title):

    blurred = cv2.GaussianBlur(
        data,
        (0, 0),
        sigmaX=18
    )

    if blurred.max() > 0:

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
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imwrite(str(path), colored)


# ==========================================
# 4. MAIN DYNAMIC VIDEO ANALYZER
# ==========================================

def analyze_video(
    video_path,
    output_dir,
    progress_callback=None,
    frame_stride=1,
    target_fps=60
):

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    heatmap_dir = output_dir / "heatmaps"
    report_dir = output_dir / "reports"
    video_dir = output_dir / "videos"

    for folder in [heatmap_dir, report_dir, video_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")

    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")

    target_fps = float(target_fps)
    target_fps_text = f"{target_fps:g}"

    # --------------------------------------
    # Load model
    # --------------------------------------

    if progress_callback:
        progress_callback(0.02, "Loading YOLO11 model...")

    model = YOLO(MODEL_NAME)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # --------------------------------------
    # Open video
    # --------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not determine video frame count.")

    duration = total_frames / fps

    # --------------------------------------
    # Output video writer
    # --------------------------------------

    raw_video = video_dir / "annotated_raw.mp4"

    output_fps = max(fps / frame_stride, 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(raw_video),
        fourcc,
        output_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    # --------------------------------------
    # Initialize variables
    # --------------------------------------

    detections = []
    activity_rows = []

    person_heatmap = np.zeros(
        (height, width),
        dtype=np.float32
    )

    vehicle_heatmap = np.zeros(
        (height, width),
        dtype=np.float32
    )

    previous_gray = None

    frame_number = 0
    processed_frames = 0

    # --------------------------------------
    # Process video
    # --------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            if (frame_number - 1) % frame_stride != 0:
                continue
            timestamp = (frame_number - 1) / fps

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=0.25,
                imgsz=640,
                device=device,
                verbose=False
            )

            result = results[0]

            annotated = result.plot(
                line_width=2,
                font_size=20
            )

            frame_classes = set()

            person_count = 0
            vehicle_count = 0

            # ----------------------------------
            # Extract detections
            # ----------------------------------

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())

                    class_name = model.names[class_id]

                    confidence = float(box.conf[0].item())

                    track_id = None

                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].cpu().numpy()
                    )

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    cx = np.clip(cx, 0, width - 1)
                    cy = np.clip(cy, 0, height - 1)

                    

                    detections.append({
                        "frame": frame_number,
                        "timestamp_sec": timestamp,
                        "class": class_name,
                        "track_id": track_id,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                    frame_classes.add(class_name)

                    if class_name == "person":

                        person_count += 1

                        cv2.circle(
                            person_heatmap,
                            (cx, cy),
                            15,
                            1,
                            -1
                        )

                    elif class_name in VEHICLE_CLASSES:

                        vehicle_count += 1

                        cv2.circle(
                            vehicle_heatmap,
                            (cx, cy),
                            15,
                            1,
                            -1
                        )

            # ----------------------------------
            # Motion analysis
            # ----------------------------------

            resized = cv2.resize(
                frame,
                (
                    640,
                    int(height * 640 / width)
                )
            )

            gray = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2GRAY
            )

            if previous_gray is None:

                motion = 0.0

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

                magnitude, _ = cv2.cartToPolar(
                    flow[..., 0],
                    flow[..., 1]
                )

                motion = float(np.mean(magnitude))

            previous_gray = gray

            # ----------------------------------
            # Store activity
            # ----------------------------------

            activity_rows.append({

                "frame": frame_number,

                "timestamp_sec": timestamp,

                "motion_magnitude": motion,

                "person_count": person_count,

                "vehicle_count": vehicle_count,

                "total_objects": person_count + vehicle_count,

                "class_diversity": len(frame_classes)

            })

            writer.write(annotated)

            processed_frames += 1

            # ----------------------------------
            # Progress reporting
            # ----------------------------------

            if progress_callback:

                progress = min(
                    frame_number / total_frames,
                    1.0
                )

                progress_callback(
                    progress * 0.85,
                    f"Analyzing frame {frame_number}/{total_frames}"
                )

    finally:

        cap.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError("No frames were processed.")

    # ======================================
    # 5. DATAFRAME GENERATION
    # ======================================

    detection_df = pd.DataFrame(
        detections,
        columns=[
            "frame",
            "timestamp_sec",
            "class",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

    metrics = pd.DataFrame(activity_rows)

    # ======================================
    # 6. ACTIVITY NORMALIZATION
    # ======================================

    metrics["motion_score"] = normalize_score(
        metrics["motion_magnitude"]
    )

    metrics["crowd_score"] = normalize_score(
        metrics["person_count"]
    )

    metrics["vehicle_score"] = normalize_score(
        metrics["vehicle_count"]
    )

    metrics["scene_complexity_score"] = (
        metrics["class_diversity"] / len(TARGET_CLASSES) * 100
    ).clip(0, 100)

    # ======================================
    # 7. CHAOS SCORE ENGINE
    # ======================================

    metrics["chaos_score_raw"] = (

        0.35 * metrics["motion_score"]

        + 0.30 * metrics["crowd_score"]

        + 0.25 * metrics["vehicle_score"]

        + 0.10 * metrics["scene_complexity_score"]

    )

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

    metrics["chaos_level"] = (
        metrics["chaos_score"]
        .apply(classify_chaos)
    )

    # ======================================
    # 8. SAVE ANALYSIS DATA
    # ======================================

    detection_df.to_csv(
        output_dir / "detection_results.csv",
        index=False
    )

    metrics.to_csv(
        report_dir / "activity_metrics.csv",
        index=False
    )

    metrics.to_csv(
        report_dir / "chaos_scores.csv",
        index=False
    )

    peaks = metrics.nlargest(
        5,
        "chaos_score"
    )

    peaks.to_csv(
        report_dir / "peak_chaos_moments.csv",
        index=False
    )

    # ======================================
    # 9. GENERATE HEATMAPS
    # ======================================

    save_heatmap(
        person_heatmap,
        heatmap_dir / "crowd_heatmap.png",
        "CROWD DENSITY"
    )

    save_heatmap(
        vehicle_heatmap,
        heatmap_dir / "vehicle_heatmap.png",
        "VEHICLE ACTIVITY"
    )

    # ======================================
    # 10. GENERATE 60 FPS H264 VIDEO
    # ======================================

    if progress_callback:
        progress_callback(0.88, "Generating smooth target-FPS video...")

    web_video = video_dir / "annotated_video_web.mp4"

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # The raw annotated video preserves the original duration.
    # If frames were sampled, its FPS is reduced by frame_stride.
    source_output_fps = float(output_fps)

    even_dimensions = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if source_output_fps < target_fps - 0.01:

        video_filter = (
            f"minterpolate=fps={target_fps_text}:"
            "mi_mode=mci:"
            "mc_mode=obmc:"
            "me_mode=bilat:"
            "me=epzs:"
            "search_param=16:"
            "vsbmc=0,"
            f"{even_dimensions}"
        )

        interpolation_mode = "Motion-compensated interpolation"

    elif source_output_fps > target_fps + 0.01:

        video_filter = (
            f"fps={target_fps_text},{even_dimensions}"
        )

        interpolation_mode = "Downsampled to target FPS"

    else:

        video_filter = even_dimensions

        interpolation_mode = "Native source FPS"

    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-stats_period", "0.5",
        "-progress", "pipe:1",
        "-nostats",
        "-i", str(raw_video),
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(web_video)
    ]

    # Read FFmpeg's progress stream so the Streamlit progress bar
    # advances during conversion instead of remaining at 88%.
    def run_ffmpeg_with_progress(cmd, expected_duration, fallback=False):

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        log_tail = []

        try:
            for line in process.stdout:
                line = line.strip()

                if line:
                    log_tail.append(line)
                    if len(log_tail) > 40:
                        log_tail.pop(0)

                if line.startswith("out_time="):
                    time_value = line.split("=", 1)[1]

                    if time_value != "N/A" and expected_duration > 0:
                        try:
                            hours, minutes, seconds = time_value.split(":")
                            elapsed = (
                                float(hours) * 3600
                                + float(minutes) * 60
                                + float(seconds)
                            )

                            fraction = min(
                                max(elapsed / expected_duration, 0.0),
                                1.0
                            )

                            overall_progress = 0.88 + (0.11 * fraction)

                            if progress_callback:
                                label = (
                                    "Converting video "
                                    if fallback
                                    else "Interpolating frames "
                                )
                                progress_callback(
                                    overall_progress,
                                    f"{label}{fraction * 100:.0f}%"
                                )

                        except (ValueError, TypeError):
                            pass

                elif line == "progress=end":
                    if progress_callback:
                        progress_callback(
                            0.99,
                            "Finalizing 60 FPS video..."
                        )

            return_code = process.wait()

        finally:
            if process.stdout:
                process.stdout.close()

            if process.poll() is None:
                process.kill()
                process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                cmd,
                output="\\n".join(log_tail)
            )

    annotated_duration = (
        processed_frames / source_output_fps
        if source_output_fps > 0
        else duration
    )

    try:
        run_ffmpeg_with_progress(
            command,
            expected_duration=annotated_duration
        )

    except subprocess.CalledProcessError as error:

        if source_output_fps < target_fps - 0.01:

            print("WARNING: Motion interpolation failed.")
            print("FFmpeg output:", error.output)
            print("Falling back to standard FPS conversion.")

            fallback_filter = (
                f"fps={target_fps_text},{even_dimensions}"
            )

            fallback_command = [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-stats_period", "0.5",
                "-progress", "pipe:1",
                "-nostats",
                "-i", str(raw_video),
                "-vf", fallback_filter,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(web_video)
            ]

            run_ffmpeg_with_progress(
                fallback_command,
                expected_duration=annotated_duration,
                fallback=True
            )

            interpolation_mode = (
                "Frame-rate conversion fallback; "
                "not motion-compensated interpolation"
            )

        else:
            raise RuntimeError(
                f"Video conversion failed: {error.output}"
            ) from error

    if not web_video.exists() or web_video.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create a valid output video.")

    verify_cap = cv2.VideoCapture(str(web_video))
    verified_output_fps = verify_cap.get(cv2.CAP_PROP_FPS)
    verify_cap.release()

    if verified_output_fps <= 0:
        verified_output_fps = target_fps

    print("VIDEO CONVERSION COMPLETED!")
    print(f"Source annotated FPS: {source_output_fps:.2f}")
    print(f"Target FPS: {target_fps:.2f}")
    print(f"Verified output FPS: {verified_output_fps:.2f}")
    print("Output mode:", interpolation_mode)
    print("Output video:", web_video)

    # ======================================
    # 11. GENERATE SUMMARY
    # ======================================

    peak = metrics.loc[
        metrics["chaos_score"].idxmax()
    ]

    summary = f"""
VICE CITY CHAOS INTELLIGENCE REPORT
========================================

Input Video: {video_path.name}

Duration: {duration:.2f} seconds

Original Frames: {total_frames}

Processed Frames: {processed_frames}

Total Detection Records: {len(detection_df)}

Unique Tracked Objects:
{detection_df.dropna(subset=["track_id"]).drop_duplicates(
    subset=["class", "track_id"]
).shape[0]}

Average Chaos Score:
{metrics["chaos_score"].mean():.2f}/100

Maximum Chaos Score:
{metrics["chaos_score"].max():.2f}/100

Minimum Chaos Score:
{metrics["chaos_score"].min():.2f}/100

Peak Chaos Timestamp:
{peak["timestamp_sec"]:.2f} seconds

Peak Chaos Level:
{peak["chaos_level"]}

Average Motion:
{metrics["motion_score"].mean():.2f}

Average Crowd:
{metrics["crowd_score"].mean():.2f}

Average Vehicle:
{metrics["vehicle_score"].mean():.2f}
"""

    summary_path = report_dir / "chaos_summary.txt"

    summary_path.write_text(
        summary,
        encoding="utf-8"
    )

    if progress_callback:
        progress_callback(1.0, "Analysis completed!")

    return {

        "video": str(web_video),

        "raw_video": str(raw_video),

        "output_fps": float(verified_output_fps),

        "target_fps": float(target_fps),

        "source_output_fps": float(source_output_fps),

        "interpolation_mode": interpolation_mode,

        "detections": detection_df,

        "metrics": metrics,

        "chaos": metrics,

        "peaks": peaks,

        "summary": summary,

        "summary_path": str(summary_path),

        "crowd_heatmap": str(
            heatmap_dir / "crowd_heatmap.png"
        ),

        "vehicle_heatmap": str(
            heatmap_dir / "vehicle_heatmap.png"
        ),

        "output_dir": str(output_dir)

    }


# ==========================================
# 12. LOCAL TEST
# ==========================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python src/dynamic_pipeline.py "
            "path/to/video.mp4"
        )

        raise SystemExit(1)

    input_video = Path(sys.argv[1])

    output_folder = (
        ROOT / "outputs" / "dynamic_test"
    )

    def show_progress(value, message):

        print(
            f"[{value * 100:.1f}%] {message}"
        )

    results = analyze_video(
        input_video,
        output_folder,
        progress_callback=show_progress,
        frame_stride=1,
        target_fps=60
    )

    print("\nDYNAMIC ANALYSIS COMPLETED!")

    print("Processed video:", results["video"])

    print("Output folder:", results["output_dir"])

    print(results["summary"])
