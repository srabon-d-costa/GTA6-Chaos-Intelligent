
import subprocess
import imageio_ffmpeg

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO = ROOT / "outputs/videos/Game_detected.mp4"

OUTPUT_VIDEO = ROOT / "outputs/videos/Game_detected_web.mp4"

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(INPUT_VIDEO)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-y",
    "-i", str(INPUT_VIDEO),
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
    str(OUTPUT_VIDEO)
]

print("Converting video to browser-compatible H.264...")

subprocess.run(command, check=True)

print("\nVIDEO CONVERSION SUCCESSFUL!")
print("Output:", OUTPUT_VIDEO)
