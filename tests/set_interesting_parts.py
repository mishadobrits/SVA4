"""
Testing apply_calculated_interesting_and_boring_parts_to_video function
"""
import numpy as np

from main import apply_calculated_interesting_and_boring_parts_to_video
from settings import Settings

input_video_path = input("write path of input video: ")
interesting_parts = np.array([[10.0, 20.1], [30.5, 40], [50.5, 60], [70.5, 80]])
settings = Settings(min_quiet_time=0.2, quiet_speed=6)
output_video_path = input("write path of output mkv video: ")
apply_calculated_interesting_and_boring_parts_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    overwrite_output_force=True,
    hide_ffmpeg_output=True,
)
