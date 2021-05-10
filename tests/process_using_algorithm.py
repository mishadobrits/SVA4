"""
Testing process_one_video_in_computer function
"""

from main import process_one_video_in_computer
from settings import Settings
from speed_up import VolumeAlgorithm

input_video_path = input("write path of input video: ")
speedup_algorithm = VolumeAlgorithm(0.024)
settings = Settings(min_quiet_time=1, quiet_speed=6)
output_video_path = input("write path of output mkv video path: ")
process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    overwrite_output_force=True,
    hide_ffmpeg_output=True,
    print_ffmpeg_commands=True
)
