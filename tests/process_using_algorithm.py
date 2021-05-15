"""
Testing process_one_video_in_computer function
"""

from main import process_one_video_in_computer
from settings import Settings
from ffmpeg_caller import FFMPEGCaller
from speed_up import VolumeAlgorithm, WebRtcVADAlgorithm

input_video_path = input("write path of input video: ")
# speedup_algorithm = VolumeAlgorithm(0.0275)
speedup_algorithm = WebRtcVADAlgorithm(3)
settings = Settings(min_quiet_time=0.2, quiet_speed=6)
output_video_path = input("write path of output mkv video path: ")

process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=True)
)
