"""
Testing process_one_video_in_computer function
"""

from main import process_one_video_in_computer
from settings import Settings
from ffmpeg_caller import FFMPEGCaller
from speed_up import (
    VolumeThresholdAlgorithm,
    WebRtcVADAlgorithm,
    SileroVadAlgorithm,
    AlgOr,
    AlgAnd
)


input_video_path = input("write path of input video: ")

# speedup_algorithm = VolumeThresholdAlgorithm(0.02)  # or
# speedup_algorithm = WebRtcVADAlgorithm(3) or
# SileroVadAlgorithm(is_adaptive=True) or
speedup_algorithm = AlgAnd(
    VolumeThresholdAlgorithm(0.023, min_quiet_time=0.2),
    WebRtcVADAlgorithm(2),
    SileroVadAlgorithm(is_adaptive=True),
)  # or any other option

settings = Settings(quiet_speed=6)

output_video_path = input("write path of output video: ")

process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=True)
)
