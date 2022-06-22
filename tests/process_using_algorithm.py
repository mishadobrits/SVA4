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
    AlgAnd, AlgAnd1, RemoveShortParts, AlgNot
)

input_video_path = input("Type the path of an input video: ")

# speedup_algorithm = VolumeThresholdAlgorithm(0.02)  # or
# speedup_algorithm = WebRtcVADAlgorithm(3) or
# speedup_algorithm = SileroVadAlgorithm(is_adaptive=True) or
speedup_algorithm = AlgAnd(    # parts where volume >= 0.25 AND silero_probability > 0.4
    VolumeThresholdAlgorithm(0.025),   # parts, where volume > 0.025 are skipped
    SileroVadAlgorithm(0.4),           # parts, where algorithm from Silero return probability of speech,
                                       # that greater than 0.4 are skipped
)  # or any other option
# speedup_algorithm = RemoveShortParts(VolumeThresholdAlgorithm(0.02), min_part_lenght=0.15) # sec

settings = Settings(quiet_speed=6)

output_video_path = input("Type the path of an output video: ")

process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=True, print_command=True),
    # audiocodec="pcm_s16le",
)
