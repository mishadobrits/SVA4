"""
Testing process_one_video_in_computer function
"""
import os, sys
this_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.join(this_path, os.pardir)
sys.path.append(parent_path)
import time

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

speedup_algorithm = AlgAnd(    # parts where volume >= 0.25 AND silero_probability > 0.4
    VolumeThresholdAlgorithm(0.025),   # parts, where volume > 0.025 are skipped
    SileroVadAlgorithm(0.4),           # parts, where algorithm from Silero return probability of speech,
                                       # that greater than 0.4 are skipped
)

settings = Settings(quiet_speed=6)

output_video_path = input("Type the path of an output video: ")

t = time.time()
process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=True, print_command=False),
    # audiocodec="pcm_s16le",
)
print(f"Total time: {round(time.time() - t, 3)} seconds")
