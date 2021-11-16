import math
import os
import random
import logging
from decimal import getcontext
from tempfile import gettempdir
from main import delete_all_sva4_temporary_objects
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from settings import Settings
from some_functions import save_v2_timecodes_to_file, v1timecodes_to_v2timecodes
from speed_up import _FakeDebugAlgorithm, AlgOr, CropLongSounds, AlgNot


def read(filename):
    with open(f"tests/{filename}.npy", "rb") as f:
        abc = np.load(f, allow_pickle=False)
    with open(f"tests/{filename}.txt", "w") as f:
        f.write("\n".join(map(str, abc)))

    return abc


def debug_and_alg(input_video_path):
    a, b, c, abc = read(0), read(1), read(2), read("rt")
    x, y, z, xyz = _FakeDebugAlgorithm(a), _FakeDebugAlgorithm(b), _FakeDebugAlgorithm(c), _FakeDebugAlgorithm(abc)


def debug_audio(input_video_path):
    from ffmpeg_caller import FFMPEGCaller
    from some_functions import WavSubclip
    ffmpeg = FFMPEGCaller(overwrite_force=False, hide_output=True)
    audio_path = "test_audio.wav"
    ffmpeg(f"-i {input_video_path} {audio_path}")
    for i in range(1000):
        start = i  # random.random() * 100
        end = i + 2  # start + random.random() * 10
        wav = WavSubclip(audio_path).subclip(start, end).to_soundarray()
        honest_audio = AudioFileClip(audio_path).subclip(start, end).to_soundarray()
        print(i, abs(wav - honest_audio).max())
        # print(wav, honest_audio)
        # assert (wav == honest_audio).all(), "not equal"


def is_bad(a: float):
    fstr = lambda elem: format(elem * 1000, "f")
    delta = 10 ** -6 / 240
    return fstr(a) == fstr(a + delta)



"""
Testing process_one_video_in_computer function
"""
import sys
sys.path.append("/content/SVA4")

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

# """
def read_v1(v1path="tmp/timecodes.v1"):
    with open(v1path) as v1file:
        v1file.readline()
        v1file.readline()
        v1 = [list(map(float, line.split(","))) for line in v1file]
    return v1


def read_v2(v2path="tmp/timecodes.v2"):
    with open(v2path) as v2file:
        v2file.readline()
        v2 = [float(line) for line in v2file]
    return v2


def check_v1_and_v2(v1path="tmp/timecodes.v1", v2path="tmp/timecodes.v2"):
    v1, v2 = read_v1(v1path), read_v2(v2path)
    print(sum((elem[1] - elem[0]) / elem[2] for elem in v1))
    print(v2[-1] / 1000)


"""

save_v2_timecodes_to_file("tmp/timecodes.v2", v1timecodes_to_v2timecodes(read_v1(), 30, 150000))
check_v1_and_v2()
#"""


# input_video_path = input("write path of input video (/content/input_video.mkv): ")
input_video_path = r"C:\Users\m\Downloads\Sites-Buffers\ "[:-1] + input("Input filename: ")  # Клименко А В Дифференциальные уравнения 12.11.2021.mp4" # input("Input video path: ") #
# speedup_algorithm = VolumeThresholdAlgorithm(0.02)  # or
# speedup_algorithm = WebRtcVADAlgorithm(3) or
# SileroVadAlgorithm(is_adaptive=True) or
speedup_algorithm = AlgAnd(
    VolumeThresholdAlgorithm(0.02),
    # WebRtcVADAlgorithm(1),
    # SileroVadAlgorithm(),
    # CropLongSounds(max_lenght_of_one_sound=0.03, threshold=0.9985),  # is_adaptive=True
)  # or any other option

settings = Settings(quiet_speed=6)

# output_video_path = input("write path of output video (/content/output_video.mkv): ")
output_video_path = r"C:\Users\m\Downloads\Sites-Buffers\output with spaces.mkv"

process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=False, print_command=True),
)
# """
