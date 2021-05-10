import itertools
import math
import os
import random
import re
import time
import wave
from tempfile import TemporaryDirectory

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.VideoFileClip import VideoFileClip

# from main import _get_ffmpeg_atempo_filter
from main import process_one_video_in_computer
from some_functions import (
    v1timecodes_to_v2timecodes,
    save_v2_timecodes_to_file,
    FFMPEGCaller,
)
from settings import Settings
from speed_up import VolumeAlgorithm

r'''sound = np.array([0, 0, 0, 0.9, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0.9, 0], dtype="float32").reshape(-1, 1)
# print(sound)
a = AudioArrayClip(sound, fps=1)
v = VideoFileClip(r"C:\Users\m\Downloads\Sites-Buffers\sva_geoma_lection_31mar21.avi.avi")
v.audio = a
print(v.audio.to_soundarray().max(axis=1).reshape(-1))

loud_parts = VolumeAlgorithm(0.5).get_interesting_parts(v)
print(loud_parts)

s = Settings(min_quiet_time=2, max_quiet_time=3)
inter, uninter = apply_settings_to_interestingpartsarray(loud_parts, s)
print(inter)
print(uninter)
inter, uninter = inter.transpose((1, 0)), uninter.transpose((1, 0))
for elem in zip(uninter, inter):
    print(elem)
""" + 0.9

arr = begin_sound_indexes[1:] > end_sound_indexes[:-1]
begin_sound_indexes = begin_sound_indexes[np.hstack([np.array([True]), arr])]
end_sound_indexes = end_sound_indexes[np.hstack([arr, np.array([True])])]
"""

# print(np.vstack([begin_sound_indexes, end_sound_indexes]))


print()
"""'''

# """
def print_v2timecodes_as_v1(filepath=r"Tempary_files/2/timecodes.v2"):
    with open(filepath) as f:
        f.readline()
        previous_elem, previous_value = 0, -1
        start_of_part, value = 0, -1
        for i, elem in enumerate(f.readlines()):
            value = (float(elem) - previous_elem) / 1000
            previous_elem = float(elem)

            if abs(value - previous_value) > 10 ** -6:
                print(
                    start_of_part,
                    i / 25,
                    round(1 / previous_value / 25)
                    if round(previous_value, 6)
                    else u"∞",
                )
                start_of_part = i / 25
                previous_value = value


# """


def watch_v2timecodes(filepath=r"Tempary_files/2/timecodes.v2"):
    with open(filepath) as f:
        fcontent = "".join(f.readlines()[1:])

    tc = np.array(list(map(float, fcontent.split())))
    # print(tc[:10])
    while True:
        time = input("input time: ")
        if time == "quit":
            break
        try:
            time = float(time)
            print(tc[int(time * 25) - 1 : int(time * 25) + 2] / 1000)
        except Exception as e:
            print("Exception occurs:")
            print(e)
            print("Try again")
            continue


def process_v1timecodes(
    inp_path="v1timecodes.npy", out_path=r"Tempary_files/2/timecodes.v2"
):
    """

    :param inp_path str:
    :param out_path:
    :return:
    """
    video = VideoFileClip(r"C:\Users\m\Downloads.mp4")
    # watch_v2timecodes()
    with open(inp_path, "rb") as f:
        v1timecodes = np.load(f)
        print(v1timecodes)

    v2timecodes = v1timecodes_to_v2timecodes(
        v1timecodes, video.fps, int(video.fps * video.duration)
    )
    save_v2_timecodes_to_file(out_path, v2timecodes)


a = str(5, p=4)
settings = Settings(min_quiet_time=1, quiet_speed=6)
with open("tests/interesting_parts.npy", "rb") as f:
    i = np.load(f)
# print(i)
i, b = settings.apply_settings_to_interestingpartsarray(i)
print(i)
print(b)

with open(r"Tempary_files/2/timecodes.v2") as f:
    fcontent = "".join(f.readlines()[1:])


#
tc = np.array(list(map(float, fcontent.split())))
index = (tc[1:] - tc[:-1]).argmin()
print(tc[index + 1] - tc[index])
print(tc[index - 1 : index + 2])

r"""v = VideoFileClip(r"Tempary_files/2/v2video1.mkv")
print(v.filename)

t = time.time()   # out10sec.mp4
input_video_path = r"Tempary_files/out2.mkv" # r"C:\Users\m\Desktop\PythonProjects\SVA_4\Tempary_files\test2.mp4" #  r"C:\Users\m\Downloads.mp4"
process_one_video_in_computer(input_video_path,
                              VolumeAlgorithm(0.05),
                              Settings(max_quiet_time=6),
               r"C:\Users\m\Desktop\PythonProjects\SVA_4\Tempary_files\2\sva4_output.mkv",
                              working_directory=r"Tempary_files\2")
print(f"SVA4 successfully finishes after {time.time() - t} seconds of working")  # text.mkv input.mp4
"""
