"""
This module only contains functions that others modules call.
I moved them to a separate file because all modules use these functions,
 and they can't call each other in a circle.
"""

import itertools
import math
import os
import numpy as np


def get_subclip_soundarray(wavio_oblect, start, end):
    framerate = wavio_oblect.rate
    return wavio_oblect.data[int(start * framerate): int(end * framerate)]


def str2error_message(msg):
    """Deletes \n from msg and replace ' '*n -> ' '"""
    return " ".join(list(msg.replace("\n", " ").split()))


def read_bytes_from_wave(waveread_obj, start_sec, end_sec):
    previous_pos, framerate = waveread_obj.tell(), waveread_obj.getframerate()

    start_pos = min(waveread_obj.getnframes(), math.ceil(framerate * start_sec))
    end_pos = min(waveread_obj.getnframes(), math.ceil(framerate * end_sec))

    waveread_obj.setpos(start_pos)
    rt_bytes = waveread_obj.readframes(end_pos - start_pos)
    waveread_obj.setpos(previous_pos)

    return rt_bytes


def input_answer(quetsion, answers_list, quit_options=["q", "Q"], attempts=10**10):
    def list2str(option_list):
        if not option_list:
            return ""
        if len(option_list) == 1:
            return option_list[1]
        return f"{', '.join(option_list[:-1])} or {option_list[-1]}"

    addition = f" (options: {list2str(answers_list)}; {list2str(quit_options)} to quit)"
    for i in range(attempts):
        if i:
            print(f"Cannot understand input '{answer}'. Available values is {addition}")
        answer = input(quetsion + addition)
        if answer in answers_list:
            return answer
        if answer in quit_options:
            print("Quiting")
            exit(0)


def v1timecodes_to_v2timecodes(v1timecodes, video_fps, length_of_video, default_output_fps=9 ** 9):
    """

    :param v1timecodes: timecodes in v1format:
        [[start0, end0, fps0], [start1, end1, fps1], ... [start_i, end_i, fps_i]]
         (same as save_timecodes_to_v1_file)
        where start and end in seconds, fps in frames per second
    :return: v2timecodes: timecodes in v2format:
        [timecode_of_0_frame_in_ms, timecode_of_1st_frame_in_ms, ... timecode_of_nth_frame_in_ms]
    """

    default_freq = 1 / default_output_fps / video_fps
    time_between_neighbour_frames = default_freq * np.ones(length_of_video, dtype=np.float64)
    for elem in v1timecodes:
        start_t, end_t = elem[0] * video_fps, elem[1] * video_fps
        # todo begin kostil
        start_t = min(start_t, length_of_video - 1)
        end_t = min(end_t, length_of_video - 1)
        # end kostil

        time_between_neighbour_frames[round(start_t): round(end_t)] = 1 / elem[2]

        """
        tc[math.floor(start_t)] += (1 - start_t % 1) * (1 / elem[2] - default_freq)
        tc[math.floor(end_t)] += (end_t % 1) * (1 / elem[2] - default_freq)
        tc[math.floor(start_t) + 1: math.floor(end_t)] = 1 / elem[2]
        """
    timecodes = cumsum(time_between_neighbour_frames)  # np.nancumsum(tc)
    # with open('v1timecodes.npy', 'wb') as f:
    #     np.save(f, v1timecodes)
    # print(f"rt[-1] = {rt[-1]}")
    return timecodes


def save_v2_timecodes_to_file(filepath, timecodes):
    """
    :param filepath: path to file for saving
    :param timecodes: list of timecodes of each frame in format
            [timecode_of_0_frame_in_ms, timecode_of_1_frame_in_ms, ... timecode_of_i_frame_in_ms]
    :return: file object (closed)
    """
    str_timecodes = [format(elem * 1000, "f") for elem in timecodes]
    # print(f"filepath = '{filepath}'")
    with open(filepath, "w") as file:
        file.write("# timestamp format v2\n")
        file.write("\n".join(str_timecodes))
    return file


def save_v1_timecodes_to_file(filepath, timecodes, videos_fps, default_fps=10 ** 10):
    """

    :param filepath: path of the file for saving
    :param timecodes: timecodes in format
        [[start0, end0, fps0], [start1, end1, fps1], ... [start_i, end_i, fps_i]]
    :param videos_fps: float fps of video
    :param default_fps: fps of uncovered pieces
    :return: closed file object in which timecodes saved
    """
    with open(filepath, "w") as file:
        file.write("# timecode format v1\n")
        file.write(f"assume {default_fps}\n")
        for elem in timecodes:
            elem = [int(elem[0] * videos_fps), int(elem[1] * videos_fps), elem[2]]
            elem = [str(n) for n in elem]
            # print(elem, ",".join(elem))
            file.write(",".join(elem) + "\n")
    return file


def cumsum(n1array):
    """
    np.nancumsum works wrong for me, so I wrote equivalent function
    :param n1array:
    :return: n1array of cumulative sums
    """
    accumalated_iter = itertools.accumulate(n1array.tolist())
    return np.array(list(itertools.chain([0], accumalated_iter)))


def ffmpeg_atempo_filter(speed):
    """
    returns string "-af {speed}" atempo filter.


    :param speed: float
    :return: atempo_filter: string argument fo ffmpeg in format atempo=1.25,atempo=2.0,atempo=2.0
    """
    if speed <= 0:
        raise ValueError(f"ffmpeg speed {speed} must be positive")
    # if speed == 1:
    #     return ""

    return f"-af atempo={speed}"
