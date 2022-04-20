"""
This module only contains functions that others modules call.
I moved them to a separate file because all modules use these functions,
 and they can't call each other in a circle.
"""
import hashlib
import itertools
import math
import os
import shutil
from tempfile import gettempdir, mkdtemp
import numpy as np
from imageio_ffmpeg import count_frames_and_secs


TEMPORARY_DIRECTORY_PREFIX = "SVA4_"


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def str2error_message(msg):
    """Deletes \n from msg and replace ' '*n -> ' '"""
    return " ".join(list(msg.replace("\n", " ").split()))


def get_working_directory_path(working_directory_path: str) -> str:
    if working_directory_path is None:
        return mkdtemp(prefix=TEMPORARY_DIRECTORY_PREFIX)
    return working_directory_path


def input_answer(quetsion: str, answers_list: list, attempts: int = 10**10):
    """
    Prints quetsion until user enters answers_list. (answers_list is also printed)

    :param quetsion: str: Question that will be displayed.
    :param answers_list: List of available answers.
    :param attempts: Number of attempts that user has.
    :return: if somewhen user wrote an answer from answers_list function immediately returns it.
             else None returned.
    """
    def list2str(option_list):
        if not option_list:
            return ""
        if len(option_list) == 1:
            return option_list[1]
        return f"{', '.join(option_list[:-1])} or {option_list[-1]}"

    addition = f" ({list2str(answers_list)}"
    for i in range(attempts):
        if i:
            msg = "Cannot understand input '{}'. Available values is {}"
            print(msg.format(answer, list2str(answers_list)))
        answer = input(quetsion + addition)
        if answer in answers_list:
            return answer


def v1timecodes_to_v2timecodes(v1timecodes, video_fps, length_of_video, default_output_fps=10 ** 6):
    """

    :param v1timecodes: timecodes in v1format:
        [[start0, end0, fps0], [start1, end1, fps1], ... [start_i, end_i, fps_i]]
         (same as save_timecodes_to_v1_file)
        where start and end in seconds, fps in frames per second
    :param video_fps: fps you want in output video
    :param length_of_video: number of frames in output frames (in case it's greater then v1timecodes[-1] * fps.
    :param default_output_fps: fps in parts not covered py v1timecodes
    :return: v2timecodes: timecodes in v2format:
        [timecode_of_0_frame_in_ms, timecode_of_1st_frame_in_ms, ... timecode_of_nth_frame_in_ms]
    """

    default_freq = 1 / default_output_fps / video_fps
    time_between_neighbour_frames = default_freq * np.ones(length_of_video, dtype=np.float64)
    for elem in v1timecodes:

        start_t, end_t = elem[0] * video_fps, elem[1] * video_fps
        # todo begin kostil
        start_t, end_t = min(start_t, length_of_video - 1), min(end_t, length_of_video - 1)
        start_i, end_i = math.floor(start_t), math.floor(end_t) + 1

        X = time_between_neighbour_frames[start_i: end_i]
        if not X.size:
            continue
            # print(start_t, end_t, start_i, end_i)
        X += 1 / elem[2] * (end_t - start_t) / (end_i - start_i)
        # print((end_i - start_t) / elem[2] - sum(X))
        # end kostil

        """
        addition = 1 / elem[2] - default_freq
        # print(start_t, end_t, addition, ":")
        ceil_start_t, floor_end_t = math.ceil(start_t), math.floor(end_t)
        # print(f"  [{ceil_start_t}-{floor_end_t}] += {addition}")
        # print(f"  [{ceil_start_t - 1}] += {addition} * {(ceil_start_t - start_t)}")
        # print(f"  [{floor_end_t}] += {addition} * {(end_t - floor_end_t)}")
        time_between_neighbour_frames[ceil_start_t: floor_end_t] += addition
        time_between_neighbour_frames[ceil_start_t - 1] += addition * (ceil_start_t - start_t)
        time_between_neighbour_frames[floor_end_t] += addition * (end_t - floor_end_t)
        # """

    timecodes = cumsum(time_between_neighbour_frames)
    return timecodes


def save_v2_timecodes_to_file(filepath, timecodes):
    """
    :param filepath: path to file for saving
    :param timecodes: list of timecodes of each frame in format
            [timecode_of_0_frame_in_ms, timecode_of_1_frame_in_ms, ... timecode_of_i_frame_in_ms]
    :return: file object (closed)
    """
    str_timecodes = [format(elem * 1000, "f") for elem in timecodes]
    for i, (elem, next_elem) in enumerate(pairwise(str_timecodes)):
        if float(elem) >= float(next_elem):
            str_timecodes[i + 1] += str(i).rjust(10, "0")
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
    #     return "-c:a copy"

    return f"-af atempo={speed}"


def create_valid_path(path_with_spaces: str):
    if any(bad_symb in os.path.abspath(path_with_spaces) for bad_symb in " '\""):
        path_hash = hashlib.sha1(path_with_spaces.encode("utf-8")).hexdigest()
        new_name = TEMPORARY_DIRECTORY_PREFIX + path_hash + os.path.splitext(path_with_spaces)[1]
        new_video_path = os.path.join(gettempdir(), new_name)
        if os.path.exists(path_with_spaces):
            shutil.copyfile(path_with_spaces, new_video_path)
        new_path = new_video_path
    else:
        new_path = path_with_spaces
    return new_path


def get_duration(video_path: str):
    nframes, secs = count_frames_and_secs(video_path)
    return secs


def get_nframes(video_path: str):
    nframes, secs = count_frames_and_secs(video_path)
    return nframes


def collide(*iters):
    was_iterated = True
    while was_iterated:
        was_iterated = False
        for it in iters:
            try:
                yield next(it)
                was_iterated = True
            except StopIteration:
                continue
