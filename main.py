"""
This module provides the main functions of this project - 'process_one_video_from_computer' and .
It processes the video in the way README says.
"""
import itertools
import os
import shutil
from os.path import join as joinpath
from pathlib import Path
from tempfile import mkdtemp
import wave
import numpy as np
from moviepy.editor import VideoFileClip
from some_functions import (
    FFMPEGCaller,
    v1timecodes_to_v2timecodes,
    save_v2_timecodes_to_file,
    read_bytes_from_wave,
    ffmpeg_atempo_filter,
)


def process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    working_directory_path=None,
    overwrite_output_force=None,
    hide_ffmpeg_output=False,
    print_ffmpeg_commands=False,
):
    """
    This function processes video (
        1) Uses 'speedup_algorithm' to get interesting parts of video
        2) calls apply_calculated_interesting_and_boring_parts_to_video function
    )
    """

    print("  Splitting audio into boring / interesting parts")
    video = VideoFileClip(input_video_path)
    interesting_parts = speedup_algorithm.get_interesting_parts(video)
    # np.save("interesting_parts.npy", interesting_parts)
    apply_calculated_interesting_and_boring_parts_to_video(
        interesting_parts,
        settings,
        input_video_path,
        output_video_path,
        working_directory_path=working_directory_path,
        overwrite_output_force=overwrite_output_force,
        hide_ffmpeg_output=hide_ffmpeg_output,
        print_ffmpeg_commands=print_ffmpeg_commands,
    )


def apply_calculated_interesting_and_boring_parts_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    overwrite_output_force=None,
    working_directory_path=None,
    hide_ffmpeg_output=False,
    print_ffmpeg_commands=False,
):
    """
    This function does what readme said
    (
        1) Changes interesting & boring parts using 'settings' rules.
        2) Applies new parts to video (changing timecodes)
                            and audio (using atempo filter to speed up)
        3) Merges result video and result audio to 'output_video_path'.
    )

    :param interesting_parts: interesting_parts array in format
        [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ...
             [start_of_piece2, end_of_piece2]].
        All values should be positions in video in seconds.
    :param settings: <class 'settings.Settings'>
    :param input_video_path:
    :param output_video_path:
    :param overwrite_output_force: = None (None/True/False)
        if the value is None and if this program needs to overwrite a file,
         this function asks for your acceptance.
        if the value is True and if this program needs to overwrite a file,
         this function overwrites it.
        if the value is False and if this program needs to overwrite a file,
         this function doesn't overwrite it.
    :param working_directory_path:
        working_directory_path = None (str/None)
        is a directory where this function saves all intermediate files.
        working_directory_path of should be str or None.
        If it's None, process_one_video_in_computer creates a temporary directory
        for this purpose (and deletes it when it finishes).
        The name of the temporary directory starts with 'SVA4_' for easy identification.
    :param hide_ffmpeg_output: = False (True/False)
        If this value is True program hides ffmpeg output
        If this value is False program doesn't hide ffmpeg output
    :param print_ffmpeg_commands:
        if this parameter is True program prints all ffmpeg commands before executing them .
        If this parameter is False it doesn't.
    :return: None
    """
    ffmpeg = FFMPEGCaller(
        print_command=print_ffmpeg_commands,
        hide_output=hide_ffmpeg_output,
        overwrite_force=overwrite_output_force,
    )

    if Path(output_video_path).suffix != ".mkv":
        output_video_path += ".mkv"

    def fpath(filename):
        return joinpath(working_directory_path, filename)

    need_to_remove_working_directory_tree = working_directory_path is None
    if working_directory_path is None:
        working_directory_path = mkdtemp(prefix="SVA4_")
        print(f"Temp floder: {working_directory_path}")

    video = VideoFileClip(input_video_path)

    interesting_parts, boring_parts = settings.apply_settings_to_interestingpartsarray(
        interesting_parts
    )

    interesting_parts[:2] = (interesting_parts[:2] * video.fps).astype(int) / video.fps
    boring_parts[:2] = (boring_parts[:2] * video.fps).astype(int) / video.fps
    interesting_parts[:2] = np.minimum(
        int(video.fps * video.duration - 1), interesting_parts[:2]
    )
    boring_parts[:2] = np.minimum(int(video.fps * video.duration - 1), boring_parts[:2])

    inter_speed, boring_speed = (
        settings.get_real_loud_speed(),
        settings.get_real_quiet_speed(),
    )
    boring_parts = np.hstack(
        [boring_parts, boring_speed * np.ones((len(boring_parts), 1))]
    )
    interesting_parts = np.hstack(
        [interesting_parts, inter_speed * np.ones((len(interesting_parts), 1))]
    )

    boring_audio_path, interesting_audio_path = (
        fpath("boring_audio.wav"), fpath("interesting_audio.wav")
    )

    final_audio_path, temp_final_audio_path = (
        fpath("final_audio.aac"), fpath("temp_audio.wav")
    )

    print(f"  writing audio with { {'speed': boring_speed}} to '{boring_audio_path}'")
    ffmpeg(f"-i {input_video_path} -vn {ffmpeg_atempo_filter(boring_speed)} {boring_audio_path}")

    print(f"  writing audio with { {'speed': inter_speed}} to '{interesting_audio_path}'")
    ffmpeg(
        f"-i {input_video_path} -vn {ffmpeg_atempo_filter(inter_speed)} {interesting_audio_path}"
    )

    timecodes = []
    with wave.open(boring_audio_path) as boring_audio,\
            wave.open(interesting_audio_path) as interesting_audio,\
            wave.open(temp_final_audio_path, "w") as temp_audio:
        temp_audio.setparams(boring_audio.getparams())
        parts_iterator = itertools.zip_longest(boring_parts, interesting_parts)
        for boring_and_interesting_part in parts_iterator:
            parts_with_file = zip(
                [boring_audio, interesting_audio], boring_and_interesting_part
            )
            for file, part in parts_with_file:
                if part is None:
                    continue
                timecodes.append([part[0], part[1], part[2] * video.fps])
                part[0], part[1] = part[0] / part[2], part[1] / part[2]
                read_bytes = read_bytes_from_wave(file, part[0], part[1])
                temp_audio.writeframes(read_bytes)

    # print(s, sum((tc[1] - tc[0]) / tc[2] for tc in timecodes))
    ffmpeg(f"-i {temp_final_audio_path} {final_audio_path}")
    temp_video_path = fpath("temp_video.mp4")

    ffmpeg(f"-i {input_video_path} -c copy -an {temp_video_path}")

    v2timecodes_path, video_path2 = fpath("timecodes.v2"), fpath("v2video.mkv")

    save_v2_timecodes_to_file(
        v2timecodes_path,
        v1timecodes_to_v2timecodes(timecodes, video.fps, video.reader.nframes),
    )
    print(
        f"  mkvmerge -o {video_path2} --timestamps 0:{v2timecodes_path} {temp_video_path}"
    )
    os.system(
        f"mkvmerge -o {video_path2} --timestamps 0:{v2timecodes_path} {temp_video_path}"
    )
    ffmpeg(
        f"-i {video_path2} -i {final_audio_path} -map 0:v -c copy -map 1:a {output_video_path}"
    )

    if need_to_remove_working_directory_tree:
        video.reader.close()  # https://stackoverflow.com/a/45393619
        video.audio.reader.close_proc()
        # if function delete direcotry before video, video deletion raises an error.
        print(f"Removing {working_directory_path} tree", end="... ")
        shutil.rmtree(working_directory_path)
        print("done.")
