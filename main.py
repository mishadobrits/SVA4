"""
This module provides the main functions of this project - 'process_one_video_from_computer' and .
It processes the video in the way README says.
"""
import itertools
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp, gettempdir
import wave
import numpy as np
from some_functions import (
    v1timecodes_to_v2timecodes,
    save_v2_timecodes_to_file,
    read_bytes_from_wave,
    ffmpeg_atempo_filter, input_answer,
)
from ffmpeg_caller import FFMPEGCaller
from moviepy.editor import VideoFileClip


TEMPORY_DIRECTORY_PREFIX = "SVA4_"
AUDIO_CHUNK_IN_SECONDS = 300  #


def process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(),
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
    np.save("interesting_parts.npy", interesting_parts)
    apply_calculated_interesting_and_boring_parts_to_video(
        interesting_parts,
        settings,
        input_video_path,
        output_video_path,
        ffmpeg_caller=ffmpeg_caller,
        is_result_cfr=is_result_cfr
    )
    video.reader.close()
    video.audio.reader.close_proc()


def apply_calculated_interesting_and_boring_parts_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    working_directory_path=None,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(),
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
    :param ffmpeg_caller: = ffmpeg_caller.FFMPEGCaller(
            print_command=False,
            hide_output=False,
            overwrite_force=None
           ) an object which calls ffmpeg from cmd with some options
           to see this options look some_functions.py:23:5
    :param working_directory_path:
        working_directory_path = None (str/None)
        is a directory where this function saves all intermediate files.
        working_directory_path of should be str or None.
        If it's None, process_one_video_in_computer creates a temporary directory
        for this purpose (and deletes it when it finishes).
        The name of the temporary directory starts with 'SVA4_' for easy identification.
    :param hide_ffmpeg_output: = False (True/False)
    :param is_result_cfr: = False (True/False)
        if this option is 'True'
    :return: None
    """
    if Path(output_video_path).suffix != ".mkv":
        output_video_path += ".mkv"
        print(f"Output path changed to {output_video_path}")

    ffmpeg = ffmpeg_caller
    overwrite_output_force = ffmpeg_caller.get_overwrite_force()
    if os.path.exists(output_video_path) and overwrite_output_force is None:
        s = f"Output file is already exists and ffmpeg_caller.overwrite_force is None."
        s += " Overwrite it?"
        answer = input_answer(s, ["y", "Y", "n", "N"])
        overwrite_output_force = answer.lower() == "y"
    if os.path.exists(output_video_path) and not overwrite_output_force:
        print(f"File {output_video_path} is already exists and overwrite_output_force = False")
        print("Quiting")
        return

    def tpath(filename):
        return os.path.join(working_directory_path, filename)

    need_to_remove_working_directory_tree = working_directory_path is None
    if working_directory_path is None:
        working_directory_path = mkdtemp(prefix=TEMPORY_DIRECTORY_PREFIX)
        print(f"Temp floder: {working_directory_path}")

    video = VideoFileClip(input_video_path)

    interesting_parts, boring_parts = settings.process_interestingpartsarray(interesting_parts)

    interesting_parts[:2] = (interesting_parts[:2] * video.fps).astype(int) / video.fps
    boring_parts[:2] = (boring_parts[:2] * video.fps).astype(int) / video.fps
    interesting_parts[:2] = np.minimum(int(video.fps * video.duration - 1), interesting_parts[:2])
    boring_parts[:2] = np.minimum(int(video.fps * video.duration - 1), boring_parts[:2])

    inter_speed = settings.get_real_loud_speed()
    boring_speed = settings.get_real_quiet_speed()

    boring_parts = np.hstack([boring_parts, boring_speed * np.ones((len(boring_parts), 1))])
    interesting_parts = np.hstack(
        [interesting_parts, inter_speed * np.ones((len(interesting_parts), 1))]
    )

    boring_audio_path = tpath("boring_audio.wav")
    interesting_audio_path = tpath("interesting_audio.wav")
    final_audio_path = tpath("final_audio.aac")
    temp_final_audio_path = tpath("temp_audio.wav")

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

        debug_sum = 0
        for boring_and_interesting_part in parts_iterator:
            parts_with_file = zip([boring_audio, interesting_audio], boring_and_interesting_part)
            for file, part in parts_with_file:
                if part is None:
                    continue
                timecodes.append([part[0], part[1], part[2] * video.fps])
                part[0], part[1] = part[0] / part[2], part[1] / part[2]
                debug_sum += (part[1] - part[0]) / part[2]
                # print(part, debug_sum)

                for start in np.arange(part[0], part[1], AUDIO_CHUNK_IN_SECONDS):
                    end = min(part[1], start + AUDIO_CHUNK_IN_SECONDS)
                    # print(start, end)
                    temp_audio.writeframes(read_bytes_from_wave(file, start, end))

    ffmpeg(f"-i {temp_final_audio_path} {final_audio_path}")
    temp_images_path = tpath("temp_video.mp4")
    tempory_video_path = tpath("tempory_video.mkv")

    ffmpeg(f"-i {input_video_path} -c copy -an {temp_images_path}")

    v2timecodes_path, video_path2 = tpath("timecodes.v2"), tpath("v2video.mkv")
    v2timecodes = v1timecodes_to_v2timecodes(timecodes, video.fps, video.reader.nframes)
    save_v2_timecodes_to_file(v2timecodes_path, v2timecodes)

    print(f"  mkvmerge -o {video_path2} --timestamps 0:{v2timecodes_path} {temp_images_path}")
    os.system(f"mkvmerge -o {video_path2} --timestamps 0:{v2timecodes_path} {temp_images_path}")

    ffmpeg(
        f"-i {video_path2} -i {final_audio_path} -map 0:v -c copy -map 1:a {tempory_video_path}"
    )
    if is_result_cfr:
        print("CFR-ing video")
        temporary_vfr_video_path = tpath("tempory_vfr_video.mkv")
        os.renames(tempory_video_path, temporary_vfr_video_path)
        ffmpeg(f"-i {temporary_vfr_video_path} -c:a copy {tempory_video_path}")

    if os.path.exists(output_video_path):
        os.remove(output_video_path)
        # If os.path.exists(output_video_path) is True and overwrite_output_force is False
        # program quits earlier
    os.renames(tempory_video_path, output_video_path)

    if need_to_remove_working_directory_tree:
        video.reader.close()  # https://stackoverflow.com/a/45393619
        video.audio.reader.close_proc()
        # if function delete direcotry before video, video deletion raises an error.
        print(f"Removing {working_directory_path} tree", end="... ")
        shutil.rmtree(working_directory_path)
        print("done.")


def delete_all_tempories_sva4_directories():
    """
    When process_one_video_in_computer or apply_calculated_interesting_and_boring_parts_to_video
    creates temporary directory its name starts with TEMPORY_DIRECTORY_PREFIX="SVA_4"
    for easy identification. If user terminates process function doesn't delete directory,
    cause of it terminated. So, function delete_all_tempories_sva4_directories deletes
    all directories which marked with TEMPORY_DIRECTORY_PREFIX
    :return: None
    """
    temp_dirs = [f.path for f in os.scandir(gettempdir()) if f.is_dir()]
    for temp_dir in filter(lambda fold: fold.startswith(TEMPORY_DIRECTORY_PREFIX), temp_dirs):
        temp_dir_full_path = os.path.join(gettempdir(), temp_dir)
        print(f"Deleting {temp_dir_full_path}")
        shutil.rmtree(temp_dir_full_path)


