"""
This module provides the main functions of this project - 'process_one_video_from_computer' and .
It processes the video in the way README says.
"""
import itertools
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp, gettempdir
import logging
from wave import Wave_read, Wave_write
import numpy as np
from moviepy.editor import VideoFileClip
from settings import Settings
from some_functions import (
    v1timecodes_to_v2timecodes,
    save_v2_timecodes_to_file,
    read_bytes_from_wave,
    ffmpeg_atempo_filter, input_answer, TEMPORARY_DIRECTORY_PREFIX, create_valid_path,
)
from ffmpeg_caller import FFMPEGCaller
from speed_up import SpeedUpAlgorithm


AUDIO_CHUNK_IN_SECONDS = 60


def process_one_video_in_computer(
    input_video_path: str,
    speedup_algorithm: SpeedUpAlgorithm,
    settings: Settings,
    output_video_path: str,
    is_result_cfr: bool = False,
    logger: logging.Logger = logging.getLogger("process_one_video_in_computer"),
    working_directory_path=None,
    ffmpeg_caller: FFMPEGCaller = FFMPEGCaller(),
):
    """
    This function processes video (
        1) Uses 'speedup_algorithm' to get interesting parts of video
        2) calls apply_calculated_interesting_to_video function
    )
    """
    new_input_video_path = create_valid_path(input_video_path)

    print("  Splitting audio into boring / interesting parts")
    interesting_parts = speedup_algorithm.get_interesting_parts(new_input_video_path)
    # np.save("interesting_parts.npy", interesting_parts)
    apply_calculated_interesting_to_video(
        interesting_parts,
        settings,
        new_input_video_path,
        output_video_path,
        logger=logger,
        working_directory_path=working_directory_path,
        ffmpeg_caller=ffmpeg_caller,
        is_result_cfr=is_result_cfr
    )


def apply_calculated_interesting_to_video(
    interesting_parts: np.array,
    settings: Settings,
    input_video_path: str,
    output_video_path: str,
    working_directory_path: str = None,
    is_result_cfr: bool = False,
    logger: logging.Logger = logging.getLogger("process_one_video_in_computer"),
    ffmpeg_caller: FFMPEGCaller = FFMPEGCaller(),
):
    """
    This function does what readme says
    (
        1) Changes interesting & boring parts using 'settings' rules.
        2) Applies new parts to video (changing timecodes)
                            and audio (using atempo filter to speed up)
        3) Merges result video and result audio to 'output_video_path'.
    )

    :param interesting_parts: interesting_parts np array in format
        [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ...
             [start_of_piece2, end_of_piece2]].
        All values should be positions in video in seconds.
    :param settings: <class 'settings.Settings'>
    :param input_video_path: str - path of input video
    :param output_video_path: str - path of output video
    :param ffmpeg_caller: = ffmpeg_caller.FFMPEGCaller(
            print_command=False,
            hide_output=False,
            overwrite_force=None
           ) an object which calls ffmpeg from cmd with some options
           to see this options look some_functions.py:23:5
    :param logger: logging.Logger
    :param working_directory_path:
        working_directory_path = None (str/None)
        is a directory where this function saves all intermediate files.
        working_directory_path of should be str or None.
        If value is None, process_one_video_in_computer creates a temporary directory
        for this purpose (and deletes it when it finishes).
        The name of the temporary directory starts with 'SVA4_' for easy identification.
    :param is_result_cfr: bool = False (True/False)
        if this option is 'True' output video will be CFR-ed using terminal comand
        "ffmpeg -i {vfr_video_path} {cfr_video_path}"
    :return: None
    """
    logger = logger or logging.getLogger('dummy')  # https://stackoverflow.com/a/13525899
    if Path(output_video_path).suffix != ".mkv":
        output_video_path += ".mkv"
        logger.log(1, f"Output path changed to {output_video_path}")

    ffmpeg = ffmpeg_caller
    overwrite_output_force = ffmpeg_caller.get_overwrite_force()
    if os.path.exists(output_video_path) and overwrite_output_force is None:
        msg = "Output file is already exists and ffmpeg_caller.overwrite_force is None."
        msg += " Overwrite it?"
        answer = input_answer(msg, ["y", "Y", "n", "N"])
        overwrite_output_force = answer.lower() == "y"
    if os.path.exists(output_video_path) and not overwrite_output_force:
        logger.log(1, f"File {output_video_path} is already exists and overwrite_output_force = False")
        logger.log(1, "Quiting")
        return

    def tpath(filename):
        """
        returns the absolute path for a file with name filename in folder working_directory_path
        """
        return os.path.join(working_directory_path, filename)

    need_to_remove_working_directory_tree = False
    if working_directory_path is None:
        working_directory_path = mkdtemp(prefix=TEMPORARY_DIRECTORY_PREFIX)
        need_to_remove_working_directory_tree = True
        logger.log(1, f"Temp floder: {working_directory_path}")

    if " " in os.path.abspath(input_video_path):
        new_video_path = tpath(f"input_video.{os.path.splitext(input_video_path)}")
        shutil.copyfile(input_video_path, new_video_path)
        input_video_path = new_video_path

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

    logger.log(1, f"writing audio with { {'speed': boring_speed}} to '{boring_audio_path}'")
    ffmpeg(f"-i {input_video_path} -vn {ffmpeg_atempo_filter(boring_speed)} {boring_audio_path}")

    logger.log(1, f"writing audio with { {'speed': inter_speed}} to '{interesting_audio_path}'")
    ffmpeg(
        f"-i {input_video_path} -vn {ffmpeg_atempo_filter(inter_speed)} {interesting_audio_path}"
    )

    v1timecodes = []
    with Wave_read(boring_audio_path) as boring_audio,\
            Wave_read(interesting_audio_path) as interesting_audio,\
            Wave_write(temp_final_audio_path) as temp_audio:
        temp_audio.setparams(boring_audio.getparams())
        parts_iterator = itertools.zip_longest(boring_parts, interesting_parts)

        for boring_and_interesting_part in parts_iterator:
            parts_with_file = zip([boring_audio, interesting_audio], boring_and_interesting_part)
            for file, part in parts_with_file:
                if part is None:
                    continue
                v1timecodes.append([part[0], part[1], part[2] * video.fps])
                part[0], part[1] = part[0] / part[2], part[1] / part[2]

                for start in np.arange(part[0], part[1], AUDIO_CHUNK_IN_SECONDS):
                    end = min(part[1], start + AUDIO_CHUNK_IN_SECONDS)
                    temp_audio.writeframes(read_bytes_from_wave(file, start, end))

    ffmpeg(f"-i {temp_final_audio_path} {final_audio_path}")
    tempory_video_path = tpath("tempory_video.mkv")

    v2timecodes_path, video_path2 = tpath("timecodes.v2"), tpath("v2video.mkv")
    v2timecodes = v1timecodes_to_v2timecodes(v1timecodes, video.fps, video.reader.nframes)
    save_v2_timecodes_to_file(v2timecodes_path, v2timecodes)

    # logger.log(1, f"mkvmerge -o {video_path2} --timestamps 0:{v2timecodes_path} {temp_images_path}")
    os.system(f"mkvmerge -o {tempory_video_path} --timestamps 0:{v2timecodes_path} -A {input_video_path} {final_audio_path}")

    if is_result_cfr:
        logger.log(1, "CFR-ing video")
        temporary_vfr_video_path = tpath("tempory_vfr_video.mkv")
        os.renames(tempory_video_path, temporary_vfr_video_path)
        ffmpeg(f"-i {temporary_vfr_video_path} -c:a copy {tempory_video_path}")

    if os.path.exists(output_video_path):
        os.remove(output_video_path)
        # If os.path.exists(output_video_path) is True and overwrite_output_force is False
        # program quits earlier
    shutil.move(tempory_video_path, output_video_path)

    if need_to_remove_working_directory_tree:
        video.audio.reader.close_proc()
        video.reader.close()  # https://stackoverflow.com/a/45393619
        # print("Here")
        # If function deletes directory before deleting a video, video deletion raises an error.

    delete_all_sva4_temporary_objects()


def delete_all_sva4_temporary_objects():
    """
    When process_one_video_in_computer or apply_calculated_interesting_to_video
    creates temporary directory or temporary file its name starts with
    TEMPORARY_DIRECTORY_PREFIX="SVA4_" for easy identification.
    If user terminates process function doesn't delete directory, cause of it terminated.
    So, function delete_all_tempories_sva4_directories deletes all directories and files which
    marked with prefix TEMPORARY_DIRECTORY_PREFIX
    :return: None
    """
    temp_dirs = [f for f in os.scandir(gettempdir()) if (f.is_dir() or f.is_file())]
    sva4_temp_dirs = [f for f in temp_dirs if f.name.startswith(TEMPORARY_DIRECTORY_PREFIX)]
    for temp_path in sva4_temp_dirs:
        full_path = os.path.join(gettempdir(), temp_path.path)
        print(f"Deleting {full_path}")
        if temp_path.is_dir():
            shutil.rmtree(full_path)
        elif temp_path.is_file():
            os.remove(full_path)
