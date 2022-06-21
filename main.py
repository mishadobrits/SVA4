"""
This module provides the main functions of this project - 'process_one_video_from_computer' and .
It processes the video in the way README says.
"""
import itertools
import json
import os
import random
import shutil
import subprocess
import warnings
from pathlib import Path
from tempfile import gettempdir
import logging
from wave import Wave_read, Wave_write
import numpy as np

from audio import save_audio_to_wav, read_bytes_from_wave, AUDIO_CHUNK_IN_SECONDS
from settings import Settings
from some_functions import (
    ffmpeg_atempo_filter, input_answer, TEMPORARY_DIRECTORY_PREFIX, create_valid_path, get_nframes, get_duration,
    get_working_directory_path, VideoV2Timecodes,
)
from ffmpeg_caller import FFMPEGCaller
from speed_up import SpeedUpAlgorithm, SpecifiedParts, AlgAnd
from multiprocessing.pool import ThreadPool as Pool


def prepare_audio(
        input_audio_path: str,
        settings: Settings,
        working_directory: str,
        ffmpeg_caller: FFMPEGCaller,
):
    """
    Calls ffmpeg to create in working_diretory boring_audio.wav with speed
    settings.get_real_loud_speed() and interesting_audio.wav with speed
    settings.get_real_quiet_speed() in parallel.
    Returns absolute pathes of boring_audio.wav, nteresting_audio.wav
    
    :param input_audio_path: 
    :param settings: 
    :param working_directory: 
    :param ffmpeg_caller: 
    :return: 
    """
    boring_audio_path = os.path.join(working_directory, "boring_audio.wav")
    interesting_audio_path = os.path.join(working_directory, "interesting_audio.wav")
    inter_speed = settings.get_real_loud_speed()
    boring_speed = settings.get_real_quiet_speed()

    def get_speeded_audio(input_audio_path, speed, output_filename, ffmpeg_caller):
        if os.path.exists(output_filename):
            return

        try:
            ffmpeg_caller(
                f"-i {input_audio_path} -vn {ffmpeg_atempo_filter(speed)} {output_filename}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    pool = Pool()
    pool.apply_async(get_speeded_audio, [input_audio_path, boring_speed, boring_audio_path, ffmpeg_caller])
    pool.apply_async(get_speeded_audio, [input_audio_path, inter_speed, interesting_audio_path, ffmpeg_caller])
    pool.close()
    pool.join()
    return boring_audio_path, interesting_audio_path


def process_one_video_in_computer(
        input_video_path: str,
        speedup_algorithm: SpeedUpAlgorithm,
        settings: Settings,
        output_video_path: str,
        is_result_cfr: bool = False,
        logger: logging.Logger = logging.getLogger("process_one_video_in_computer"),
        working_directory_path=None,
        ffmpeg_caller: FFMPEGCaller = FFMPEGCaller(),
        ffmpeg_preprocess_audio: str = "-filter:a dynaudnorm",
        audiocodec: str = "flac",
        videochunk_sec: float = 60 * 10,
):
    overwrite_output_force = ffmpeg_caller.get_overwrite_force()
    if os.path.exists(output_video_path) and overwrite_output_force is None:
        msg = "Output file is already exists and ffmpeg_caller.overwrite_force is None."
        msg += " Overwrite it?"
        answer = input_answer(msg, ["y", "Y", "n", "N"])
        overwrite_output_force = answer.lower() == "y"
        ffmpeg_caller.set_overwrite_force(overwrite_output_force)

    working_directory_path = get_working_directory_path(working_directory_path)
    video_with_wav_auido = os.path.join(
        working_directory_path, TEMPORARY_DIRECTORY_PREFIX + "video_with_wavaudio.mkv"
    )

    duration = get_duration(input_video_path)
    out_pathes = []
    for start in np.arange(0, duration, videochunk_sec):
        end = min(start + videochunk_sec, duration)
        name = f"{round(start, 2)}-{round(end, 2)}-{random.randint(0, 10**10)}"
        name = TEMPORARY_DIRECTORY_PREFIX + name

        in_filepath = os.path.join(working_directory_path, name + "-inpfile.mkv")
        ffmpeg_caller(f'-i "{input_video_path}" -c copy -ss {start} -to {end} "{in_filepath}"')

        out_filepath = os.path.join(working_directory_path, name + "-outfile.mkv")
        out_pathes.append(out_filepath)

        folder_filepath = os.path.join(working_directory_path, name + "-folder")
        os.mkdir(folder_filepath)

        new_logger = logger.getChild(f"{round(start, 2)}-{round(end, 2)}")

        process_one_videochunk_in_computer(
            in_filepath,
            speedup_algorithm,
            settings,
            out_filepath,
            working_directory_path=folder_filepath,
            is_result_cfr=is_result_cfr,
            logger=new_logger,
            ffmpeg_preprocess_audio=ffmpeg_preprocess_audio,
            ffmpeg_caller=ffmpeg_caller,
        )

    if not output_video_path.endswith(".mkv"):
        output_video_path += ".mkv"
        warnings.warn("output_video_path must end with '.mkv'")

    args = ["mkvmerge", "-o", video_with_wav_auido]
    for elem in out_pathes:
        args.append(elem)
        args.append("+")
    args.pop()

    subprocess.call(args)
    if audiocodec != "pcm_s16le":
        print(f'-i "{video_with_wav_auido}" -c:v copy -acodec {audiocodec} "{output_video_path}"')
        ffmpeg_caller(f'-i "{video_with_wav_auido}" -c:v copy -acodec {audiocodec} "{output_video_path}"')
    else:
        shutil.move(video_with_wav_auido, output_video_path)
    delete_all_sva4_temporary_objects()


def process_one_videochunk_in_computer(
    input_video_path: str,
    speedup_algorithm: SpeedUpAlgorithm,
    settings: Settings,
    output_video_path: str,
    is_result_cfr: bool = False,
    logger: logging.Logger = logging.getLogger("process_one_video_in_computer"),
    working_directory_path=None,
    ffmpeg_caller: FFMPEGCaller = FFMPEGCaller(),
    ffmpeg_preprocess_audio: str = "-filter:a dynaudnorm",

):
    """
    This function processes video (
        1) Extracts audio using "ffmpeg -i {inp_path} {ffmpeg_preprocess_audio} -ar 44100 path/audio.wav".
        2) Uses 'speedup_algorithm' to get interesting parts of video.
        3) Calls apply_calculated_interesting_to_video function.
    )
    Param ffmpeg_preprocess_audio:
        Applied in audio extraction in cmd
             "ffmpeg -i {inp_path} {ffmpeg_preprocess_audio} -ar 44100 path/audio.wav".
        Main examples:
            '' - No filter.
                Takes 0 additional time, recommended using if you're sure about your speed up algorithm.
            '-filter:a dynaudnorm'. Applies the dynaudnorm ffmpeg filter (normalizes volume in audio),
                which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
                Noise volume and very quiet speech increases not enough to hear.
                Takes ~minute to complete for 80m 1GB video.
            '-filter:a loudnorm' Applies the loudnorm ffmpeg filter (normalizes volume in audio),
                which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
                Noise volume and very quiet speech increases enough to hear.
                Takes ~10 minutes to complete for 80m 1GB video.
            '-filter:a "volume=1.5"' Increases volume in 1.5 time.
                Takes ~20 sec to complete for 80m 1GB video.
            '-filter:a "volume=10dB"' Increases volume by 10 dB.
                Takes ~20 sec to complete for 80m 1GB video.
    """
    video_path2 = create_valid_path(input_video_path)
    video_path3 = os.path.join(gettempdir(), TEMPORARY_DIRECTORY_PREFIX + str(random.randint(0, int(1E10))) + ".mkv")
    ffmpeg_caller(f"-i {video_path2} -c copy {video_path3}")
    working_directory_path = get_working_directory_path(working_directory_path)

    def get_interesting_parts_function(return_dict: dict):
        print("  Splitting audio into boring / interesting parts")
        try:
            return_dict["ip"] = speedup_algorithm.get_interesting_parts(video_path3)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    input_wav = save_audio_to_wav(video_path3, ffmpeg_preprocess_audio)
    interesting_parts = {}

    pool = Pool()
    pool.apply_async(prepare_audio, (input_wav, settings, working_directory_path, ffmpeg_caller))
    pool.apply_async(get_interesting_parts_function, args=(interesting_parts,))
    pool.close()
    pool.join()
    interesting_parts = interesting_parts["ip"]

    # np.save("interesting_parts.npy", interesting_parts)
    apply_calculated_interesting_to_video(
        interesting_parts,
        settings,
        video_path3,
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
    overwrite_output_force = True
    if os.path.exists(output_video_path) and not overwrite_output_force:
        logger.log(1, f"File {output_video_path} is already exists and overwrite_output_force = False")
        logger.log(1, "Quiting")
        return

    def tpath(filename):
        """
        returns the absolute path for a file with name filename in folder working_directory_path
        """
        return os.path.join(working_directory_path, filename)

    working_directory_path = get_working_directory_path(working_directory_path)

    if " " in os.path.abspath(input_video_path):
        new_video_path = tpath(f"input_video.{os.path.splitext(input_video_path)}")
        shutil.copyfile(input_video_path, new_video_path)
        input_video_path = new_video_path

    nframes = get_nframes(input_video_path)

    interesting_parts, boring_parts = settings.process_interestingpartsarray(interesting_parts)
    interesting_parts[:2] = np.minimum(nframes - 1, interesting_parts[:2])
    boring_parts[:2] = np.minimum(nframes - 1, boring_parts[:2])

    boring_parts = np.hstack(
        [boring_parts, settings.get_real_quiet_speed() * np.ones((len(boring_parts), 1))]
    )
    interesting_parts = np.hstack(
        [interesting_parts, settings.get_real_loud_speed() * np.ones((len(interesting_parts), 1))]
    )
    input_wav = save_audio_to_wav(input_video_path)
    boring_audio_path, interesting_audio_path = prepare_audio(
        input_wav, settings, working_directory_path, ffmpeg_caller
    )
    final_audio_path = tpath("temp_audio.wav")

    v1timecodes = [[0, 0, 1]]
    v2timecodes = VideoV2Timecodes(input_video_path, working_directory=working_directory_path)
    with Wave_read(boring_audio_path) as boring_audio,\
            Wave_read(interesting_audio_path) as interesting_audio,\
            Wave_write(final_audio_path) as temp_audio:
        temp_audio.setparams(boring_audio.getparams())
        parts_iterator = itertools.zip_longest(boring_parts, interesting_parts)
        cur_time = 0
        for boring_and_interesting_part in parts_iterator:
            parts_with_file = zip([boring_audio, interesting_audio], boring_and_interesting_part)
            for file, part in parts_with_file:
                if part is None:
                    continue

                start_frame = v2timecodes.get_frame_number(part[0])
                end_frame = v2timecodes.get_frame_number(part[1])
                speed = part[2]
                start_sec, end_sec = v2timecodes[start_frame] / 1000, v2timecodes[end_frame] / 1000
                cur_time += (end_sec - start_sec) / speed
                v1timecodes.append([v1timecodes[-1][1], start_frame, 10 ** 7])
                v1timecodes.append([start_frame, end_frame, part[2]])

                for start in np.arange(start_sec / speed, end_sec / speed, AUDIO_CHUNK_IN_SECONDS):
                    end = min(end_sec / speed, start + AUDIO_CHUNK_IN_SECONDS)
                    temp_audio.writeframes(read_bytes_from_wave(file, start, end))

        v1timecodes.append([end_frame, len(v2timecodes), 10 ** 7])
        # print(cur_time, temp_audio.getnframes() / temp_audio.getframerate())

    tempory_video_path = tpath("tempory_video.mkv")

    v2timecodes_path, video_path2 = tpath("timecodes.v2"), tpath("v2video.mkv")
    v2timecodes.apply_v1_timecodes(v1timecodes)
    # print(v2timecodes[-1] / 1000)
    v2timecodes.save(v2timecodes_path)

    global_tracks_info_str = subprocess.check_output(['mkvmerge', '-J', input_video_path])
    global_tracks_info_json = json.loads(global_tracks_info_str)
    for track_info in global_tracks_info_json["tracks"]:
        if track_info["type"] == "video":
            video_track_id = track_info["id"]
    os.system(f"mkvmerge -o {tempory_video_path} --timestamps {video_track_id}:{v2timecodes_path} -A {input_video_path} {final_audio_path}")

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

    # delete_all_sva4_temporary_objects(working_directory_path)


def delete_all_sva4_temporary_objects(path=gettempdir()):
    """
    When process_one_video_in_computer or apply_calculated_interesting_to_video
    creates temporary directory or temporary file its name starts with
    TEMPORARY_DIRECTORY_PREFIX="SVA4_" for easy identification.
    If user terminates process function doesn't delete directory, cause of it terminated.
    So, function delete_all_tempories_sva4_directories deletes all directories and files which
    marked with prefix TEMPORARY_DIRECTORY_PREFIX
    :return: None
    """
    temp_dirs = [f for f in os.scandir(path) if (f.is_dir() or f.is_file())]
    sva4_temp_dirs = [f for f in temp_dirs if f.name.startswith(TEMPORARY_DIRECTORY_PREFIX)]
    for temp_path in sva4_temp_dirs:
        full_path = os.path.join(gettempdir(), temp_path.path)
        print(f"Deleting {full_path}")
        try:
            if temp_path.is_dir():
                shutil.rmtree(full_path)
            elif temp_path.is_file():
                os.remove(full_path)
        except Exception as e:
            print(f"Cannot delete {full_path} due to error {e}")
