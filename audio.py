import bisect
import itertools
import math
import os
import traceback
from tempfile import gettempdir
from wave import Wave_read
import numpy as np
import wavio
from typing import List
from ffmpeg_caller import FFMPEGCaller
from some_functions import TEMPORARY_DIRECTORY_PREFIX

AUDIO_CHUNK_IN_SECONDS = 60


class Audio:
    def to_soundarray(self, start: float, end: float) -> np.array:
        raise NotImplementedError

    def subclip(self, start: float = 0, end: float = None):
        if end is None:
            end = self.duration
        return PartsOfAudio(self, [[start, end]])


class WavFile(Audio):
    def __init__(self, path: str, actual_duration: float = None):
        self.path = path
        with Wave_read(path) as wav_file:
            self.sample_width = wav_file.getsampwidth()
            self.nchannels = wav_file.getnchannels()
            self.fps = wav_file.getframerate()
            self.duration = wav_file.getnframes() / self.fps
        self.k = actual_duration / self.duration if actual_duration else 1

    def to_soundarray(self, start: float = 0, end: float = None) -> np.array:
        end = end if end else self.duration
        sample_width = self.sample_width
        sample_range = wavio._sampwidth_ranges[sample_width]
        read_bytes = read_bytes_from_wave(Wave_read(self.path), start / self.k, end / self.k)
        array = wavio._wav2array(self.nchannels, sample_width, read_bytes).astype("float64")

        # breakpoint()
        array = (array - sample_range[0]) / (sample_range[1] - sample_range[0])
        return 2 * array - 1  # fit to [-1, 1] range


class PartsOfAudio(Audio):
    def __init__(self, audio: Audio, intervals: List[List[float]], already_sorted=False):
        self.audio = audio
        self.intervals = np.array(intervals) if already_sorted else np.array(np.sort(intervals, axis=0))
        self.self_timeline = np.cumsum(self.intervals[:, 1] - self.intervals[:, 0])
        self.self_timeline = np.hstack([np.array([0]), self.self_timeline])
        self.duration = self.self_timeline[-1]
        self.fps = self.audio.fps

    def convert_from_self_tl(self, sorted_list_of_intervals: List[List[float]]):
        def get_time_in_audio_tl(t_in_self_tl, self_tl_ind):
            return t_in_self_tl - self.self_timeline[self_tl_ind] + self.intervals[self_tl_ind, 0]

        self_tl_intervals = np.minimum(sorted_list_of_intervals, self.duration - 0.001)
        self_tl_start_index = bisect.bisect_right(self.self_timeline, self_tl_intervals[0][0]) - 1

        rt = []
        self_tl_ind = self_tl_start_index
        for self_tl_interval in self_tl_intervals:
            audio_tl_interval = [float("inf"), float("inf") - 1]
            while self.self_timeline[self_tl_ind] < self_tl_interval[1]:
                if self.self_timeline[self_tl_ind] <= self_tl_interval[0] < self.self_timeline[self_tl_ind + 1]:
                    audio_tl_interval[0] = get_time_in_audio_tl(self_tl_interval[0], self_tl_ind)
                if self.self_timeline[self_tl_ind] <= self_tl_interval[1] < self.self_timeline[self_tl_ind + 1]:
                    audio_tl_interval[1] = get_time_in_audio_tl(self_tl_interval[1], self_tl_ind)

                left = max(self.intervals[self_tl_ind][0], audio_tl_interval[0])
                right = min(self.intervals[self_tl_ind][1], audio_tl_interval[1])
                # print(left, right)
                if left < right:
                    rt.append([left, right])
                self_tl_ind += 1
            self_tl_ind -= 1

        return rt

    def to_soundarray(self, start: float = 0, end: float = None, lo: int = 0, hi: int = None):
        end = end if end else self.duration
        # tl = timeline

        hi = hi if hi else len(self.self_timeline)
        start_ind = bisect.bisect_right(self.self_timeline, start, lo=lo, hi=hi) - 1
        start_in_audio_tl = start - self.self_timeline[start_ind] + self.intervals[start_ind, 0]
        end_ind = bisect.bisect_left(self.self_timeline, end, lo=lo, hi=hi) - 1
        end_in_audio_tl = end - self.self_timeline[end_ind] + self.intervals[end_ind, 0]

        if start_ind == end_ind:
            interval = [start_in_audio_tl, end_in_audio_tl]
            intervals = [interval]
        else:
            first_interval = [start_in_audio_tl, self.intervals[start_ind][1]]
            middle_intervals = self.intervals[start_ind + 1: end_ind]
            last_interval = [self.intervals[end_ind][0], end_in_audio_tl]
            intervals = itertools.chain([first_interval], middle_intervals, [last_interval])

        return np.vstack([self.audio.to_soundarray(*interv) for interv in intervals])


def save_audio_to_wav(input_video_path, ffmpeg_preprocess_audio="", ffmpeg_caller=FFMPEGCaller(overwrite_force=False, hide_output=True, print_command=True)):
    """
    Saves videos audio to wav and returns its path
    :param ffmpeg_preprocess_audio:
    :param input_video_path:
    :return: path od audio
    """
    input_video_path = os.path.abspath(input_video_path)
    filename = TEMPORARY_DIRECTORY_PREFIX + str(hash(input_video_path)) + ".wav"
    filepath = os.path.join(gettempdir(), filename)
    if os.path.exists(filepath):
        return filepath

    ffmpeg_caller(f'-i "{input_video_path}" -ar 48000 {ffmpeg_preprocess_audio} "{filepath}"')
    return filepath


def read_bytes_from_wave(waveread_obj: Wave_read, start_sec: float, end_sec: float):
    """
    Reads bytes from wav file 'waveread_obj' from start_sec up to end_sec.

    :param waveread_obj: Wave_read
    :param start_sec: float
    :param end_sec: float
    :return: rt_bytes: bytes: read bytes
    """
    previous_pos, framerate = waveread_obj.tell(), waveread_obj.getframerate()

    start_pos = min(waveread_obj.getnframes(), round(framerate * start_sec))
    end_pos = min(waveread_obj.getnframes(), round(framerate * end_sec))

    waveread_obj.setpos(start_pos)
    rt_bytes = waveread_obj.readframes(end_pos - start_pos)
    waveread_obj.setpos(previous_pos)

    return rt_bytes
