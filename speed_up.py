"""
This module contains algorithms that get moviepy_video_object and return a list of interesting parts
in format [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ...
[start_of_piecen, end_of_piecen]]
All values should be positions in video in seconds.
All algorithms must be inherited from the base class 'SpeedUpAlgorithm.'
Currently, there are
 'VolumeThresholdAlgorithm(sound_threshold)'
 'WebRtcVADAlgorithm(aggressiveness)'
 'SileroVadAgorithm()'


"""
import math
import os
import random
import wave
from math import ceil
from tempfile import NamedTemporaryFile

import librosa as librosa
import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from some_functions import str2error_message, TEMPORARY_DIRECTORY_PREFIX, save_audio_to_wav, WavSubclip


def _apply_min_quiet_time_to_interesting_parts_array(min_quiet_time, interesting_parts):
    begin_sound_indexes = interesting_parts[:, 0]
    end_sound_indexes = interesting_parts[:, 1]

    end_sound_indexes[:-1] += min_quiet_time

    is_changing = begin_sound_indexes[1:] > end_sound_indexes[:-1]
    if not is_changing.size:
        return interesting_parts.reshape((-1, 2))

    begin_sound_indexes = begin_sound_indexes[np.hstack([True, is_changing])]
    end_sound_indexes = end_sound_indexes[np.hstack([is_changing, True])]

    interesting_parts = np.vstack([begin_sound_indexes, end_sound_indexes])
    return interesting_parts.transpose((1, 0))


class SpeedUpAlgorithm:
    """
    Base class for all Algorithms
    """
    def __init__(self):
        pass

    def get_interesting_parts(self, video_path: str):
        """
        All classes inherited from SpeedUpAlgorithm must overload get_interesting_parts method,
        because this method used by main.apply_calculated_interesting_to_video and
        main.process_one_video_in_computer

        :param moviepy_video: VideoClip
        :return: np.array of interesting parts in usual format
                (format in look settings.process_interestingpartsarray.__doc__)
        """
        msg = f"All classes inherited from {__class__} must overload get_loud_parts method"
        raise AttributeError(str2error_message(msg))

    def __str__(self):
        return f"{type(self).__name__}"


class WavSoundAlgorithm(SpeedUpAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that uses only sound.
    """
    def get_interesting_parts(self, video_path: str):
        wav_audio_path = save_audio_to_wav(video_path)
        wav_audio = WavSubclip(wav_audio_path)
        return self.get_interesting_parts_from_wav(wav_audio)

    def get_interesting_parts_from_wav(self, wav_audio: WavSubclip):
        msg = f"All classes inherited from {__class__} must overload" + \
              " get_interesting_parts_from_wav method"
        raise AttributeError(msg)


class PiecemealWavSoundAlgorithm(WavSoundAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that uses only sound.
    """
    def __init__(self, chunk_in_seconds: float = 60):
        self.chunk = chunk_in_seconds
        super(PiecemealWavSoundAlgorithm, self).__init__()

    def get_interesting_parts_from_wav(self, wav_audio):
        interesting_parts = []
        print(f"from {math.ceil(wav_audio.duration / self.chunk)}: ", end="")
        for start in np.arange(0, wav_audio.duration, self.chunk):
            print(round(start / self.chunk), end=", ")
            end = min(start + self.chunk, wav_audio.duration)
            wav_part = wav_audio.subclip(start, end)

            chunk_interesting_parts = np.array(self.get_interesting_parts_from_wav_part(wav_part))
            interesting_parts.append(start + chunk_interesting_parts)
        print()
        return np.vstack(interesting_parts)

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        msg = f"All classes inherited from {__class__} must overload" + \
              " get_interesting_parts_from_wav method"
        raise AttributeError(msg)


class VolumeThresholdAlgorithm(PiecemealWavSoundAlgorithm):
    """
    Returns pieces where volume >= sound_threshold as interesting parts

    min_quiet_time - the program doesn't accelerate the
     first min_quiet_time seconds in each of boring piece.

    """
    def __init__(self,
                 sound_threshold: float,
                 min_quiet_time: float = 0.25,
                 chunk_in_seconds: float = 60):
        self.sound_threshold = sound_threshold
        self.min_q_time = min_quiet_time
        super(VolumeThresholdAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def set_sound_threshold(self, value: float):
        self.sound_threshold = value

    def get_sound_threshold(self):
        """:returns sound_threshold: float"""
        return self.sound_threshold

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        sound = np.abs(wav_audio_chunk.to_soundarray())
        sound = sound.max(axis=1).reshape(-1)
        sound = np.hstack([-1, sound, self.sound_threshold + 1, -1])

        is_voice = (sound > self.sound_threshold).astype(int)
        borders = is_voice[1:] - is_voice[:-1]
        begin_sound_indexes = np.arange(len(borders))[borders > 0]
        end_sound_indexes = np.arange(len(borders))[borders < 0]

        interesting_parts = np.vstack([begin_sound_indexes, end_sound_indexes])
        interesting_parts = interesting_parts.transpose((1, 0)) / wav_audio_chunk.fps
        return _apply_min_quiet_time_to_interesting_parts_array(self.min_q_time, interesting_parts)

    def __str__(self):
        return f"{type(self).__name__}(sound_threshold={self.get_sound_threshold()})"


"""
class EnergyThresholdAlgorithm(PiecemealSoundAlgorithm):
    def __init__(self, energy_threshold, audio_chunk):
        super(EnergyThresholdAlgorithm, self).__init__()
        pass  # todo
"""


class WebRtcVADAlgorithm(PiecemealWavSoundAlgorithm):
    """
    This algorithm selects speech from video using Voice Activity Detection (VAD)
    algorithm coded by google (link https://github.com/wiseman/py-webrtcvad)
    and returns them as interesting parts
    """
    def __init__(self,
                 aggressiveness: int = 1,
                 min_quiet_time: float = 0.25,
                 frame_duration: int = 30):
        """

        :param aggressiveness: parameter to VAD
        :param min_quiet_time: as usual
        :param frame_duration: must be 10, 20 or 30 - VAD parameter
        :param chunk_in_seconds:
        """
        try:
            import webrtcvad
        except ImportError as import_error:
            err = ImportError("WebRtcVADAlgorithm algorithm requires installed 'webrtcvad' module")
            raise err from import_error

        self.aggressiveness = None
        self.sample_rate = 48000
        self.frame_duration = frame_duration  # ms

        self.min_quiet_time = min_quiet_time
        self.set_aggressiveness(aggressiveness)

        super(WebRtcVADAlgorithm, self).__init__(chunk_in_seconds=60)

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        from webrtcvad import Vad

        array, old_fps = wav_audio_chunk.to_soundarray(), wav_audio_chunk.fps
        array = AudioArrayClip(array, old_fps).set_fps(self.sample_rate).to_soundarray()[:, 0]
        array = (abs(array) * 2 ** 16).astype("int16")

        prev_value = False
        chunk = 2 * int(self.sample_rate * self.frame_duration / 1000)
        array = np.hstack([[0] * chunk, array, [0] * 2 * chunk])
        begins_of_speech, ends_of_speech = [], []
        for i in range(0, len(array) - chunk, chunk):
            cur_sound = array[i: i + chunk]
            value = Vad(self.aggressiveness).is_speech(cur_sound.data, self.sample_rate)
            # I tried self.vad.is_speech(cur_sound.data, sample_rate),
            # but some how it isn't the same.
            if value and not prev_value:
                begins_of_speech.append(i / self.sample_rate)
            if not value and prev_value:
                ends_of_speech.append(i / self.sample_rate)
            prev_value = value

        interesting_parts = np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0))
        return _apply_min_quiet_time_to_interesting_parts_array(self.min_quiet_time,
                                                                interesting_parts)

    def get_aggressiveness(self):
        """:returns aggressiveness: int 0, 1, 2 or 3"""
        return self.aggressiveness

    def set_aggressiveness(self, aggressiveness: int):
        """sets aggressiveness = value"""
        assert aggressiveness in [0, 1, 2, 3],\
            f"aggressiveness must be 0, 1, 2 or 3. {aggressiveness} were given"
        self.aggressiveness = aggressiveness

    def __str__(self):
        return f"{type(self).__name__}({self.aggressiveness})"


class SileroVadAlgorithm(SpeedUpAlgorithm):
    """
    This algorithm selects speech from text using VAD algorithm
    from this (https://github.com/snakers4/silero-vad) project
    and returns them as interesting parts.
    """
    def __init__(self, *vad_args, **vad_kwargs):
        try:
            import torch
        except ImportError:
            msg = "the {} class requires installed torch module."
            raise ImportError(msg.format(__class__))
        super(SileroVadAlgorithm, self).__init__()

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad')
        (self.get_speech_ts,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

        self.vad_args, self.vad_kwargs = vad_args, vad_kwargs

    def get_interesting_parts(self, video_path: str):
        vad_func = self._get_vad_func()
        temporary_file_name = save_audio_to_wav(video_path)

        dict_of_interesting_parts = vad_func(temporary_file_name)
        # Todo I don't by what value we should divide timestamps. 16000 works.
        #  It should be replaced by an expression depending on vad_args, vad_kwargs
        #  https://t.me/silero_speech/1392
        list_of_interesting_parts = [[elem['start'] / 16000, elem['end'] / 16000]
                                     for elem in dict_of_interesting_parts]
        return np.array(list_of_interesting_parts)

    def _get_vad_func(self):
        """

        :return: is_speech_func: str: "path/to/wav" -> bool
        """
        return lambda wav_path: self.get_speech_ts(self.read_audio(wav_path),
                                                   self.model,
                                                   *self.vad_args,
                                                   **self.vad_kwargs)

    def __str__(self):
        self_str = f"{type(self).__name__}("
        if self.vad_args:
            self_str += f"vad_args={self.vad_args}, "
        if self.vad_kwargs:
            self_str += f"vad_args={self.vad_kwargs}, "
        if self_str.endswith(", "):
            self_str = self_str[:-2]
        self_str += ")"
        return self_str


class AlgNot(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns
     [pieces of parts that alg selects as boring] as interesting.
    Syntaxis:
        alg = AlgNot(alg)

    """
    def __init__(self, algorithm: SpeedUpAlgorithm):
        super(AlgNot, self).__init__()
        self.alg = algorithm

    def get_interesting_parts(self, video_path: str):
        interesting_parts = self.alg.get_interesting_parts(video_path)
        begins_timestamps, ends_timestamps = interesting_parts[:, 0], interesting_parts[:, 1]
        new_begins_timestamps = np.hstack(([0], ends_timestamps))
        new_ends_timestamps = np.hstack((begins_timestamps, [VideoFileClip(video_path).duration]))
        return np.vstack((new_begins_timestamps, new_ends_timestamps)).transpose((1, 0))

    def __str__(self):
        return f"(not {self.alg})"


class AlgAnd(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns pieces of parts that all algorithms
     select as interesting.
    Syntaxis:
        alg = AlgAnd(alg1, alg2, alg3 ... algn)

    """
    def __init__(self, *algorithms):
        super(AlgAnd, self).__init__()
        self.algs = algorithms

    def get_interesting_parts(self, path: str):
        # list_of_ip = list of interesting parts
        rt_interesting_parts, n = [], len(self.algs)

        list_of_ip = [0] * n
        for i, alg in enumerate(self.algs):
            print(f"Calculating interesting parts using {alg}")
            list_of_ip[i] = alg.get_interesting_parts(path)

        cur_alg_indexes = [0 for _ in range(n)]
        while all(cur_alg_indexes[i] < len(list_of_ip[i]) for i in range(n)):
            begins = [list_of_ip[i][cur_alg_indexes[i]][0] for i in range(n)]
            ends = [list_of_ip[i][cur_alg_indexes[i]][1] for i in range(n)]

            max_begin, is_intersection = max(begins), True
            for i in range(n):
                if list_of_ip[i][cur_alg_indexes[i]][1] <= max_begin:
                    cur_alg_indexes[i] += 1
                    is_intersection = False

            if is_intersection:
                min_end = min(ends)
                rt_interesting_parts.append((max_begin, min_end))
                cur_alg_indexes[ends.index(min_end)] += 1

        return np.array(rt_interesting_parts)

    def __str__(self):
        result = " and ".join(map(str, self.algs))
        return f"({result})"


class AlgOr(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns pieces of parts that at least one
     algorithm selects as interesting.
    Syntaxis:
        alg = AlgOr(alg1, alg2, alg3 ... algn)

    """
    def __init__(self, *algorithms):
        super(AlgOr, self).__init__()
        self.algs = algorithms
        self.real_algorithm = AlgNot(AlgAnd(*[AlgNot(alg) for alg in algorithms]))

    def get_interesting_parts(self, video_path: str):
        return self.real_algorithm.get_interesting_parts(video_path)

    def __str__(self):
        result = " or ".join(map(str, self.algs))
        return f"({result})"


class _FakeDebugAlgorithm(SpeedUpAlgorithm):
    def __init__(self, interesting_parts):
        super(_FakeDebugAlgorithm, self).__init__()
        self.interesting_parts = np.array(interesting_parts, dtype="float64")

    def get_interesting_parts(self, video_path: str):
        duration = 10 ** 10 if not video_path else VideoFileClip(video_path).duration
        return np.minimum(self.interesting_parts, duration)

    def __str__(self):
        return f"{__class__.__name__}({self.interesting_parts.tolist()})"


class CropLongSounds(SpeedUpAlgorithm):
    def __init__(self, max_lenght_of_one_sound=0.05, threshold=0.995):
        self.step = max_lenght_of_one_sound
        self.threshold = threshold
        super(CropLongSounds, self).__init__()

    def get_interesting_parts(self, video_path: str):
        def cdot(a, b):
            if type(a) == type(b) == int:
                return a * b
            return (a * b).sum()

        def cos(a, b):
            if not cdot(a, a) * cdot(b, b):
                return 1
            return cdot(a, b) / (cdot(a, a) * cdot(b, b)) ** 0.5

        temporary_file_name = save_audio_to_wav(video_path)
        spec = 1
        interesting_parts = []
        with wave.open(temporary_file_name) as input_audio:
            duration = input_audio.getnframes() / input_audio.getframerate()
            for i in np.arange(0, duration - self.step, self.step):
                if random.random() < 0.01:
                    print(i)
                sound, rate = librosa.load(temporary_file_name, offset=i, duration=self.step)
                prev_spec = spec
                spec = librosa.feature.mfcc(sound, rate)
                if cos(spec, prev_spec) < self.threshold:
                    interesting_parts.append([i, i + self.step])
        interesting_parts.append([duration - self.step, self.step])
        return np.array(interesting_parts)

    def __str__(self):
        s = f"{__class__.__name__}(max_lenght_of_one_sound={self.step}, threshold={self.threshold})"
        return s
