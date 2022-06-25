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
 'AlgAnd'
 'AlgOr'
 'AlgNot'
"""
import math
import os, sys
import tempfile
import wave
import numpy as np
from typing import List
if 'google.colab' in sys.modules:
    tqdm = lambda x, *args, **kwargs: x
else:
    from tqdm.auto import tqdm
from audio import save_audio_to_wav, WavFile, AUDIO_CHUNK_IN_SECONDS, PartsOfAudio
from ffmpeg_caller import FFMPEGCaller
from some_functions import str2error_message, get_duration, TEMPORARY_DIRECTORY_PREFIX



def do_nothing(*args, **kwargs):
    pass


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

        :param video_path: str
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
        wav_audio = WavFile(wav_audio_path)
        return self.get_interesting_parts_from_wav(wav_audio)

    def get_interesting_parts_from_wav(self, wav_audio: WavFile):
        msg = f"All classes inherited from {__class__} must overload" + \
              " get_interesting_parts_from_wav method"
        raise AttributeError(msg)


class PiecemealWavSoundAlgorithm(WavSoundAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that uses only sound.
    """
    def __init__(self, chunk_in_seconds: float = AUDIO_CHUNK_IN_SECONDS):
        self.chunk = chunk_in_seconds
        super(PiecemealWavSoundAlgorithm, self).__init__()

    def get_interesting_parts_from_wav(self, wav_audio):
        interesting_parts = []
        for start in tqdm(np.arange(0, wav_audio.duration, self.chunk), leave=False, desc=str(self)):
            end = min(start + self.chunk, wav_audio.duration)

            wav_part = wav_audio.subclip(start, end)
            chunk_interesting_parts = np.array(self.get_interesting_parts_from_wav_part(wav_part))
            if chunk_interesting_parts.size:
                interesting_parts.append(start + chunk_interesting_parts)

        return np.vstack(interesting_parts) if interesting_parts else np.zeros((0, 2))

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavFile):
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

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavFile):
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
                 frame_duration: int = 30,
                 chunk_in_seconds: float = 60):
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

        super(WebRtcVADAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavFile):
        from webrtcvad import Vad

        array, old_fps = wav_audio_chunk.to_soundarray(), wav_audio_chunk.fps
        index = np.arange(0, len(array), old_fps / self.sample_rate).astype(int)
        array = array[index]
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


class SileroVadAlgorithm(PiecemealWavSoundAlgorithm):
    """
    This algorithm selects speech from text using VAD algorithm
    from this (https://github.com/snakers4/silero-vad) project
    and returns them as interesting parts.
    """
    def __init__(self, *vad_args, onnx: bool = True, chunk_in_seconds: int = 60, **vad_kwargs):
        import torch

        super(SileroVadAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=False,
                                                onnx=onnx)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

        self.vad_args, self.vad_kwargs = vad_args, vad_kwargs

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavFile):
        import torchaudio
        import torch
        available_rate = 16000
        sound = wav_audio_chunk.to_soundarray()[:, 0]
        sound = torch.tensor(sound, dtype=torch.float32)

        transform = torchaudio.transforms.Resample(orig_freq=wav_audio_chunk.fps,
                                                   new_freq=available_rate)
        sound = transform(sound)   # https://github.com/snakers4/silero-vad/blob/76687cbe25ffdf992ad824a36bfe73f6ae1afe72/utils_vad.py#L86

        dict_of_interesting_parts = self.get_speech_timestamps(
            sound,
            self.model,
            *self.vad_args,
            **self.vad_kwargs
        )
        # Todo I don't by what value we should divide timestamps. 16000 works.
        #  It should be replaced by an expression depending on vad_args, vad_kwargs
        #  https://t.me/silero_speech/1392
        list_of_interesting_parts = [[elem['start'] / available_rate, elem['end'] / available_rate]
                                     for elem in dict_of_interesting_parts]
        return np.array(list_of_interesting_parts)

    def __str__(self):
        answer = f"{type(self).__name__}("
        if self.vad_args:
            answer += f"vad_args={self.vad_args}, "
        if self.vad_kwargs:
            answer += f"vad_kwargs={self.vad_kwargs}, "
        if answer.endswith(", "):
            answer = answer[:-2]
        answer += ")"
        return answer


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
        return self.reverse(interesting_parts, get_duration(video_path))

    def get_interesting_parts_from_wav(self, wav_audio_chunk: WavFile):
        interesting_parts = self.alg.get_interesting_parts_from_wav(wav_audio_chunk)
        return self.reverse(interesting_parts, wav_audio_chunk.duration)

    @staticmethod
    def reverse(interesting_parts: List[List[int]], duration: float):
        begins_timestamps, ends_timestamps = interesting_parts[:, 0], interesting_parts[:, 1]
        new_begins_timestamps = np.hstack(([0], ends_timestamps))
        new_ends_timestamps = np.hstack((begins_timestamps, [duration]))
        return np.vstack((new_begins_timestamps, new_ends_timestamps)).transpose((1, 0))

    def __str__(self):
        return f"(not {self.alg})"


class AlgAnd2(SpeedUpAlgorithm):
    def __init__(self, *algs: List[WavSoundAlgorithm]):
        self.algs = algs

    def get_interesting_parts(self, video_path: str):
        duration = get_duration(video_path)
        parts = [[0, duration]]
        for alg in self.algs:
            parts = __class__.get_interesting_parts_only_in_intervals(alg, video_path, parts)
        return parts

    @staticmethod
    def get_interesting_parts_only_in_intervals(
             wavsoundalg: WavSoundAlgorithm,
             video_path: str,
             intervals: List[List[float]]
     ) -> List[List[float]]:
         wavaudio_path = save_audio_to_wav(video_path)
         parts_of_audio = PartsOfAudio(WavFile(wavaudio_path), intervals)
         inter_parts = wavsoundalg.get_interesting_parts_from_wav(parts_of_audio)
         return np.array(parts_of_audio.convert_from_self_tl(inter_parts))


class AlgAnd0(SpeedUpAlgorithm):
    def __init__(self, *args: WavSoundAlgorithm, minimal_lenght_of_inter_part_sec=0.15, logger_func=print):
        self.args = args
        self.minimal_lenght_of_inter_part_sec = minimal_lenght_of_inter_part_sec
        self.print = logger_func

    def get_interesting_parts(self, video_path: str):
        wavaudio_path = save_audio_to_wav(video_path)
        audio = WavFile(wavaudio_path)
        parts = np.array([[0, audio.duration]])
        for index, alg in enumerate(self.args):
            # print(f"Applying {str(alg)}\nfrom {math.ceil(remaining_duration / AUDIO_CHUNK_IN_SECONDS)}: ", end="")  #
            cur_duration = 0  #

            new_parts = []
            for part in parts:
                new_parts.append([part[0]])
                for i in range(math.floor((part[1] - part[0]) / AUDIO_CHUNK_IN_SECONDS)):
                    border = part[0] + i * AUDIO_CHUNK_IN_SECONDS
                    new_parts[-1].append(border)
                    new_parts.append([border])
                new_parts[-1].append(part[1])
            parts = new_parts
            new_parts = []
            for part in tqdm(parts, leave=False):
                if part[1] - part[0] < self.minimal_lenght_of_inter_part_sec:
                    continue
                new_parts.append(part[0] + alg.get_interesting_parts_from_wav(audio.subclip(*part)))

                cur_duration += part[1] - part[0]  #
                # if math.floor((cur_duration - (part[1] - part[0])) / AUDIO_CHUNK_IN_SECONDS) < math.floor(cur_duration / AUDIO_CHUNK_IN_SECONDS): #
                #     print(math.floor(cur_duration / AUDIO_CHUNK_IN_SECONDS), end=", ")  #
            # print()  #

            parts = np.vstack(new_parts)
        return parts


class AlgAnd1(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns pieces of parts that all algorithms
     select as interesting.
    Syntaxis:
        alg = AlgAnd(alg1, alg2, alg3 ... algn)

    """
    def __init__(self, *algorithms):
        super(__class__, self).__init__()
        self.algs = algorithms

    def get_interesting_parts(self, path: str):
        # list_of_ip = list of interesting parts
        rt_interesting_parts, n = [], len(self.algs)

        list_of_ip = [0] * n
        for i, alg in enumerate(self.algs):
            # print(f"Calculating interesting parts using {alg}")
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
    def __init__(self, *algorithms, fast=True):
        super(AlgOr, self).__init__()
        self.algs = algorithms
        self.real_algorithm = AlgNot(AlgAnd(*[AlgNot(alg) for alg in algorithms], fast=fast))

    def get_interesting_parts(self, video_path: str):
        return self.real_algorithm.get_interesting_parts(video_path)

    def __str__(self):
        result = " or ".join(map(str, self.algs))
        return f"({result})"


class RemoveShortParts(SpeedUpAlgorithm):
    def __init__(self, alg: SpeedUpAlgorithm, min_part_lenght: float = 0.15):
        super(RemoveShortParts, self).__init__()
        self.alg = alg
        self.min_part_lenght = min_part_lenght

    def get_interesting_parts(self, video_path: str):
        parts = self.alg.get_interesting_parts(video_path)
        parts = parts[parts[:, 1] - parts[:, 0] > self.min_part_lenght, :]
        return parts


class SpecifiedParts(SpeedUpAlgorithm):
    def __init__(self, interesting_parts):
        super(SpecifiedParts, self).__init__()
        self.interesting_parts = np.array(interesting_parts, dtype="float64")

    def get_interesting_parts_from_wav(self, wavaudio):
        return np.minimum(self.interesting_parts, wavaudio.duration)

    def get_interesting_parts(self, video_path: str):
        duration = 10 ** 10 if not video_path else get_duration(video_path)
        return np.minimum(self.interesting_parts, duration)

    def __str__(self):
        return f"{__class__.__name__}({self.interesting_parts.tolist()})"


class CropLongSounds(SpeedUpAlgorithm):
    def __init__(self, max_lenght_of_one_sound=0.05, threshold=0.995):
        self.step = max_lenght_of_one_sound
        self.threshold = threshold
        super(CropLongSounds, self).__init__()

    def get_interesting_parts(self, video_path: str):
        import librosa as librosa
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


class SubtitlesParts(SpeedUpAlgorithm):
    def __init__(self, pred_extend=0.1, after_extend=0.25):
        self.pred_extend = pred_extend
        self.after_extend = after_extend
        super(SubtitlesParts, self).__init__()

    def get_interesting_parts(
            self,
            video_path: str,
            ffmpeg_caller=FFMPEGCaller(hide_output=True, print_command=True, overwrite_force=True)
    ):
        temp_file_name = os.path.join(tempfile.gettempdir(), TEMPORARY_DIRECTORY_PREFIX + "temp_subtitles.srt")
        ffmpeg_caller(f"-i {video_path} -map 0:s:0 {temp_file_name}")
        with open(temp_file_name, "r") as f:
            subtitles = f.readlines()

        timecodes_strings = subtitles[1::4]

        def timecode2float(s: str):
            s = s.strip()
            hr, min, sec = s.split(":")
            sec = sec.replace(",", ".")
            return int(hr) * 60 ** 2 + int(min) * 60 + float(sec)

        def timecode_str2float(timecode_str: str):
            start, end = timecode_str.split("-->")
            return [timecode2float(start), timecode2float(end)]

        timecodes = np.array(list(map(timecode_str2float, timecodes_strings)))
        timecodes[:, 0] -= self.pred_extend
        timecodes[:, 1] += self.after_extend
        timecodes = _apply_min_quiet_time_to_interesting_parts_array(0, timecodes)
        return timecodes


def AlgAnd(*algs: SpeedUpAlgorithm, fast=True):
    if all(isinstance(alg, PiecemealWavSoundAlgorithm) for alg in algs) and fast:
        return AlgAnd0(*algs)
    return AlgAnd1(*algs)
