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
import os
from math import ceil
from tempfile import NamedTemporaryFile
import numpy as np
from moviepy.editor import VideoClip
from some_functions import str2error_message, TEMPORARY_DIRECTORY_PREFIX


class SpeedUpAlgorithm:
    """
    Base class for all Algorithms
    """
    def get_interesting_parts(self, moviepy_video: VideoClip):
        """
        All classes inherited from SpeedUpAlgorithm must overload get_interesting_parts method,
        because this method used by main.apply_calculated_interesting_and_boring_parts_to_video and
        main.process_one_video_in_computer.

        :param moviepy_video: VideoClip
        :return: np.array of interesting parts in usual format
                (format in look settings.process_interestingpartsarray.__doc__)
        """
        msg = f"All classes inherited from {__class__} must overload get_loud_parts method"
        raise AttributeError(str2error_message(msg))

    def __str__(self):
        return f"{type(self).__name__}"


class PiecemealBaseAlgorithm(SpeedUpAlgorithm):
    """
    If class inherited from PiecemealBaseAlgorithm, it must have get_interesting_parts_from_chunk
    method.
    When apply_calculated_interesting_and_boring_parts_to_video or process_one_video_in_computer
    calls 'get_interesting_parts(video)' method, 'get_interesting_parts' divides video into
    chunks with duration a 'chunk_in_seconds', applies a 'get_interesting_parts_from_chunk'
    to each of them, and concatenates the result.

    Public functions:
        __init__ (chunk_in_seconds):
        get_interesting_parts (moviepy_video):
    """
    def __init__(self, chunk_in_seconds: float = 60):
        self.chunk = chunk_in_seconds

    def get_interesting_parts(self, moviepy_video: VideoClip):
        video_chunks = [
            moviepy_video.subclip(i * self.chunk, min((i + 1) * self.chunk, moviepy_video.duration))
            for i in range(ceil(moviepy_video.duration / self.chunk))
        ]
        loud_parts = []
        print(f"from {len(video_chunks)}:", end=" ")
        for i, audio_chunk in enumerate(video_chunks):
            print(i, end=", ")
            prefix_duration = i * self.chunk
            chunk_interesting_parts = np.array(self.get_interesting_parts_from_chunk(audio_chunk))
            loud_parts.append(prefix_duration + chunk_interesting_parts)
        print()
        return np.vstack(loud_parts)

    def get_interesting_parts_from_chunk(self, video_chunk: VideoClip):
        msg = "All class inherited from {} must overload {} method. Class {} doesn't."
        msg = msg.format(__class__, "get_interesting_parts_from_chunk", type(self))
        raise AttributeError(msg)


class PiecemealSoundAlgorithm(PiecemealBaseAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that use only sound.
    """
    def get_interesting_parts(self, moviepy_video: VideoClip):
        class FakeVideo:
            """
            class contains only info about AudioClip, for faster working
            """
            def __init__(self, moviepy_video=None):
                self.audio = None
                self.duration = 0
                if moviepy_video:
                    self.set_audio(moviepy_video.audio)

            def set_audio(self, audio):
                self.audio = audio
                self.duration = audio.duration

            def subclip(self, start, end):
                fake_video = FakeVideo()
                fake_video.set_audio(self.audio.subclip(start, end))
                return fake_video

        fake_video = FakeVideo(moviepy_video)
        return super(PiecemealSoundAlgorithm, self).get_interesting_parts(fake_video)


class VolumeThresholdAlgorithm(PiecemealSoundAlgorithm):
    """
    Returns pieces where volume >= sound_threshold as interesting parts
    """
    def __init__(self, sound_threshold: float, chunk_in_seconds: float = 60):
        self.sound_threshold = sound_threshold
        super(VolumeThresholdAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def set_sound_threshold(self, value: float):
        self.sound_threshold = value

    def get_sound_threshold(self):
        """:returns sound_threshold: float"""
        return self.sound_threshold

    def get_interesting_parts_from_chunk(self, video_chunk: VideoClip):
        audio_chunk = video_chunk.audio

        sound = np.abs(audio_chunk.to_soundarray())
        sound = sound.max(axis=1).reshape(-1)
        sound = np.hstack([0, sound, 0])

        is_voice = (sound > self.sound_threshold).astype(int)
        borders = is_voice[1:] - is_voice[:-1]
        begin_sound_indexes = np.arange(len(borders))[borders > 0]
        end_sound_indexes = np.arange(len(borders))[borders < 0]

        interesting_parts = np.vstack([begin_sound_indexes, end_sound_indexes])
        return interesting_parts.transpose((1, 0)) / audio_chunk.fps


class WebRtcVADAlgorithm(PiecemealSoundAlgorithm):
    """
    This algorithm selects speech from video using Voice Activity Detection (VAD)
    algorithm coded by google (link https://github.com/wiseman/py-webrtcvad)
    and returns them as interesting parts
    """
    def __init__(self, aggressiveness: int = 1, chunk_in_seconds: float = 60):
        try:
            import webrtcvad
        except ImportError as import_error:
            err = ImportError("WebRtcVADAlgorithm algorithm requires installed 'webrtcvad' module")
            raise err from import_error

        self.aggressiveness = None
        self.set_aggressiveness(aggressiveness)

        super(WebRtcVADAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def get_interesting_parts_from_chunk(self, video_chunk: VideoClip):
        from webrtcvad import Vad

        sample_rate = 48000
        frame_duration = 10  # ms

        audio = video_chunk.audio.set_fps(sample_rate)
        sound = audio.to_soundarray()[:, 0]
        sound = (abs(sound) * 2 ** 16).astype("int16")

        prev_value = False
        chunk = 2 * int(sample_rate * frame_duration / 1000)
        sound = np.hstack([[0] * chunk, sound, [0] * 2 * chunk])
        begins_of_speech, ends_of_speech = [], []
        for i in range(0, len(sound) - chunk, chunk):
            cur_sound = sound[i: i + chunk]
            value = Vad(self.aggressiveness).is_speech(cur_sound.data, sample_rate)
            # I tried self.vad.is_speech(cur_sound.data, sample_rate),
            # but some how it isn't the same.
            if value and not prev_value:
                begins_of_speech.append(i / sample_rate)
            if not value and prev_value:
                ends_of_speech.append(i / sample_rate)
            prev_value = value

        # print(np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0)))
        return np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0))

    def get_aggressiveness(self):
        """:returns aggressiveness: int 0, 1, 2 or 3"""
        return self.aggressiveness

    def set_aggressiveness(self, aggressiveness: int):
        """sets aggressiveness = value"""
        assert aggressiveness in [0, 1, 2, 3],\
            f"aggressiveness must be 0, 1, 2 or 3. {aggressiveness} were given"
        self.aggressiveness = aggressiveness


class SileroVadAlgorithm(SpeedUpAlgorithm):
    """
    This algorithm selects speech from text using VAD algorithm
    from this (https://github.com/snakers4/silero-vad) project
    and returns them as interesting parts.
    """
    def __init__(self, is_adaptive: bool = False, vad_args: list = None, vad_kwargs: dict = None):
        try:
            import torch
        except ImportError:
            msg = "the {} class requires installed torch module."
            raise ImportError(msg.format(__class__))

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad')
        (self.get_speech_ts,
         self.get_speech_ts_adaptive,
         self.save_audio,
         self.read_audio,
         self.state_generator,
         self.single_audio_stream,
         self.collect_chunks) = self.utils

        self.is_adaptive, self.vad_args, self.vad_kwargs = None, None, None

        self.set_vad_args(vad_args)
        self.set_vad_kwargs(vad_kwargs)
        self.set_is_adaptive(is_adaptive)

    def get_interesting_parts(self, moviepy_video: VideoClip):
        vad_func = self._get_vad_func()
        with NamedTemporaryFile(prefix=TEMPORARY_DIRECTORY_PREFIX, suffix=".wav") as temp_file:
            temporary_file_name = temp_file.name
        moviepy_video.audio.write_audiofile(temporary_file_name)

        # temporary_file_name = wav_path = r"C:\Users\m\AppData\Local\Temp/SVA4_lyox5qfk.wav"
        dict_of_interesting_parts = vad_func(temporary_file_name)
        os.remove(temporary_file_name)
        # Todo I don't by what value we should divide timestamps. 16000 works.
        #  It should be replaced by an expression depending on vad_args, vad_kwargs
        list_of_interesting_parts = [[elem['start'] / 16000, elem['end'] / 16000]
                                     for elem in dict_of_interesting_parts]
        # with open("list_of_interesting_parts.npy", "wb") as f:
        #     np.save(f, list_of_interesting_parts)
        return np.array(list_of_interesting_parts)

    def get_is_adaptive(self):
        return self.is_adaptive

    def set_is_adaptive(self, value: bool):
        assert type(value) == bool, "is_adaptive argument must be bool"
        self.is_adaptive = value

    def get_vad_args(self):
        return self.vad_args

    def set_vad_args(self, vad_args: list = None):
        self.vad_args = vad_args if vad_args is not None else []

    def get_vad_kwargs(self):
        return self.vad_kwargs

    def set_vad_kwargs(self, vad_kwargs: dict = None):
        self.vad_kwargs = vad_kwargs if vad_kwargs is not None else {}

    def _get_vad_func(self):
        """

        :return: is_speech_func: str: "path/to/wav" -> bool
        """
        if self.is_adaptive:
            return lambda wav_path: self.get_speech_ts_adaptive(self.read_audio(wav_path),
                                                                self.model,
                                                                *self.vad_args,
                                                                **self.vad_kwargs)
        return lambda wav_path: self.get_speech_ts(self.read_audio(wav_path),
                                                   self.model,
                                                   *self.vad_args,
                                                   **self.vad_kwargs)
