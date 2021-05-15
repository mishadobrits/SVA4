"""
This module contains algorithms that get moviepy_video_object and return a list of interesting parts
in format [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ...
[start_of_piecen, end_of_piecen]]
All values should be positions in video in seconds.
All algorithms must be inherited from the base class 'SpeedUpAlgorithm.'
Currently, there is only 'VolumeAlgorithm(sound_threshold)'
that returns pieces of the video where 'volume >= sound_threshold' as interesting parts.
"""

from math import ceil
import numpy as np
from webrtcvad import Vad

from some_functions import str2error_message


class SpeedUpAlgorithm:
    """
    Base class for all Algorithms
    """
    def get_interesting_parts(self, moviepy_video):
        msg = f"All classes inherited from {__class__} must overload get_loud_parts method"
        raise AttributeError(str2error_message(msg))


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
    def __init__(self, chunk_in_seconds=60):
        self.chunk = chunk_in_seconds

    def get_interesting_parts(self, moviepy_video):
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

    def get_interesting_parts_from_chunk(self, video_chunk):
        msg = "All class inherited from {} must overload {} method. Class {} doesn't."
        msg = msg.format(__class__, "get_interesting_parts_from_chunk", type(self))
        raise AttributeError(msg)


class PiecemealSoundAlgorithm(PiecemealBaseAlgorithm):
    def get_interesting_parts(self, moviepy_video):
        class FakeVideo:
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


class VolumeAlgorithm(PiecemealSoundAlgorithm):
    """
    Returns pieces where volume >= sound_threshold as interesting parts
    """
    def __init__(self, sound_threshold, chunk_in_seconds=60):
        self.sound_threshold = sound_threshold
        super(VolumeAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def set_sound_threshold(self, value):
        self.sound_threshold = value

    def get_sound_threshold(self):
        return self.sound_threshold

    def get_interesting_parts_from_chunk(self, video_chunk):
        audio_chunk = video_chunk.audio

        sound = np.abs(audio_chunk.to_soundarray())
        sound = sound.max(axis=1).reshape(-1)
        sound = np.hstack([0, sound, 0])

        is_voice = (sound > self.sound_threshold).astype(int)
        borders = is_voice[1:] - is_voice[:-1]
        begin_sound_indexes = np.arange(len(borders))[borders > 0]
        end_sound_indexes = np.arange(len(borders))[borders < 0]

        rt = np.vstack([begin_sound_indexes, end_sound_indexes])
        return rt.transpose((1, 0)) / audio_chunk.fps


class WebRtcVADAlgorithm(PiecemealSoundAlgorithm):
    def __init__(self, aggressiveness=1, chunk_in_seconds=60):
        self.aggressiveness = aggressiveness
        self.vad = Vad(self.aggressiveness)
        super(WebRtcVADAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def get_interesting_parts_from_chunk(self, video_chunk):
        sample_rate = 48000
        frame_duration = 10  # ms

        audio = video_chunk.audio.set_fps(sample_rate)
        sound = audio.to_soundarray()[:, 0]
        sound = (abs(sound) * 2 ** 16).astype("int16")

        prev_value = False
        chunk = 2 * int(sample_rate * frame_duration / 1000)
        sound = np.hstack([[0] * chunk, sound, [0] * 2 * chunk])
        begins_of_speech, ends_of_speech = [], []
        debug_frame = b"\x00\x00" * 480
        for i in range(0, len(sound) - chunk, chunk):
            cur_sound = sound[i: i + chunk]
            # print("any", cur_sound.any(), len(cur_sound.data), cur_sound.data == debug_frame)
            value = Vad(self.aggressiveness).is_speech(cur_sound.data, sample_rate)
            # print(self.vad.is_speech(b"\x00\x00" * 480, sample_rate), Vad(1).is_speech(b"\x00\x00" * 480, sample_rate))
            if value and not prev_value:
                begins_of_speech.append(i / sample_rate)
            if not value and prev_value:
                ends_of_speech.append(i / sample_rate)
            prev_value = value

        # print(np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0)))
        return np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0))
