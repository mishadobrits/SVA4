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
from some_functions import str2error_message


class SpeedUpAlgorithm:
    """
    Base class for all Algorithms
    """
    def get_interesting_parts(self, moviepy_video):
        msg = f"All classes inherited from {__class__} must overload get_loud_parts method"
        raise AttributeError(str2error_message(msg))


class VolumeAlgorithm(SpeedUpAlgorithm):
    """
    Returns pieces where volume >= sound_threshold as interesting parts
    """
    def __init__(self, sound_threshold, chunk=5 * 60):
        self.chunk = chunk
        self.sound_threshold = sound_threshold

    def get_interesting_parts(self, moviepy_video):
        audio = moviepy_video.audio
        audio_chunks = [
            audio.subclip(i * self.chunk, min((i + 1) * self.chunk, audio.duration))
            for i in range(ceil(audio.duration / self.chunk))
        ]
        loud_parts = []
        print(f"from {len(audio_chunks)}:", end=" ")
        for i, audio_chunk in enumerate(audio_chunks):
            print(i, end=", ")
            prefix_duration = i * self.chunk
            loud_parts.append(prefix_duration + self.is_voice_in_chunk(audio_chunk))
        print()
        # print(loud_parts)
        return np.vstack(loud_parts)

    def is_voice_in_chunk(self, audio_chunk):
        sound = np.abs(audio_chunk.to_soundarray())
        sound = sound.max(axis=1).reshape(-1)
        sound = np.hstack([0, sound, 0])
        # sound -> [0, sound, 2, 0]
        is_voice = (sound > self.sound_threshold).astype(int)
        # print(is_voice)
        borders = is_voice[1:] - is_voice[:-1]
        begin_sound_indexes = np.arange(len(borders))[borders > 0]
        end_sound_indexes = np.arange(len(borders))[borders < 0]

        return (
            np.vstack([begin_sound_indexes, end_sound_indexes]).transpose((1, 0))
            / audio_chunk.fps
        )
