from audio import PartsOfAudio, WavFile
from speed_up import VolumeThresholdAlgorithm, AlgAnd, SileroVadAlgorithm, \
    SpecifiedParts


path = r"C:\Users\m\Downloads\Sites-Buffers\part.wav"
audio = WavFile(path).subclip(60, 65)

v = VolumeThresholdAlgorithm(0.1)
s = SileroVadAlgorithm()
res1 = AlgAnd(v, s).get_interesting_parts(path)
# print(res1)


intervals = v.get_interesting_parts(path)
res2 = get_interesting_parts_only_in_intervals(s, path, intervals)
print(intervals)
print(res1.tolist())
print(res2)
l = min(len(res2), len(res1))
print(res2- res1)


