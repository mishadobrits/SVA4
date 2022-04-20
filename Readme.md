# In general
Initially, this project aimed to process video lectures.<br> 
This program offered the opportunity to separate speech from not-speech and speed up not-speech parts.<br>
But now, you can use it to speed up boring parts using your list of interesting parts of a video and even write a new algorithm to divide video into interesting and boring pieces.<br>

# Note
By default, this program doesn't rewrite frames but only changes their timecodes, so it works very fast and generates VFR video.
Program skips (extremely speeds) some parts, because of which video players don't have time to play video, because of which desynchronization occurs for a while.<br>  
Also note, that this program accepts only CFR videos so,
if you want to process VFR video or get the resulting CFR video, you have to convert VFR video to CFR.<br>
For this purpose, you can use, for example, FFmpeg. Just run in cmd following command.<br>
`ffmpeg -i path/to/your/vfr/video -c:a copy path/to/output/cfr/video`<br>
(You don't need to decode audio, because it's in normal format).<br>
Also, if `is_result_cfr = True` is setted, main functions runs this command on their own.

And further, when this program starts it creates temporary diretories, and
if user interrupts the code program don't delete it, because of it interrupted.
So, there is a function `main.delete_all_sva4_temporary_objects` deletes all directories and files
that hadn't be deleted. (It is possible, because all temporary directories created by this project
marked with prefix `"SVA_4"` in the start of its name)

Initially there was only a VolumeThresholdAlgorithm, because of which loud_parts was synonym to interesting_parts
and quiet_parts was synonym to boring_parts. I will rename all pieces of code with old names in future,
until it happened keep that quiet=boring and loud=interesting. Sorry for this.<br>

# Instalation
1. Download the mkvmerge from the official website https://mkvtoolnix.download/.
2. Make mkvmerge seen from the command line.
3. Download or clone this code from Github.<br>
   
Modules `webrtcvad`, `torch` and `torchaudio` are used only for their algorithms,
so they if you don't use the appropriate algorithm, these modules are unnecessary.<br>


Now you are ready to start work with it.

# How to use it?<br>
Lets write code, which cuts uninteresting parts.<br>
If you want to process video using a built-in project algorithm.<br> 
1. Firstly, you have to download your video from the Internet (if it located there).
   For example, you saved it in
   `input_video_path = input("write path of input video: ")`
2. The second step is to choose an algorithm that returns a list of interesting parts.
   At this moment, these algorithms are available.
   * Base Algorithms
     * `speed_up.VolumeThresholdAlgorithm(sound_threshold, min_quiet_time=0.25)` accepts float `sound_threshold` and
       returns all pieces where volume >= sound_threshold as interesting parts.
       If gap between two interesting parts <= `min_quiet_time` seconds, algorithm merges this two pieces.<br>
       For example, `speedup_algorithm = VolumeThresholdAlgorithm(0.03)`
     * `speed_up.WebRtcVADAlgorithm(aggressiveness=1)` accepts `aggressiveness=0, 1, 2 or 3`
       selects speech from video using Voice Activity Detection (VAD) algorithm coded by google
       (link https://github.com/wiseman/py-webrtcvad) and returns them as interesting parts.<br>
       For example, `speedup_algorithm = WebRtcVADAlgorithm(2)`.<br>
       This algorithm requires `webrtcvad` module installed.<br>
     * `speed_up.SileroVadAlgorithm(*vad_args, onnx=True, **vad_kwargs)` selects speech from text using VAD algorithm
       from this (https://github.com/snakers4/silero-vad) project and returns them as interesting parts.<br>
       If `onnx=True` programm will use onnx model otherwise it uses pythorch model. For more info check https://github.com/snakers4/silero-vad.
        For example, `speedup_algorithm = SileroVadAlgorithm(trig_sum=0.25, neg_trig_sum=0.7)`<br>
       `SileroVadAlgorithm` requires installed `torch` and `torchaudio` modules.
   * Complex algoritms     
     * `AlgNot(alg)` accepts `alg` as arguments and swap interesting and boring parts.
        For example, `AlgNot(SileroVadAlgorithm())`
     * `speed_up.AlgAnd(alg1, alg2, alg3, ... algn, fast=True)` accepts algorithms as arguments
       and returns parts which all algorithms select as interesting parts.
       If `fast` is False, the function applies `alg[i]` to the whole video, else the function
       applies alg_i only to the parts, which was returned by `alg[i - 1]`.
     * For example, 
       ```
       speedup_algorithm = AlgAnd(
          VolumeThresholdAlgorithm(0.02, min_quiet_time=0.2),
          WebRtcVADAlgorithm(2),
          SileroVadAlgorithm(is_adaptive=True),
          fast=True,
       )
       ```
     * `speed_up.AlgOr(alg1, alg2, alg3, ... algn, fast=True) = AlgNot(speed_up.AlgAnd(AlgNot(alg1), AlgNot(alg2), AlgNot(alg3),
       ... AlgNot(algn)), fast=True)` accepts algorithms as argument
       and returns all parts which at least one algorithm selects as interesting parts.
       For example, 
       ```
       speedup_algorithm = AlgOr(
          VolumeThresholdAlgorithm(0.5),
          WebRtcVADAlgorithm(1),
          SileroVadAlgorithm(trig_sum=0.35, neg_trig_sum=0.5)),
       )
       ```
     
3. Thirdly, you should set some params.
   The program uses the `settings.Settings`  object to contain them. This class only contains all parameters that the program needs.
   Description of supported parameters here.<br>
     * `loud_speed` - speed of interesting parts of video/audio.
     * `quiet_speed` - speed of borring parts of video/audio.
     * `global_speed` - multiplies loud_speed and quiet_speed.
     * `max_quiet_time` - in every boring video piece, the program skips part starting from `max_quiet_time` seconds.
   
   For example, `settings = Settings(min_quiet_time=0.2, quiet_speed=6)`
4. The last but not least is to choose path for output **mkv** video
   (output video must be mkv, if output file extension isn't mkv `process_one_video_in_computer` adds `".mkv"` to the `output_video_path`).
   For example, let be `output_video_path = input("write path of output mkv video: ")`. <br> 
5. <b> [optional]</b> The `ffmpeg_preprocess_audio` argument:
* Used in audio extraction before applying speedup-alg and video editing
   in cmd calling
          `"ffmpeg -i {inp_path} {ffmpeg_preprocess_audio} -ar 44100 path/audio.wav"`.
* Main examples:
* * `''` - No filter.
             Takes 0 additional time, recommended using if you're sure about your speed up algorithm.
* * <i><b><u>[default]</u></b></i> `'-filter:a dynaudnorm'`.  Applies the dynaudnorm ffmpeg filter (normalizes volume in audio),
             which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
             Noise volume and very quiet speech increases not enough to hear.
             Takes ~minute to complete for 80m 1GB video.
* * `'-filter:a loudnorm'` Applies the loudnorm ffmpeg filter (normalizes volume in audio),
             which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
             Noise volume and very quiet speech increases enough to hear.
             Takes ~10 minutes to complete for 80m 1GB video.
* * `'-filter:a "volume=1.5"'` Increases volume in 1.5 time.
             Takes ~20 sec to complete for 80m 1GB video.
* * `'-filter:a "volume=10dB"'` Increases volume by 10 dB.
             Takes ~20 sec to complete for 80m 1GB video.
6. When you finish all of these steps, run<br>
`process_one_video_in_computer(input_video_path, speedup_algorithm, settings, output_video_path)`
to process your video.<br>
   Also, `process_one_video_in_computer` takes some optional kwargs<br>
   * `working_directory_path = None` (`str/None`) is a directory where this function saves all intermediate files.
      If `working_directory_path` is `None`,
     `process_one_video_in_computer` creates a temporary directory for this purpose (and deletes it when it finishes).
      The name of the temporary directory starts with 'SVA4_' for easy identification.
   * `ffmpeg_caller = `
     ```
     ffmpeg_caller.FFPEG_Caller(
         overwrite_force = None,
         hide_ffmpeg_output = False,
         print_ffmpeg_commands = False
     )
     ```
        * `overwrite_force = None` (`True/False/None`)
             * if the value is `None` and if this program needs to overwrite a file, this function asks for your acceptance.
             * if the value is `True` and if this program needs to overwrite a file, this function overwrites it.
             * if the value is `False` and if this program needs to overwrite a file, this function doesn't overwrite it.
        * `hide_output = False` (`True/False`) 
            * If this value is `True` program hides ffmpeg output
            * If this value is `False` program doesn't hide ffmpeg output
        * `print_command = False` (`True/False`)
            * If this parameter is `True` program prints all ffmpeg commands before executing them .
            * If this parameter is `False` it doesn't. 
   * `is_result_cfr = False` if this option is True, `apply_calculated_interesting_to_video`
     and `process_one_video_in_computer` returns CFR video, but they works much longer.
       
   
In total, the code is (tests/process_using_algorithm.py)<br>
```
"""
Testing process_one_video_in_computer function
"""

from main import process_one_video_in_computer
from settings import Settings
from ffmpeg_caller import FFMPEGCaller
from speed_up import (
    VolumeThresholdAlgorithm,
    WebRtcVADAlgorithm,
    SileroVadAlgorithm,
    AlgOr,
    AlgAnd
)

input_video_path = input("write path of input video: ")

# speedup_algorithm = VolumeThresholdAlgorithm(0.02)  # or
# speedup_algorithm = WebRtcVADAlgorithm(3) or
# SileroVadAlgorithm(is_adaptive=True) or
speedup_algorithm = AlgAnd(
    VolumeThresholdAlgorithm(0.02, min_quiet_time=0.2),
    WebRtcVADAlgorithm(2),
    SileroVadAlgorithm(is_adaptive=True),
)  # or any other option

settings = Settings(quiet_speed=6)

output_video_path = input("write path of output video: ")

process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    is_result_cfr=False,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=True, hide_output=True)
)
```

Using this program to apply interesting parts to the video is the same except for steps 2 and 5. <br>
* 2-th step. Instead of choosing algorithm, you should generate your `interesting_parts_list` in format<br>
`[[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ... [start_of_piecen, end_of_piecen]]`<br>
All values should be positions in video in seconds.<br>
* 5-th step. Use `apply_calculated_interesting_to_video` function instead of `process_one_video_in_computer`<br>
Syntax<br>
```
apply_calculated_interesting_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    is_result_cfr=True,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=None, hide_output=False)
)
```

In total, code is (test/set_interesting_parts.py)<br>
```
"""
Testing apply_calculated_interesting_to_video function
"""
import numpy as np

from main import apply_calculated_interesting_to_video
from settings import Settings
from ffmpeg_caller import FFMPEGCaller


input_video_path = input("write path of input video: ")
interesting_parts = np.array([[10.0, 20.1], [30.5, 40], [50.5, 60], [70.5, 80]])
settings = Settings(min_quiet_time=0.2, quiet_speed=6)
output_video_path = input("write path of output mkv video: ")

apply_calculated_interesting_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    is_result_cfr=True,
    ffmpeg_caller=FFMPEGCaller(overwrite_force=None, hide_output=False)
)
```
