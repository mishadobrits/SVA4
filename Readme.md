
# Introdution
Lets take a look at this two videos (a minute from each one).<br>

| [![Editing tutorial2](http://img.youtube.com/vi/AmTwR03VM_w/0.jpg)](https://www.youtube.com/embed/AmTwR03VM_w?start=40&amp;end=103 "Editing tutorial") <br> (click to play 1m3s subclip)<br> Popular video (>500k subs channel) | [![Editing tutorial2](http://img.youtube.com/vi/M2pQtX0NS8E/0.jpg)](https://www.youtube.com/embed/M2pQtX0NS8E?start=300&amp;end=363 "Editing tutorial") <br> (click to play 1m0s subclip)<br> Not that popular lecture |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

After watching these subclips, we can see that the left tutorial is more watchable than the right one.
It is because the right lecture contains a lot of parts where the lector is thinking or typing, 
while the left tutorial it is well edited and doesn't contain these moments.
The example shows us the importance of cutting out not-speech parts for creating good-quality tutorial. (and getting)
However, manual editing takes requires effort and takes some time (at least 1-3 duration of editing video), which is not very affordable for everyone.
That's why I code program, that automatically finds not-speech parts and cuts them out.<br>
So the progam splits the video <br>
[![Editing tutorial2](http://img.youtube.com/vi/M2pQtX0NS8E/0.jpg "Original video")](https://www.youtube.com/embed/M2pQtX0NS8E?start=300&amp;end=363 "Editing tutorial")
<br> Into the two videos <br>

| [![Sound only](http://img.youtube.com/vi/9D36FOnVd84/0.jpg)](https://www.youtube.com/embed/9D36FOnVd84 "Sound only")<br>Result of programm: Speech only | [![Sound only](http://img.youtube.com/vi/WE_QvSqljGs/0.jpg)](https://www.youtube.com/embed/WE_QvSqljGs "Sound only") <br> Only not speech |   
|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|

<br> And we can see that the result of program is better than the original video.
Of course the result of this program isn't ideal, like in editing tutorial, but it is much better, for no effort.

# How to use it?
The easiest way is to use it - [google.collab](https://colab.research.google.com/drive/1mM0Tz2pYrzbn1e4r0NzBUIneJVWeZZrA) <br>
But you can use it locally
1. Download the mkvmerge from the official website https://mkvtoolnix.download/.
2. Make mkvmerge seen from the command line.
3. Download or clone this code from Github and install all libraries.<br>

In the folder `tests/process_using_algorithm.py` there is a working example.
You can change some parameters (description below and int the collab example) if default ones results too weak or too strong acceleration. 

*Modules `webrtcvad`, `torch` and `torchaudio` are used only for their algorithms,
so they if you don't use the appropriate algorithm, these modules are unnecessary.<br>

# How to call it from a python script?<br>

Also you can call the magic function from your python script.
The basic example would be
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

Lets break down what the code does.<br>
1. Firstly, we need to specify path to input video (and download it if needed).
   For example, we saved it in the
   `input_video_path = input("write path of input video: ")`
2. The second step is to choose an algorithm that returns a list of interesting parts.
   At this moment, these algorithms are available.
   * Base Algorithms: 
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
   * Complex algoritms (takes as arguments other algorithm and combines their result) 
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
     
3. Thirdly, we should set some params.
   The program uses the `settings.Settings` object to contain them. This class only contains all parameters that the program needs.
   Description of supported parameters here.<br>
     * `loud_speed` - speed of interesting parts of video/audio.
     * `quiet_speed` - speed of borring parts of video/audio.
     * `global_speed` - multiplies loud_speed and quiet_speed.
     * `max_quiet_time` - in every boring video piece, the program skips part starting from `max_quiet_time` seconds.
   
   For example, `settings = Settings(min_quiet_time=0.2, quiet_speed=6)`
4. The last but not least is to choose path for output **mkv** video
   (output video must be mkv, if output file extension isn't mkv `process_one_video_in_computer` adds `".mkv"` to the `output_video_path`).
   For example, let be `output_video_path = input("write path of output mkv video: ")`. <br> 

5. When we finish all of these steps, we should run magic function<br>
`process_one_video_in_computer(input_video_path, speedup_algorithm, settings, output_video_path)`
that does the trick.<br>
6. Also, `process_one_video_in_computer` takes some optional additional kwargs<br>
   
   1. The `ffmpeg_preprocess_audio` argument:
      used in audio extraction before applying speedup-alg and video editing
      in cmd calling <br>
      `"ffmpeg -i {inp_path} {ffmpeg_preprocess_audio} -ar 44100 path/audio.wav"`.<br>
      Main examples:
      
      * `''` - No filter.
         Takes 0 additional time, recommended using if you're sure about your speed up algorithm.
      * <i><b><u>[default]</u></b></i> `'-filter:a dynaudnorm'`.  Applies the dynaudnorm ffmpeg filter (normalizes volume in audio),
           which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
           Noise volume and very quiet speech increases not enough to hear.
           Takes ~minute to complete for 80m 1GB video.
       * `'-filter:a loudnorm'` Applies the loudnorm ffmpeg filter (normalizes volume in audio),
           which helps VolumeThresholdAlgorithm and SileroVadAlgorithm.
           Noise volume and very quiet speech increases enough to hear.
           Takes ~10 minutes to complete for 80m 1GB video.
      * `'-filter:a "volume=1.5"'` Increases volume in 1.5 time.
          Takes ~20 sec to complete for 80m 1GB video.
      * `'-filter:a "volume=10dB"'` Increases volume by 10 dB.
          Takes ~20 sec to complete for 80m 1GB video. <br>
   2. `working_directory_path = None` (`str/None`) is a directory where this function saves all intermediate files.
      If `working_directory_path` is `None`,
      `process_one_video_in_computer` creates a temporary directory for this purpose (and deletes it when it finishes).
      The name of the temporary directory starts with 'SVA4_' for easy identification.
   3. `ffmpeg_caller = `
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
   4. `is_result_cfr = False` if this option is True, `apply_calculated_interesting_to_video`
    and `process_one_video_in_computer` returns CFR video, but they works much longer.
   5. `audiocodec = "flac"` auiocodec of result video
      * `"pcm_s16le"` - the fastest, not compact, not played by default windows player
      * `"flac"` - **default**, not played by default windows player
      * `"mp3"` - very compact, played by default windows player
      * ...

#License
MIT