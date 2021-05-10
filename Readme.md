# In general
Initially, this project aimed to process video lectures.<br> 
This program offered the opportunity to separate speech from not-speech and speed up not-speech parts.<br>
But now, you can use it to speed up boring parts using your list of interesting parts of a video and even write a new algorithm to divide video into interesting and boring pieces.<br>
<br>
This program doesn't rewrite
# How to use it?<br>
If you want to process video using a built-in project algorithm.<br> 
1. Firstly, you have to download your video from the Internet.
   For example, you saved it in
   `input_video_path = input("write input path: ")`
1. The second step is to choose an algorithm that returns a list of interesting parts.
   At this moment, there is only `VolumeAlgorithm(sound_threshold)` that returns all pieces where volume >= sound_threshold.
   Currently, I'm working on embedding a voice activity detection algorithm coded by google into this code.
   For example, `speedup_algorithm = VolumeAlgorithm(0.1)`
1. Thirdly, you should set some params.
   The program uses the `Settings`  object to do this. This class only contains all parameters that the program needs. Description of supported parameters here<br>
     1. `loud_speed` - speed of loud parts of video/audio.
     1. `quiet_speed` - speed of quiet parts of video/audio.
     1. `global_speed` - multiplies loud_speed and quiet_speed.
     1. `min_quiet_time` - the program doesn't accelerate the first `min_quiet_time` seconds in every boring piece.
     1. `max_quiet_time` - in every boring video piece, the program skips part starting from `max_quiet_time`  seconds.
     1. `sound_threshold` - a threshold between loud sound and quiet sound.<br>
   
   For example `settings = Settings(sound_threshold=0.1, min_quiet_time=0.2, quiet_speed=6)`
1. The last but not least is to choose path for output **mkv** video
   (output video must be mkv, if output file extension isn't mkv `process_one_video_in_computer` adds `".mkv"` to the `output_video_path`).
   For example, let be `output_video_path = input("write output path: ")`. 

1. When you finish all of these steps, run<br>
`process_one_video_in_computer(input_video_path, speedup_algorithm, settings, output_video_path)`
to process your video. Also, `process_one_video_in_computer` takes some kwargs<br>
    * `overwrite_outputput_force = None` (`True/False/None`)
        * if the value is `None` and if this program needs to overwrite a file, this function asks for your acceptance.
        * if the value is `True` and if this program needs to overwrite a file, this function overwrites it.
        * if the value is `False` and if this program needs to overwrite a file, this function doesn't overwrite it.
   * `working_directory_path = None` (`str/None`) is a directory where this function saves all intermediate files.
     `working_directory_path` of should be `str` or `None`. If it's `None`,
     `process_one_video_in_computer` creates a temporary directory for this purpose (and deletes it when it finishes).
      The name of the temporary directory starts with 'SVA4_' for easy identification.
   * `hide_ffmpeg_output = False` (`True/False`) 
       * If this value is `True` program hides ffmpeg output
       * If this value is `False` program doesn't hide ffmpeg output
   * `print_ffmpeg_commands = False` (`True/False`)
       * If this parameter is `True` program prints all ffmpeg commands before executing them .
       * If this parameter is `False` it doesn't. 
       
   
In total, the code is (tests/process_using_algorithm.py)<br>
```
from main import process_one_video_in_computer
from settings import Settings
from speed_up import VolumeAlgorithm

input_video_path = input("write input path: ")
speedup_algorithm = VolumeAlgorithm(0.024)
settings = Settings(min_quiet_time=1, quiet_speed=6)
output_video_path = input("write output path: ")
process_one_video_in_computer(
    input_video_path,
    speedup_algorithm,
    settings,
    output_video_path,
    overwrite_output_force=True,
    hide_ffmpeg_output=True,
)
```

Using the program to apply interesting parts to the video is the same except for steps 2 and 5. <br>
Instead of choosing algorithm, you should generate your `interesting_parts_list` in format<br>
`[[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ... [start_of_piecen, end_of_piecen]]`<br>
All values should be positions in video in seconds.<br>

In total, code is (test/set_interesting_parts.py)<br>
```
import numpy as np

from main import apply_calculated_interesting_and_boring_parts_to_video
from settings import Settings

input_video_path = input("write input path: ")
interesting_parts = np.array([[10.0, 20.1], [30.5, 40], [50.5, 60], [70.5, 80]])
settings = Settings(min_quiet_time=0.2, quiet_speed=6)
output_video_path = input("write output path: ")
apply_calculated_interesting_and_boring_parts_to_video(
    interesting_parts,
    settings,
    input_video_path,
    output_video_path,
    overwrite_output_force=True,
    hide_ffmpeg_output=True,
)
```
