"""
This file contains FFMPEGCaller class to call ffmpeg with some options.
Options description is in FFMPEGCaller.__doc__
"""
import os


class FFMPEGCaller:
    """
    Usage
        ffmpeg = FFMPEGCaller()
        # some code ...
        ffmpeg(string_command)
    is equivalent to
       # some code
       os.command("ffmpeg " + string_command)

    Also there is some options you can set:
        :print_command = False: bool
             if this parameter is 'True' ffmpeg(command) prints command before calling.
        :hide_output = False: bool
             if this parameter is 'True' ffmpeg adds " -hide_banner -loglevel error"
             to the end of command what hides usual ffmpeg output.
        :overwrite_force True, False or None:
            - if this parameter is 'True' ouput_file will be overwrited
            - if this parameter is 'False' ouput_file will be overwrited
            - if this parameter is 'None' command line asks you
                'File '{output_file}' already exists. Overwrite?'
            This feature works good only if there is only 1 output file.  #todo
            If there is several output file this feture works for the last one

    """

    def __init__(self, overwrite_force=None, print_command=False, hide_output=False):
        self.print_command = print_command
        self.hide_output = hide_output
        self.overwrite_force = overwrite_force

    def set_print_command(self, value):
        """sets print_command field"""
        self.print_command = value

    def get_print_command(self):
        """returns print_command field"""
        return self.print_command

    def set_hide_output(self, value):
        """sets hide_output field"""
        self.hide_output = value

    def get_hide_output(self):
        """returns hide_output field"""
        return self.hide_output

    def set_overwrite_force(self, value):
        """sets overwrite_force field"""
        self.overwrite_force = value

    def get_overwrite_force(self):
        """returns overwrite_force field"""
        return self.overwrite_force

    def __call__(self, command):
        if os.path.exists(list(command.split())[-1]):
            if self.overwrite_force:
                command = "-y " + command
            elif type(self.overwrite_force) == bool:
                command = "-n " + command
            else:  # self.overwrite_force -is None
                pass

        command = "ffmpeg " + command
        if self.hide_output:
            command += " -hide_banner -loglevel error"
        if self.print_command:
            print(command)
        return os.system(command)