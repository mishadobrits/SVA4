"""
This module provides <class Settings> only contains parameters that video processing requires.
Their description is
    loud_speed - speed of loud parts of video/audio.
    quiet_speed - speed of quiet parts of video/audio.
    global_speed - multiplies loud_speed and quiet_speed.
    min_quiet_time - the program doesn't accelerate the first min_quiet_time seconds in every boring piece.
    max_quiet_time - in every boring video piece, the program skips part starting from max_quiet_time seconds.
    sound_threshold - a threshold between loud sound and quiet sound.
Example of usage of Settings class
    settings = Settings(min_quiet_time=1, quiet_speed=1)


And it contains apply_settings_to_interestingpartsarray method that takes interesting_parts_array
 and returns modified according to all parameters interesting_parts_array and boring_parts_array
 (Example of usage of this method:
      new_interesting_parts_array, new_boring_parts_array = setting.apply_settings_to_interestingpartsarray(old_interesting_parts_array)
 )
"""
# setting.py
import numbers
import numpy as np
from some_functions import str2error_message


SETTING_EXTENTION = "SVA_settings"
TRIVIAL_DICT = {
    "global_speed": 1,
    "loud_speed": 1,
    "quiet_speed": 6,
    "min_quiet_time": 0.25,
    "max_quiet_time": 2,
    "volume_coefficient": 1,
    "quiet_volume_coefficient": 0.35,
    "loud_volume_coefficient": 1,
    "max_volume": 1,
    "decrease": 1,
    "brightness": 1,
    "contras_ratio": 1,
    "rotate_image": 0,
    "inverted": False,
}


class Settings:
    """
    Settings class provide ability to store settings data, to change it, to read and save from/to file,
    convert that data to dict and from dict
    argumets
        you can specify either filepath either kwargs otherwise (if you specify both of them) excecption will be raised
    ----filepath - Settings will be read from file
    ----kwargs in format global_speed=value
                         loud_speed=value
                              ...
                         inverted=value
            some of them can be skipped

    fields: self.
      (float)
        loud_speed - speed of loud parts of video/audio.
        quiet_speed - speed of quiet parts of video/audio.
        global_speed - multiplies loud_speed and quiet_speed.
        min_quiet_time - the program doesn't accelerate the first min_quiet_time seconds in every boring piece.
        max_quiet_time - in every boring video piece, the program skips part starting from max_quiet_time seconds.
        sound_threshold - a threshold between loud sound and quiet sound.

        # todo All parameter starting from here are not supported by video processing

        quiet_volume_coefficient - multiply sound in quiet video parts   # not supported yet
        loud_volume_coefficient - multipld by sound in loud video parts  # not supported yet
        volume_coefficient - multiply quiet_volume_coefficient and loud_volume_coefficient  # not supported yet
        max_volume - maximal able volume of sound: all frames above this value will decreased to this volume  # not supported yet

        decrease - number to !decrease! image  # not supported yet
        brightness - number to make more bright # not supported yet
        contras_ratio - number to make more bright # not supported yet
        rotate_image (only 0, 1, 2, 3) - how many 90-turnes of video # not supported yet
        inverted (bool) - is image reversed relative vertical axis  # not supported yet

    methods:
        self.to_dict() - convert self to dict
        self.save_to_file() - save self to file
        self.set_X(value), self.get_X - set field X value and get field X
                for all fields X

    """

    def __init__(self, filepath="", **kwargs):
        """
        If filepath is True, Settings will be read from file.
        You can specify only one among (filepath, kwargs).
        Filepath must be end by SETTING_EXTENTION.
        Format of **kwargs is the same like format of self.to_dict()
        """
        if filepath and kwargs:
            err_msg = "Settings.__init__() takes filepath or kwargs\nfilepath = {}\nkwargs = {}\nwere given"
            raise ValueError(err_msg.format(str(filepath), str(kwargs)))

        filepath = filepath.strip()
        if filepath and not filepath.endswith("." + SETTING_EXTENTION):
            error = "'filepath' must ends with '{}', '{}' were given"
            raise ValueError(error.format("." + SETTING_EXTENTION, filepath))

        if filepath:
            with open(filepath, "r") as file:
                kwargs = eval("".join(file.readlines()))

        for elem in TRIVIAL_DICT:
            self.__dict__[elem] = kwargs.get(elem, TRIVIAL_DICT[elem])

        for elem in kwargs:
            if elem not in TRIVIAL_DICT:
                err_msg = "'{}' is an invalid keyword argument for Settings()\nList of valid options {}"
                raise TypeError(err_msg.format(elem, list(TRIVIAL_DICT.keys())))

    def check_type_decorators_generator(excpected_type=numbers.Number):
        def check_type_decorator(func):
            def wrapper(self, arg1, *args, **kwargs):
                if not isinstance(arg1, excpected_type):
                    msg = f"""function '{Settings.__name__}.{func.__name__}'
                         expected '{excpected_type.__name__}' type
                         (or inherited classes).
                         Type({arg1}) = '{type(arg1).__name__}' were given."""
                    raise TypeError(str2error_message(msg))
                return func(self, arg1, *args, **kwargs)

            return wrapper

        return check_type_decorator

    @check_type_decorators_generator()
    def set_global_speed(self, value):
        self.global_speed = abs(value)

    def get_global_speed(self):
        return self.global_speed

    @check_type_decorators_generator()
    def set_loud_speed(self, value):
        self.loud_speed = abs(value)

    def get_loud_speed(self):
        return self.loud_speed

    def get_real_loud_speed(self):
        return self.loud_speed * self.global_speed

    @check_type_decorators_generator()
    def set_quiet_speed(self, value):
        self.quiet_speed = abs(value)

    def get_quiet_speed(self):
        return self.quiet_speed

    def get_real_quiet_speed(self):
        return self.quiet_speed * self.global_speed

    @check_type_decorators_generator()
    def set_min_quiet_time(self, value):
        self.min_quiet_time = abs(value)

    def get_min_quiet_time(self):
        return self.min_quiet_time

    @check_type_decorators_generator()
    def set_max_quiet_time(self, value):
        self.max_quiet_time = abs(value)

    def get_max_quiet_time(self):
        return self.max_quiet_time

    @check_type_decorators_generator()
    def set_volume_coefficient(self, value):
        self.volume_coefficient = abs(value)

    def get_volume_coefficient(self):
        return self.volume_coefficient

    @check_type_decorators_generator()
    def set_quiet_volume_coefficient(self, value):
        self.quiet_volume_coefficient = abs(value)

    def get_quiet_volume_coefficient(self):
        return self.quiet_volume_coefficient

    @check_type_decorators_generator()
    def set_loud_volume_coefficient(self, value):
        self.loud_volume_coefficient = abs(value)

    def get_loud_volume_coefficient(self):
        return self.loud_volume_coefficient

    @check_type_decorators_generator()
    def set_max_volume(self, value):
        self.max_volume = abs(value)

    def get_max_volume(self):
        return self.max_volume

    @check_type_decorators_generator()
    def set_decrease(self, value):
        self.decrease = abs(value)

    def get_decrease(self):
        return self.decrease

    def get_brightness(self):
        return self.brightness

    @check_type_decorators_generator()
    def set_brightness(self, value):
        self.brightness = abs(value)

    def get_contras_ratio(self):
        return self.contras_ratio

    @check_type_decorators_generator()
    def set_contras_ratio(self, value):
        if value <= 0:
            return
        self.contras_ratio = value

    @check_type_decorators_generator(int)
    def set_rotate_image(self, value):
        self.rotate_image = value % 4

    def get_rotate_image(self):
        return self.rotate_image

    def get_sound_threshold(self):
        return self.sound_threshold

    @check_type_decorators_generator(float)
    def set_sound_threshold(self, value):
        self.sound_threshold = value

    @check_type_decorators_generator(bool)
    def set_inverted(self, value):
        self.inverted = value

    def get_inverted(self):
        return self.inverted

    def full_dict(self):
        """convert self to dict
        for inverse operation use smth = Settings(**dictionary)"""
        rt = {}
        for elem in TRIVIAL_DICT:
            if self.__dict__[elem] != TRIVIAL_DICT[elem]:
                rt[elem] = self.__dict__[elem]
        return rt

    def __getitem__(self, key):
        return self.full_dict()[key]

    def to_dict(self):
        full_dict, part_dict = self.full_dict(), {}
        for elem in full_dict:
            if full_dict[elem] != TRIVIAL_DICT[elem]:
                part_dict[elem] = full_dict[elem]
        return part_dict

    def __str__(self):
        self_dict = self.to_dict()
        temp = ", ".join(
            ["{} = {}".format(elem, self_dict[elem]) for elem in self_dict]
        )
        return "Settings({})".format(temp)

    def save_to_file(self, filepath):
        """Save self to filepath"""
        if not filepath.endswith("." + SETTING_EXTENTION):
            filepath += "." + SETTING_EXTENTION
        with open(filepath, "w") as file:
            print("{}: {}".format(filepath, self.to_dict()))
            file.write(str(self.to_dict()))

    def is_trivial(self):
        return not bool(self.to_dict())

    def apply_settings_to_interestingpartsarray(self, interesting_parts):
        """
        Changes lenght

        :param interesting_parts: interesting parts at usual format
         [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], [start_of_piece2, end_of_piece2]]
        :return: new calculated interesting_parts_np_array and boring_parts_np_array in the same format
        """
        min_q, max_q = self.get_min_quiet_time(),
        begin_sound_indexes, end_sound_indexes = (
            interesting_parts[:, 0],
            interesting_parts[:, 1],
        )

        end_sound_indexes[:-1] += min_q

        is_changing = begin_sound_indexes[1:] > end_sound_indexes[:-1]
        begin_sound_indexes = begin_sound_indexes[np.hstack([True, is_changing])]
        end_sound_indexes = end_sound_indexes[np.hstack([is_changing, True])]

        interesting_parts = np.vstack(
            [begin_sound_indexes, end_sound_indexes]
        ).transpose((1, 0))
        boring_parts_beginings = np.hstack([0, end_sound_indexes[:-1]])
        boring_parts_ends = np.minimum(
            begin_sound_indexes,
            boring_parts_beginings + max_q - min_q,
        )

        boring_parts_ends[0] = np.minimum(
            begin_sound_indexes[0],
            boring_parts_beginings[0] + self.get_max_quiet_time(),
        )
        boring_parts = np.vstack([boring_parts_beginings, boring_parts_ends]).transpose(
            (1, 0)
        )

        return interesting_parts, boring_parts
