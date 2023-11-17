"""
@TODO: Put a module wide description here
"""
from __future__ import annotations

import typing
from datetime import datetime

import numpy
import pandas

from dateutil.parser import parse as parse_date

from randomod.utilities import is_sequence_type
from randomod.utilities import value_is_number


class Day:
    """
    A simple wrapper around an integer value between 1 and 366 to represent a consistent number of a day of a year

    These takes leap year into account, where 2021/5/23 will have the same value as 2020/5/23
    """
    __slots__ = ['__day']

    LEAP_DAY_OR_FIRST_OF_MARCH = 60

    def __init__(
            self,
            day: typing.Union[
                str,
                pandas.Timestamp,
                numpy.datetime64,
                datetime,
                int,
                dict,
                typing.Sequence[typing.Union[str, int]]]
    ):
        if day is None:
            raise ValueError("The day is not defined; 'None' has been passed.")

        if is_sequence_type(day) and len(day) == 1:
            day = day[0]

        if is_sequence_type(day):
            possible_args = [
                int(float(argument))
                for argument in day
                if value_is_number(argument)
            ]
            if len(possible_args) == 1:
                day = possible_args[0]
            elif len(possible_args) == 2:
                # We are going to interpret this as month-day
                day = pandas.Timestamp(year=2020, month=possible_args[0], day=possible_args[1])
            elif len(possible_args) > 3:
                # We're going to interpret this as year-month-day. Further args may include time, but those are not
                # important for this
                day = pandas.Timestamp(year=possible_args[0], month=possible_args[1], day=possible_args[2])
            else:
                raise ValueError("A list of no numbers was passed; a Day cannot be interpretted.")

        if isinstance(day, str) and value_is_number(day):
            day = float(day)

        if isinstance(day, float):
            day = int(day)

        if isinstance(day, int) and (day < 1 or day > 366):
            raise ValueError(f"'{day}' cannot be used as a day number - only days between 1 and 366 are allowable.")

        if isinstance(day, str):
            day = parse_date(day)

        if not isinstance(day, pandas.Timestamp) and isinstance(day, (numpy.core.datetime64, datetime)):
            day = pandas.Timestamp(day)

        # Due to the leap day, the day of the year number changes every four years, making the numbers inconsistent.
        # All day of the year numbers will be one behind after the non-existent February 29th, so that number
        # is incremented by 1 to ensure that it matches in and out of leap years.
        if isinstance(day, pandas.Timestamp):
            if not day.is_leap_year and day >= datetime(day.year, month=3, day=1, tzinfo=day.tzinfo):
                day = day.dayofyear + 1
            else:
                day = day.dayofyear

        self.__day = numpy.core.uint16(day)

    @property
    def day_number(self) -> int:
        """
        The number of the day of the year; consistent between leap and non-leap years

        Note: This number will not always point to the true day of the year number. All values post-February 28th
        on non-leap years will be increased by 1 to make the value consistent across leap and non-leap years.

        Returns:
            The number of the day of the year; consistent between leap and non-leap years.
        """
        return self.__day

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        this_year = datetime.now().year
        not_leap_year = datetime.now().year % 4 != 0

        # The day will have been adjusted if this weren't a leap year and was at or after the last day in February,
        # so reverse the adjustment to get the right date
        if not_leap_year and self.__day >= self.LEAP_DAY_OR_FIRST_OF_MARCH:
            day = self.__day - 1
        else:
            day = self.__day

        parsed_date = datetime.strptime(f"{this_year}-{day}", "%Y-%j")
        representation = parsed_date.strftime("%B %-d")
        return representation

    def __eq__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day == other.day_number

    def __ge__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day >= other.day_number

    def __le__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day <= other.day_number

    def __gt__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day > other.day_number

    def __lt__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day < other.day_number

    def __hash__(self):
        return hash(self.__repr__())
