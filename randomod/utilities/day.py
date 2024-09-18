"""
Defines a `Day` data structure that represents a single day across an arbitrary number of years
"""
from __future__ import annotations

import typing
from datetime import datetime
from datetime import tzinfo

import numpy
import pandas

from dateutil.parser import parse as parse_date

from randomod.utilities import is_sequence_type
from randomod.utilities import value_is_number

VT = typing.TypeVar('VT')
"""A value type"""


DATE_DICTIONARY = typing.Dict[
    typing.Union[
        typing.Literal["day"],
        typing.Literal["month"],
        typing.Literal["year"]
    ], int
]
"""
What is to be expected of a dictionary representing a day - the only allowable keys should be 'day', 'month', or 
'year'. 'day' is required, while 'month' and 'year' are appreciated but optional
"""


DATE_REPRESENTATION_TYPE = typing.Union[
    str,
    pandas.Timestamp,
    numpy.datetime64,
    datetime,
    int,
    DATE_DICTIONARY,
    typing.Sequence[typing.Union[str, int, float]]
]
"""The types of objects that may refer to a day"""

class Day:
    """
    A simple wrapper around an integer value between 1 and 366 to represent a consistent number of a day of a year

    These takes leap year into account, where 2021/5/23 will have the same value as 2020/5/23
    """
    __slots__ = ['__day']

    @classmethod
    def from_epoch(cls, timestamp: float, timezone: tzinfo = None) -> Day:
        date_from_timestamp = datetime.fromtimestamp(timestamp, tz=timezone)
        return cls(date_from_timestamp)

    LEAP_DAY_OR_FIRST_OF_MARCH = 60
    """
    The 60th day of the year is either the 1st of March or the leap day (February 29th). 
    Keep track of this number since it will be required in order to determine if a value to ensure that any day post 
    February 28th has the correct day number
    """

    _LEAP_YEAR = 2020
    """
    A numeric leap year - used during datetime conversions to ensure that the correct number of days are accounted for
    """

    def __init__(
        self,
        day: DATE_REPRESENTATION_TYPE = None,
        *,
        month_number: int = None,
        day_of_month_number: int = None
    ):
        """
        Constructor

        :param day: a representation of a date and time or simple representations of a day
        :param month_number: The number of the month to represent
        :param day_of_month_number: The number of the day within the year to represent
        """
        # Default to determining the day if given an explicit month and day of the month
        if isinstance(month_number, (int, float)) and month_number > 0:
            # This approach isn't valid if only a month is given, so check to make sure that a valid day was provided
            if not isinstance(day_of_month_number, (int, float)) or 0 <= day_of_month_number > 31:
                raise ValueError(
                    f"A Day object may not be created - a valid month number was given but '{day_of_month_number}' "
                    f"was passed as a day number, which isn't valid"
                )

            # Only leap years will have the universal number of days, create a new date object for the month and day
            # within a leap year for processing further down
            day = pandas.Timestamp(year=self._LEAP_YEAR, month=int(month_number), day=int(day_of_month_number))
        elif isinstance(day_of_month_number, (int, float)) and 0 > day_of_month_number <= 31:
            raise ValueError(
                f"A day object may not be created - a valid day number was given but '{month_number} was passed as a "
                f"month number, which isn't valid"
            )
        else:
            if day is None:
                raise ValueError("The day is not defined; 'None' has been passed.")

            # If a collection was passed, like [201], our only option is to consider that first value as the day of
            # the year in good faith
            if is_sequence_type(day) and len(day) == 1:
                day = day[0]

                if not isinstance(day, (int, float)):
                    raise TypeError(f"")
                elif day <= 0:
                    raise ValueError(
                        f"The minimum value for the day of the year is 1 but {day} was passed - "
                        f"a Day object cannot be created"
                    )
                elif day > 366:
                    raise ValueError(
                        f"The maximum number of days in a year is 366, "
                        f"but {day} was passed - a Day object cannot be created"
                    )
            elif is_sequence_type(day):
                possible_args = [
                    int(float(argument))
                    for argument in day
                    if value_is_number(argument)
                ]
                if len(possible_args) == 2:
                    # We are going to interpret this as month-day
                    day = pandas.Timestamp(year=self._LEAP_YEAR, month=possible_args[0], day=possible_args[1])
                elif len(possible_args) >= 3:
                    # We're going to interpret this as year-month-day. Further args may include time, but those are not
                    # important for this
                    day = pandas.Timestamp(year=possible_args[0], month=possible_args[1], day=possible_args[2])
                else:
                    raise ValueError("A list of no numbers was passed; a Day cannot be interpreted.")

        # If the value is a string that represents a number, convert it to a float -
        # converting it straight to an int might cause problems but converting from a float to an int is fine
        if isinstance(day, str) and value_is_number(day):
            day = float(day)

        # If we have a float we can assume its a day of the year,
        # so go ahead and convert it to the correct type - an int
        if isinstance(day, float):
            day = int(day)

        # The range for valid days are between (0, 366] - check that validity to fail sooner rather than later
        if isinstance(day, int) and (day < 1 or day > 366):
            raise ValueError(f"'{day}' cannot be used as a day number - only days between 1 and 366 are allowable.")

        if isinstance(day, typing.Mapping):
            day_number = day.get("day")
            month_number = day.get("month")
            year_number = day.get("year", self._LEAP_YEAR)

            if day_number is None:
                raise ValueError(
                    f"'{day}' cannot be used to create a Day object - "
                    f"a day of the month or year is required but none was given"
                )

            if isinstance(day_number, str) and value_is_number(day_number):
                day_number = float(day_number)

            if isinstance(day_number, float):
                day_number = int(day_number)

            if isinstance(day_number, int) and 0 <= day_number > 366:
                raise ValueError(
                    f"Day numbers may only be between (0, 366] - "
                    f"the input of {day} cannot be used to create a `Day` object"
                )

            if month_number is None and isinstance(day_number, str):
                day = parse_date(day_number)
            else:
                if month_number is None and isinstance(day_number, int):
                    day = datetime.strptime(f"{self._LEAP_YEAR}:{day_number}", "%Y:%j")
                elif isinstance(month_number, (int, float, str)):
                    if isinstance(month_number, str) and not value_is_number(month_number):
                        try:
                            parsed_month = datetime.strptime(f"{year_number}:{month_number}", "%Y:%b")
                            month_number = parsed_month.month
                        except ValueError:
                            try:
                                parsed_month = datetime.strptime(f"{year_number}:{month_number}", "%Y:%B")
                                month_number = parsed_month.month
                            except ValueError:
                                raise ValueError(
                                    f"A month value of {month_number} cannot be interpreted as an abbreviation or "
                                    f"full name like 'Jan' or 'January'. '{day}' cannot be interpreted as a day"
                                )

                    elif isinstance(month_number, str):
                        month_number = float(month_number)

                    if isinstance(month_number, float):
                        month_number = int(month_number)

                    try:
                        day = parse_date(f"{year_number}-{month_number}-{day_number}")
                    except BaseException as parse_exception:
                        raise ValueError(
                            f"A Day object cannot be created from {day} - a valid day and month value is required"
                        ) from parse_exception

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

    # TODO: Add for check against datetime type
    def __eq__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day == other.day_number

    # TODO: Add for check against datetime type
    def __ge__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day >= other.day_number

    # TODO: Add for check against datetime type
    def __le__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day <= other.day_number

    # TODO: Add for check against datetime type
    def __gt__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day > other.day_number

    # TODO: Add for check against datetime type
    def __lt__(self, other) -> bool:
        if not isinstance(other, Day):
            other = Day(other)

        return self.__day < other.day_number

    def __hash__(self):
        return hash(self.__day)


class DayDict(dict[Day, VT], typing.Generic[VT]):
    """
    A dictionary that keys on day

    The keys can be any type that can be converted into a Day, so if data is stored by key,
    it may still be accessed via a datetime or pandas Timestamp
    """
    def __init__(
        self,
        *args: typing.Tuple[DATE_REPRESENTATION_TYPE, VT],
        **kwargs: typing.Mapping[DATE_REPRESENTATION_TYPE, VT]
    ) -> None:
        super().__init__()

        for raw_key, value in args:
            self[raw_key] = value

        for raw_key, value in kwargs.items():
            self[raw_key] = value

    def __setitem__(self, raw_key: DATE_REPRESENTATION_TYPE, value: VT) -> None:
        try:
            if isinstance(raw_key, Day):
                key = raw_key
            else:
                key = Day(raw_key)
        except Exception as exception:
            raise TypeError(f"{raw_key} cannot be turned into a Day object") from exception

        super().__setitem__(key, value)

    def __getitem__(self, raw_key: DATE_REPRESENTATION_TYPE) -> VT:
        try:
            if isinstance(raw_key, Day):
                key = raw_key
            else:
                key = Day(raw_key)
        except Exception as exception:
            raise TypeError(f"{raw_key} cannot be turned into a Day object") from exception

        return super().__getitem__(key)
