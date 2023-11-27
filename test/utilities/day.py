import typing
import unittest

from datetime import datetime

from numpy import datetime64
from pandas import Timestamp

from randomod.utilities import Day


def all_are_equal(sequence: typing.Sequence) -> bool:
    """
    Check to see if all items in the sequence are equivalent

    :param sequence: The collection of values to compare
    :return: True if all items in the sequence are equal
    """
    for comparative_index in range(len(sequence) - 1):
        value_to_check = sequence[comparative_index]
        for comparison_index, value_to_compare in enumerate(sequence[comparative_index + 1:]):
            if value_to_check != value_to_compare:
                print(
                    f"value at index {comparative_index} is not equal to the value at index "
                    f"{comparison_index + comparative_index}: {value_to_check} != {value_to_compare}"
                )
                return False

    return True


class DayTests(unittest.TestCase):
    def test_creation(self):
        test_day = datetime(year=2023, month=11, day=24)

        # The test day isn't on a leap year and is after February 21st. This has to be shifted to account for that
        # extra day. This is only needed when giving an absolute day number
        absolute_leap_year_day = int(test_day.strftime("%j")) + 1

        day_from_string = Day(f'{test_day.year}-{test_day.month}-{test_day.day}')
        day_from_short_dict = Day({'day': test_day.day, 'month': test_day.month})
        day_from_full_dict = Day({"day": test_day.day, "month": str(test_day.month), "year": str(test_day.year)})

        day_from_day_int_dict = Day({"day": absolute_leap_year_day})
        day_from_month_abbreviation_dict = Day(
            {
                "day": test_day.day,
                "month": test_day.strftime("%b"),
                "year": 2020
            }
        )
        day_from_month_name_dict = Day({"day": test_day.day, "month": test_day.strftime("%B")})

        day_from_int = Day(absolute_leap_year_day)

        day_from_datetime = Day(test_day)
        day_from_timestamp = Day.from_epoch(1700855463.340028)
        day_from_datetime64 = Day(datetime64(test_day))
        day_from_pandas_timestamp = Day(Timestamp(test_day))
        day_from_single_value_sequence = Day([329.4])
        day_from_two_value_sequence = Day([11, 24])
        day_from_three_value_sequence = Day([2023, 11, 24])

        all_days = [
            day_from_string,
            day_from_short_dict,
            day_from_full_dict,
            day_from_day_int_dict,
            day_from_month_abbreviation_dict,
            day_from_month_name_dict,
            day_from_int,
            day_from_datetime,
            day_from_timestamp,
            day_from_datetime64,
            day_from_pandas_timestamp,
            day_from_single_value_sequence,
            day_from_two_value_sequence,
            day_from_three_value_sequence
        ]

        self.assertTrue(all_are_equal(all_days))

        self.assertTrue(all([day.day_number == 329 for day in all_days]))


if __name__ == '__main__':
    unittest.main()
