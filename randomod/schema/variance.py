"""
@TODO: Put a module wide description here
"""
from __future__ import annotations

import enum
import typing
import random

from datetime import datetime
from datetime import timedelta
from random import Random

import pandas
from pydantic import BaseModel
from pydantic import Field

from randomod.utilities.common import get_datetime_format


class ThresholdAdjustment(str, enum.Enum):
    """
    Instructions for where to sample for new values in relation to the observation's threshold
    """
    NEXT = "next"
    STAY = "stay"
    PREVIOUS = "previous"

    @classmethod
    def random(cls, random_number_generator: typing.Union[Random, int] = None) -> str:
        if random_number_generator is None or isinstance(random_number_generator, int):
            random_number_generator = Random(random_number_generator)

        if not isinstance(random_number_generator, Random):
            random_number_generator = random.Random()

        return random_number_generator.choices(list(cls), weights=[1, 3, 1], k=1)[0]


class VarianceEntry(BaseModel):
    """
    Instructions for where to sample values within a range of time
    """
    start_date: datetime
    end_date: datetime
    threshold_adjustment: ThresholdAdjustment

    def to_rows(self, resolution: timedelta = timedelta(hours=1)) -> typing.List[typing.Dict[str, typing.Union[str, datetime]]]:
        rows: typing.List[typing.Dict[str, typing.Union[str, datetime]]] = []

        if resolution.total_seconds() % 86400 == 0:
            # Truncate the date to the day
            current_time: datetime = self.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif resolution.total_seconds() % 3600 == 0:
            # Truncate the date to the hour
            current_time: datetime = self.start_date.replace(minute=0, second=0, microsecond=0)
        elif resolution.total_seconds() % 60 == 0:
            # Truncate the date to the nearest minute
            current_time: datetime = self.start_date.replace(second=0, microsecond=0)
        else:
            current_time: datetime = self.start_date.replace(microsecond=0)

        while current_time < self.end_date:
            current_time += resolution

            rows.append({
                "valid_date": current_time,
                "adjustment": self.threshold_adjustment.value,
            })

        return rows

    def __str__(self):
        start_date = self.start_date.strftime(get_datetime_format(self.start_date.tzinfo))
        end_date = self.end_date.strftime(get_datetime_format(self.end_date.tzinfo))

        if self.threshold_adjustment.STAY:
            return f"Pick a value within the same threshold as the observation from {start_date} through {end_date}"
        elif self.threshold_adjustment.NEXT:
            return f"Try to pick a value from the threshold above the observation's from " \
                   f"{start_date} through {end_date}"

        return f"Try to pick a value from the threshold below the observation's from {start_date} through {end_date}"

    def __repr__(self):
        start_date = self.start_date.strftime(get_datetime_format(self.start_date.tzinfo))
        end_date = self.end_date.strftime(get_datetime_format(self.end_date.tzinfo))

        return f"({start_date}, {end_date}]: {self.threshold_adjustment.value}"


class Variance(BaseModel):
    """
    A series of values that will direct the model where to sample values in relation to observations within
    periods of time
    """
    entries: typing.List[VarianceEntry]

    def get_entry_for_date(self, date: typing.Union[datetime, pandas.Timestamp]) -> typing.Optional[VarianceEntry]:
        for entry in self.entries:
            if entry.start_date < date <= entry.end_date:
                return entry

        return None

    def to_frame(self, resolution: timedelta = timedelta(hours=1)) -> pandas.DataFrame:
        rows: typing.List[typing.Dict[str, typing.Union[str, datetime]]] = []

        for entry in self.entries:
            rows.extend(entry.to_rows(resolution=resolution))

        return pandas.DataFrame(rows)

    @classmethod
    def create_random_variance(
        cls,
        start_date: datetime,
        end_date: datetime,
        random_number_generator: typing.Union[int, Random] = None
    ) -> Variance:
        if random_number_generator is None or isinstance(random_number_generator, int):
            random_number_generator = Random(random_number_generator)

        if not isinstance(random_number_generator, Random):
            random_number_generator = Random()

        entries: typing.List[VarianceEntry] = list()

        duration = end_date - start_date
        minimum_entry_hours = 3
        maximum_entry_hours = 4 if duration < timedelta(hours=24) else 8

        current_start_date = start_date

        while current_start_date < end_date:
            entry_duration = timedelta(hours=random_number_generator.randint(minimum_entry_hours, maximum_entry_hours))
            entry_duration = min(entry_duration, end_date - current_start_date)
            end_time = min(current_start_date + entry_duration, end_date)
            adjustment = ThresholdAdjustment.random(random_number_generator=random_number_generator)
            entries.append(
                VarianceEntry(
                    start_date=current_start_date,
                    end_date=end_time,
                    threshold_adjustment=adjustment
                )
            )
            current_start_date = end_time

        return cls(entries=entries)

    def __str__(self):
        return ", ".join([str(entry) for entry in self.entries])

    def __repr__(self):
        return ", ".join([repr(entry) for entry in self.entries])