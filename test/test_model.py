import functools
import typing
import unittest
import random

from datetime import datetime
from datetime import timedelta

import pandas

from randomod.schema.variance import Variance
from randomod import model

import test.utility

START_DATE = datetime(year=2023, month=10, day=8, hour=12)
END_DATE = datetime(year=2023, month=11, day=8, hour=12)
RESOLUTION = timedelta(hours=1)
THRESHOLD_COUNT = 5
THRESHOLD_VALUE_ADJUSTER = 8
MINIMUM_VALUE = 15
MAXIMUM_VALUE = 150
EPSILON_DIGITS = 4
LOCATIONS = [
    f"location_{index}"
    for index in range(1, 11)
]

test.utility.apply_seed()


def next_generated_value(random_number_generator: random.Random, previous_value: float = None) -> float:
    if previous_value is None:
        previous_value = (MINIMUM_VALUE + MAXIMUM_VALUE) / 2.0

    half_below = previous_value / 2.0
    half_above = previous_value + half_below

    if half_below < MINIMUM_VALUE:
        difference = half_above - half_below
        half_below = MINIMUM_VALUE
        half_above = MINIMUM_VALUE + difference
    elif half_above > MAXIMUM_VALUE:
        difference = half_above - half_below
        half_above = MAXIMUM_VALUE
        half_below = MAXIMUM_VALUE - difference

    return random_number_generator.uniform(
        min(MINIMUM_VALUE, half_below),
        max(MAXIMUM_VALUE, half_above)
    )


def make_test_dataframe(
    location: str,
    value_column: str,
    location_column: str,
    time_column: str,
    random_number_generator: random.Random
) -> pandas.DataFrame:
    current_date = START_DATE

    frame_input = {
        value_column: [],
        time_column: [],
        location_column: []
    }

    expected_rows = ((END_DATE - START_DATE) / RESOLUTION) + 1

    def add_input(value: float, time: datetime):
        frame_input[value_column].append(value)
        frame_input[time_column].append(time)
        frame_input[location_column].append(location)

    current_row_count = 0
    previous_value = next_generated_value(random_number_generator=random_number_generator)
    while current_date <= END_DATE:
        random_value = next_generated_value(
            random_number_generator=random_number_generator,
            previous_value=previous_value
        )
        previous_value = random_value
        add_input(random_value, current_date)
        current_date = current_date + RESOLUTION
        current_row_count += 1

    observation_frame = pandas.DataFrame(frame_input)

    assert observation_frame.shape[0] == expected_rows

    minimum_value = observation_frame[value_column].min()
    maximum_value = observation_frame[value_column].max()
    row_count = observation_frame.shape[0]

    threshold_values = [
        minimum_value + (
                (minimum_value + maximum_value - THRESHOLD_VALUE_ADJUSTER) / THRESHOLD_COUNT
        ) * threshold_index
        for threshold_index in range(1, THRESHOLD_COUNT + 1)
    ]

    thresholds = [
        {
            f"threshold_{index + 1}": threshold_values[index]
            for index in range(len(threshold_values))
        }
    ] * row_count
    threshold_frame = pandas.DataFrame(thresholds)
    single_location_calculation_input = pandas.concat([observation_frame, threshold_frame], axis=1)
    return single_location_calculation_input


def make_multilocation_test_dataframe(
    value_column: str,
    location_column: str,
    time_column: str,
    random_number_generator: random.Random
) -> pandas.DataFrame:
    individual_locations: typing.Iterable[pandas.DataFrame] = [
        make_test_dataframe(
            location=location,
            value_column=value_column,
            location_column=location_column,
            time_column=time_column,
            random_number_generator=random_number_generator
        )
        for location in LOCATIONS
    ]
    return functools.reduce(
        lambda first_frame, second_frame: pandas.concat([first_frame, second_frame]),
        individual_locations
    )


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.value_column = "value"
        self.time_column = "valid_date"
        self.location_column = "location"

        random_number_generator = random.Random(test.utility.RANDOM_VALUE_SEED)

        self.variance = Variance.create_random_variance(
            start_date=START_DATE,
            end_date=END_DATE,
            random_number_generator=random_number_generator
        )
        self.threshold_names = [f"threshold_{index}" for index in range(1, THRESHOLD_COUNT + 1)]
        self.single_location_calculation_input = make_test_dataframe(
            location=LOCATIONS[0],
            value_column=self.value_column,
            location_column=self.location_column,
            time_column=self.time_column,
            random_number_generator=random_number_generator
        )
        self.multilocation_calculation_input = make_multilocation_test_dataframe(
            value_column=self.value_column,
            location_column=self.location_column,
            time_column=self.time_column,
            random_number_generator=random_number_generator
        )

    def test_calculate_values(self):
        test.utility.apply_seed()
        generated_data = model.calculate_values(
            group=self.single_location_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.variance,
            location=LOCATIONS[0],
            location_column=self.location_column,
            time_column=self.time_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        simulation = pandas.DataFrame(generated_data)

        self.assertEqual(self.single_location_calculation_input.shape[0], simulation.shape[0] + 1)

        values = simulation[self.value_column]
        minimum = values.min()
        maximum = values.max()
        summed_values = values.sum()
        mean_error = (values - self.single_location_calculation_input[self.value_column]).mean()
        median = values.median()
        mean = values.mean()
        std = values.std()
        self.assertAlmostEqual(mean_error, 2.1923470110156065, EPSILON_DIGITS)
        self.assertAlmostEqual(minimum, 18.032763298862683, EPSILON_DIGITS)
        self.assertAlmostEqual(maximum, 164.53403399640078, EPSILON_DIGITS)
        self.assertAlmostEqual(median, 82.10641271987954, EPSILON_DIGITS)
        self.assertAlmostEqual(mean, 84.13175452957545, EPSILON_DIGITS)
        self.assertAlmostEqual(summed_values, 62594.025370004136, EPSILON_DIGITS)
        self.assertAlmostEqual(std, 40.12870150027859, EPSILON_DIGITS)

    def test_run_model(self):
        single_location_simulation = model.run_model(
            control_and_thresholds=self.single_location_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.variance,
            time_column=self.time_column,
            location_column=self.location_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        self.assertEqual(self.single_location_calculation_input.shape[0], single_location_simulation.shape[0] + 1)

        multiple_location_simulation = model.run_model(
            self.multilocation_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.variance,
            time_column=self.time_column,
            location_column=self.location_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        self.assertEqual(
            self.multilocation_calculation_input.shape[0],
            multiple_location_simulation.shape[0] + len(LOCATIONS)
        )


if __name__ == '__main__':
    unittest.main()
