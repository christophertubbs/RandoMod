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
from randomod.utilities import get_unique_sequence_values

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
    for index in range(1, 201)
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


def create_threshold_value_table(row_count: int, threshold_values: typing.List[float], random_number_generator: random.Random) -> pandas.DataFrame:
    threshold_value_input = {}
    for threshold_index, center_value in enumerate(threshold_values):
        threshold_number = str(threshold_index + 1)
        threshold_name = "threshold_" + threshold_number
        adjustment_amount = 0.25
        adjustment = center_value * adjustment_amount
        lower_lower_bound = center_value - adjustment
        upper_lower_bound = center_value + adjustment
        lower_bounds = [
            random_number_generator.uniform(
                lower_lower_bound,
                upper_lower_bound
            )
            for _ in range(row_count)
        ]
        upper_bounds = [
            random_number_generator.uniform(lower_bound, lower_bound + adjustment)
            for lower_bound in lower_bounds
        ]

        threshold_value_input[threshold_name] = [
            random_number_generator.uniform(lower_bound, upper_bound)
            for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)
        ]
    return pandas.DataFrame(threshold_value_input)


def make_thresholds_for_location(
    observation_data: pandas.DataFrame,
    location: str,
    location_column: str,
    value_column: str,
    random_number_generator: random.Random,
    by_time: bool = None,
    time_column: str = None
) -> pandas.DataFrame:
    values = observation_data[value_column]

    minimum_value = values.min()
    maximum_value = values.max()

    threshold_values = [
        minimum_value + (
                (minimum_value + maximum_value - THRESHOLD_VALUE_ADJUSTER) / THRESHOLD_COUNT
        ) * threshold_index
        for threshold_index in range(1, THRESHOLD_COUNT + 1)
    ]

    if isinstance(observation_data.index, pandas.MultiIndex):
        threshold_frame_contents = get_unique_sequence_values([
            {
                name: index_value
                for name, index_value in zip(observation_data.index.names, index_combination)
            }
            for index_combination in observation_data.index
        ])
    else:
        threshold_frame_contents = get_unique_sequence_values([
            {
                observation_data.index.names[0] or "index": value
            }
            for value in observation_data.index
        ])

    for row in threshold_frame_contents:
        if location_column not in row:
            row[location_column] = location

    if by_time and time_column in observation_data:
        for row, time_value in zip(threshold_frame_contents, observation_data[time_column]):
            if time_column not in row:
                row[time_column] = time_value

    threshold_frame = pandas.DataFrame(threshold_frame_contents)
    raw_threshold_values = create_threshold_value_table(threshold_frame.shape[0], threshold_values, random_number_generator)
    threshold_frame = threshold_frame.join(raw_threshold_values)
    threshold_frame = threshold_frame.set_index(
        keys=[location_column, time_column] if by_time else [location_column],
        drop=True
    )

    return threshold_frame


def make_test_dataframe(
    location: str,
    value_column: str,
    location_column: str,
    time_column: str,
    random_number_generator: random.Random,
    by_time: bool = None
) -> pandas.DataFrame:
    current_date = START_DATE
    if by_time:
        pair_keys = [location_column, time_column]
    else:
        pair_keys = [location_column]

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

    observation_frame = pandas.DataFrame(frame_input).set_index(keys=pair_keys, drop=True)

    assert observation_frame.shape[0] == expected_rows

    return observation_frame


def make_single_test_dataframe(
    location: str,
    value_column: str,
    location_column: str,
    time_column: str,
    random_number_generator: typing.Union[random.Random, int] = None,
    by_time: bool = None
):
    if random_number_generator is None:
        random_number_generator = random.Random(test.utility.RANDOM_VALUE_SEED)
    elif isinstance(random_number_generator, int):
        random_number_generator = random.Random(random_number_generator)

    observation_frame: pandas.DataFrame = make_test_dataframe(
        location=location,
        value_column=value_column,
        location_column=location_column,
        time_column=time_column,
        random_number_generator=random_number_generator,
        by_time=by_time
    )

    threshold: pandas.DataFrame = make_thresholds_for_location(
        observation_data=observation_frame,
        location=location,
        location_column=location_column,
        value_column=value_column,
        random_number_generator=random_number_generator,
        by_time=by_time,
        time_column=time_column
    )

    single_location_calculation_input = observation_frame.join(threshold, how='inner')
    return single_location_calculation_input


def make_multiple_location_test_dataframe(
    value_column: str,
    location_column: str,
    time_column: str,
    random_number_generator: typing.Union[random.Random, int] = None,
    by_time: bool = None
) -> pandas.DataFrame:
    if random_number_generator is None:
        random_number_generator = random.Random(test.utility.RANDOM_VALUE_SEED)
    elif isinstance(random_number_generator, int):
        random_number_generator = random.Random(random_number_generator)

    print(
        f"Generating a dataframe for multiple locations {'with' if by_time else 'without'} "
        f"a time element for thresholds"
    )
    generation_start_time = datetime.now()

    individual_location_start_time = datetime.now()
    print(f"Generating data for all of the different locations...")
    individual_locations: typing.Dict[str, pandas.DataFrame] = {
        location: make_test_dataframe(
            location=location,
            value_column=value_column,
            location_column=location_column,
            time_column=time_column,
            random_number_generator=random_number_generator,
            by_time=by_time
        )
        for location in LOCATIONS
    }
    print(
        f"It took {datetime.now() - individual_location_start_time} to generate 'observation' data for "
        f"{len(individual_locations)} locations"
    )

    threshold_generation_start_time = datetime.now()
    print(f"Generating thresholds for each location...")
    thresholds: typing.Mapping[str, pandas.DataFrame] = {
        location: make_thresholds_for_location(
            observation_data=individual_locations[location],
            location=location,
            location_column=location_column,
            value_column=value_column,
            random_number_generator=random_number_generator,
            by_time=by_time,
            time_column=time_column
        )
        for location in LOCATIONS
    }
    print(
        f"It took {datetime.now() - threshold_generation_start_time} to generate thresholds for "
        f"{len(thresholds)} locations"
    )

    combination_start_time = datetime.now()
    print(f"Combining observations with their thresholds...")
    paired_locations_and_thresholds = {
        location: frame.join(thresholds.get(location), how='inner')
        for location, frame in individual_locations.items()
        if location in thresholds
    }
    print(f"It took {datetime.now() - combination_start_time} to combine observations and their thresholds")

    final_combination_start_time = datetime.now()
    print(f"Combining all observations and thresholds into a single frame")
    final_frame = functools.reduce(
        lambda first_frame, second_frame: pandas.concat([first_frame, second_frame]),
        paired_locations_and_thresholds.values()
    )
    print(
        f"It took {datetime.now() - final_combination_start_time} to combine all observations and "
        f"thresholds into a single frame"
    )

    print(
        f"It took {datetime.now() - generation_start_time} to generate a multi-location dataset "
        f"{'with' if by_time else 'without'} a time element for thresholds"
    )
    return final_frame


class TestModel(unittest.TestCase):
    value_column: str
    location_column: str
    time_column: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.value_column = "value"
        cls.time_column = "valid_date"
        cls.location_column = "location"

        random_number_generator = random.Random(test.utility.RANDOM_VALUE_SEED)

        cls.variance = Variance.create_random_variance(
            start_date=START_DATE,
            end_date=END_DATE,
            random_number_generator=random_number_generator
        )

        cls.threshold_names = [f"threshold_{index}" for index in range(1, THRESHOLD_COUNT + 1)]
        print(f"[{cls.__class__.__name__}.setUp] Constructing Test Data")
        print(f"[{cls.__class__.__name__}.setUp] Constructing Single location input")

        cls.single_location_calculation_input = make_single_test_dataframe(
            location=LOCATIONS[0],
            value_column=cls.value_column,
            location_column=cls.location_column,
            time_column=cls.time_column,
            random_number_generator=random_number_generator
        )

        print(f"[{cls.__class__.__name__}.setUp] Constructing multiple location input")
        cls.multiple_location_calculation_input = make_multiple_location_test_dataframe(
            value_column=cls.value_column,
            location_column=cls.location_column,
            time_column=cls.time_column,
            random_number_generator=random_number_generator
        )

        print(f"[{cls.__class__.__name__}.setUp] Constructing Single location input with timed thresholds")
        cls.timed_single_location_calculation_input = make_single_test_dataframe(
            location=LOCATIONS[0],
            value_column=cls.value_column,
            location_column=cls.location_column,
            time_column=cls.time_column,
            random_number_generator=random_number_generator,
            by_time=True
        )

        print(f"[{cls.__class__.__name__}.setUp] Constructing multiple location input with timed thresholds")
        # 1 pretend dollar to whoever speeds this call up
        cls.timed_multiple_location_calculation_input = make_multiple_location_test_dataframe(
            value_column=cls.value_column,
            location_column=cls.location_column,
            time_column=cls.time_column,
            random_number_generator=random_number_generator,
            by_time=True
        )

        print(f"[{cls.__class__.__name__}] Test data has been prepared")

    def test_calculate_values(cls):
        test.utility.apply_seed()
        generated_data = model.calculate_values(
            group=cls.single_location_calculation_input,
            threshold_names=cls.threshold_names,
            variance=cls.variance,
            location=LOCATIONS[0],
            location_column=cls.location_column,
            time_column=cls.time_column,
            value_column=cls.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        simulation = pandas.DataFrame(generated_data).set_index(cls.location_column)

        cls.assertEqual(cls.single_location_calculation_input.shape[0], simulation.shape[0] + 1)

        values = simulation[cls.value_column]
        minimum = values.min()
        maximum = values.max()
        summed_values = values.sum()
        mean_error = (values - cls.single_location_calculation_input[cls.value_column]).mean()
        median = values.median()
        mean = values.mean()
        std = values.std()
        cls.assertAlmostEqual(mean_error, -2.3582806954098, EPSILON_DIGITS)
        cls.assertAlmostEqual(minimum, 17.573044169399243, EPSILON_DIGITS)
        cls.assertAlmostEqual(maximum, 144.0720234370857, EPSILON_DIGITS)
        cls.assertAlmostEqual(median, 77.2989369636704, EPSILON_DIGITS)
        cls.assertAlmostEqual(mean, 79.51445807678013, EPSILON_DIGITS)
        cls.assertAlmostEqual(summed_values, 59158.75680912442, EPSILON_DIGITS)
        cls.assertAlmostEqual(std, 38.66943581335177, EPSILON_DIGITS)

        del simulation

    def test_run_model(cls):
        single_location_simulation = model.run_model(
            control_and_thresholds=cls.single_location_calculation_input,
            threshold_names=cls.threshold_names,
            variance=cls.variance,
            time_column=cls.time_column,
            location_column=cls.location_column,
            value_column=cls.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        cls.assertEqual(cls.single_location_calculation_input.shape[0], single_location_simulation.shape[0] + 1)

        del single_location_simulation

        multiple_location_simulation = model.run_model(
            cls.multiple_location_calculation_input,
            threshold_names=cls.threshold_names,
            variance=cls.variance,
            time_column=cls.time_column,
            location_column=cls.location_column,
            value_column=cls.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        cls.assertEqual(
            cls.multiple_location_calculation_input.shape[0],
            multiple_location_simulation.shape[0] + len(LOCATIONS)
        )

        del multiple_location_simulation

        timed_single_location_simulation = model.run_model(
            control_and_thresholds=cls.timed_single_location_calculation_input,
            threshold_names=cls.threshold_names,
            variance=cls.variance,
            time_column=cls.time_column,
            location_column=cls.location_column,
            value_column=cls.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        del timed_single_location_simulation

        timed_multiple_location_simulation = model.run_model(
            cls.timed_multiple_location_calculation_input,
            threshold_names=cls.threshold_names,
            variance=cls.variance,
            time_column=cls.time_column,
            location_column=cls.location_column,
            value_column=cls.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )




if __name__ == '__main__':
    unittest.main()
