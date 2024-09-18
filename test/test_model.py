"""
Tests used to make sure that model calculations and the run model function work correctly
"""
import functools
import itertools
import os
import typing
import unittest
import random
from concurrent import futures

from datetime import datetime
from datetime import timedelta

import pandas

from randomod.schema.variance import ThresholdAdjustment
from randomod.schema.variance import Variance
from randomod import model

import test.utility
from randomod.utilities import Day
from randomod.utilities import get_unique_sequence_values

START_DATE = datetime(year=2023, month=10, day=8, hour=12)
"""The start time for model input data"""

END_DATE = datetime(year=2023, month=11, day=8, hour=12)
"""The end time for model input data"""

RESOLUTION = timedelta(hours=1)
"""The amount of time between values generated observations"""

THRESHOLD_COUNT = 5
"""The number of generated thresholds to use"""

THRESHOLD_VALUE_ADJUSTER = 8
"""
A value used to adjust threshold bounds to ensure that threshold bounds are always over the greatest value 
for a set of observations
"""

MINIMUM_VALUE = 15
"""The smallest value that may be generated"""

MAXIMUM_VALUE = 150
"""The largest value that may be generated"""

EPSILON_DIGITS = 4
"""The number of digits to use when comparing generated output and expected output"""

LOCATIONS = [
    f"location_{index}"
    for index in range(1, 201)
]
"""The locations to generate data for"""

OVERFLOW_THRESHOLD_INDEX = 999

test.utility.apply_seed()


def next_generated_value(random_number_generator: random.Random, previous_value: float = None) -> float:
    """
    Generates a new value in relation to the previous

    :param random_number_generator: The random number generator responsible for determining values
    :param previous_value: The value to use as the root of the next value
    :return: A random value
    """
    if previous_value is None:
        # If no value was passed (generally the case for the first value), generate a value in relation to the
        # midpoint of the minimum and maximum
        previous_value = (MINIMUM_VALUE + MAXIMUM_VALUE) / 2.0

    # Create a range of values for the generated value that expand from a range half of the previous value below it
    # to half the previous value above it.
    # If the previous value were 40, for instance, the range for the next value would be [20, 60]
    # This approach generates wild behavior at larger values, so that might need to be addressed so that values don't
    # look ridiculous
    half_below = previous_value / 2.0
    half_above = previous_value + half_below

    # If the range might extend below the minimum value, shift the range up
    if half_below < MINIMUM_VALUE:
        difference = half_above - half_below
        half_below = MINIMUM_VALUE
        half_above = MINIMUM_VALUE + difference
    elif half_above > MAXIMUM_VALUE:
        # If the range might extend above the maximum value, shift the range down
        difference = half_above - half_below
        half_above = MAXIMUM_VALUE
        half_below = MAXIMUM_VALUE - difference

    # Choose a random value from within the range of the minimum and maximum and the half steps generated above
    return random_number_generator.uniform(
        min(MINIMUM_VALUE, half_below),
        max(MAXIMUM_VALUE, half_above)
    )


def create_threshold_value_table(
    row_count: int,
    threshold_values: typing.List[float],
    random_number_generator: random.Random
) -> pandas.DataFrame:
    """
    Create a table containing values to be joined to generated metadata to serve as thresholds

    :param row_count: The number of threshold sets to generate
    :param threshold_values: The generated center points for each stage of the thresholds
    :param random_number_generator: A random number generator used to generate semi-reliable values
    :return: A table containing thresholds that may be merged with metadata used as indices
    """
    threshold_value_input = {}

    # Loop through each established threshold and create random values rooted around and established
    # threshold center point
    # Create values that will align with the established row count
    for threshold_index, center_value in enumerate(threshold_values):
        # First establish a name for the threshold
        threshold_number = str(threshold_index + 1)
        threshold_name = "threshold_" + threshold_number

        # Create an adjustment value to ensure that thresholds may reliably encompass generated values
        # If the center value was 40, this would give us 10
        adjustment_amount = 0.25
        adjustment = center_value * adjustment_amount

        # The bounds for the random number generator will now be created by generating random bounds based on the
        # threshold center points
        # If the center value for the threshold being generated is 40, this will give us a range of bounds for our
        # final lower bound from [30, 50]
        lower_lower_bound = center_value - adjustment
        upper_lower_bound = center_value + adjustment

        # Lower bounds for generation will be generated based on those bounds for every expected row
        lower_bounds = [
            random_number_generator.uniform(
                lower_lower_bound,
                upper_lower_bound
            )
            for _ in range(row_count)
        ]

        # Now establish upper bounds based on the generated bound for each lower bound
        # If a generated lower bound was 32.5, this will give us a range of potential upper bounds from
        # [32.75, 42.5]. The adjustment amount is added to the lower bound to ensure that the upper bound is
        # never equal to the lower bound
        upper_bounds = [
            random_number_generator.uniform(lower_bound + adjustment_amount, lower_bound + adjustment)
            for lower_bound in lower_bounds
        ]

        # Now use the generated upper and lower bounds to generate values for each expected row
        threshold_value_input[threshold_name] = [
            random_number_generator.uniform(lower_bound, upper_bound)
            for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)
        ]

    # Return the data in a dataframe for efficient joining
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
    """
    Make a table describing thresholds for a location based on 'observed' values

    :param observation_data: The values that are considered to be true and reasonable for the given location
    :param location: The name of the location
    :param location_column: The name of the column where the location name should lie
    :param value_column: The name of the column where the observed value may be found
    :param random_number_generator: A random number generator used to make semi-reliable data
    :param by_time: Whether to include the time dimension when generating values. This helps mimic variable percentiles
    :param time_column: The column to use that will help identify the time to use when generating time-based thresholds
    :return: A dataframe used to serve as artificial thresholds for a location based on observation data
    """
    values = observation_data[value_column]

    minimum_value = values.min()
    maximum_value = values.max()

    threshold_width = minimum_value + (minimum_value + maximum_value - THRESHOLD_VALUE_ADJUSTER) / THRESHOLD_COUNT
    """The expected width of each range of allowable threshold values"""

    # Create a series of equidistant values to serve as threshold boundaries that will ALWAYS exceed the minimum and
    # maximum values
    threshold_values = [
        threshold_width * threshold_index
        for threshold_index in range(1, THRESHOLD_COUNT + 1)
    ]

    # If there is multiple indices, we're likely working with a location and time and will need to step through the
    # index differently as compared to a single index since it will be multidimensional
    # Given a 1-dimensional index named 'location' with a single value of 'location_1', the multiindex logic will
    # yield an index of {'location': 'l'}. If the logic for single indexed values is used on multiindexed values of
    # [{location: (location1, date1)}, {location: (location1, date2)}, ...] where we need
    # [{location: location1, date: date1}, {location: location1, date: date2}, ...]
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

    # Make sure that the location is added if it somehow hasn't yet
    for row in threshold_frame_contents:
        if location_column not in row:
            row[location_column] = location

    # Add the time if it hasn't been added
    if by_time and time_column in observation_data:
        for row, time_value in zip(threshold_frame_contents, observation_data[time_column]):
            if time_column not in row:
                row[time_column] = time_value

    threshold_frame = pandas.DataFrame(threshold_frame_contents)
    raw_threshold_values = create_threshold_value_table(
        threshold_frame.shape[0],
        threshold_values,
        random_number_generator
    )
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


def create_thresholds_for_location(
    location: str,
    location_column: str,
    lower_wobble: float,
    upper_wobble: float,
    day_column: str,
    threshold_names: typing.Sequence[str],
    random_number_generator: random.Random
) -> typing.List[typing.Dict[str, typing.Union[float, str]]]:
    thresholds: typing.List[typing.Dict[str, typing.Union[float, str]]] = []

    initial_threshold_value: float = random_number_generator.uniform(MINIMUM_VALUE, MAXIMUM_VALUE)
    increase_amount: float = random_number_generator.uniform(
        initial_threshold_value + lower_wobble,
        initial_threshold_value + upper_wobble
    )
    prior_threshold_value = None
    for day_number in range(1, 366):
        thresholds_for_location = {
            location_column: location,
            day_column: Day(day_number),
        }

        for threshold_name in threshold_names:
            if prior_threshold_value is None:
                prior_threshold_value = initial_threshold_value + random_number_generator.uniform(
                    lower_wobble,
                    upper_wobble
                )
            else:
                prior_threshold_value += random_number_generator.uniform(
                    increase_amount + lower_wobble,
                    increase_amount + upper_wobble
                )
            thresholds_for_location[threshold_name] = prior_threshold_value

        thresholds.append(thresholds_for_location)
        prior_threshold_value = None
    return thresholds

def make_context_free_thresholds(
    location_column: str,
    day_column: str,
    threshold_names: typing.Sequence[str],
    random_number_generator: random.Random
):
    generation_start_time: datetime = datetime.now()
    thresholds: typing.List[typing.Dict[str, typing.Union[float, str]]] = []
    lower_wobble: float = -1.5
    upper_wobble: float = 1.5

    with futures.ProcessPoolExecutor() as executor:
        process_list: typing.List[futures.Future] = [
            executor.submit(
                create_thresholds_for_location,
                location,
                location_column,
                lower_wobble,
                upper_wobble,
                day_column,
                threshold_names,
                random_number_generator
            )
            for location in LOCATIONS
        ]

        while process_list:
            process: futures.Future = process_list.pop()
            if process.done():
                thresholds.extend(process.result())
            else:
                process_list.append(process)

    print(f"Context free thresholds generated in {datetime.now() - generation_start_time}")
    return pandas.DataFrame(thresholds)


def get_threshold_for_value(row: pandas.Series) -> str:
    value = row['value']
    thresholds = (
        (threshold_name, row[threshold_name])
        for threshold_name in [name for name in row.index.values if name.lower().startswith('threshold')]
    )

    previous_maximum: float = 0.0
    threshold_name: str = ""
    for threshold_name, threshold_value in thresholds:
        if previous_maximum < value <= threshold_value:
            break
        previous_maximum = threshold_value

    return threshold_name


def determine_threshold_index(row: pandas.Series, value_column_name: str, threshold_names: typing.List[str]) -> int:
    possible_thresholds = sorted([
        (threshold_index, row[name])
        for threshold_index, name in enumerate(threshold_names)
        if name in row
           and row[name] > row[value_column_name]
    ], key=lambda pair: pair[1])
    if possible_thresholds:
        return possible_thresholds[0][0]
    return OVERFLOW_THRESHOLD_INDEX


def verify_value_variance(
    row: pandas.Series,
    variance_column: str,
    control_threshold_column: str,
    simulation_threshold_column: str
) -> bool:
    variance = row[variance_column]
    control_threshold_index = row[control_threshold_column]
    simulation_threshold_index = row[simulation_threshold_column]

    if variance == ThresholdAdjustment.STAY:
        return control_threshold_index == simulation_threshold_index
    elif variance == ThresholdAdjustment.PREVIOUS:
        return control_threshold_index > simulation_threshold_index or control_threshold_index == 0 and simulation_threshold_index == 0

    return control_threshold_index < simulation_threshold_index or control_threshold_index == OVERFLOW_THRESHOLD_INDEX and simulation_threshold_index == 999


class TestModel(unittest.TestCase):
    value_column: str
    location_column: str
    time_column: str
    day_column: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.value_column = "value"
        cls.time_column = "valid_date"
        cls.location_column = "location"
        cls.day_column = "day"
        cls.control_threshold_column = "control_threshold"
        cls.simulation_threshold_column = "simulation_threshold"
        cls.simulation_variance_column = "simulation_variance"
        cls.simulation_suffix = "_sim"
        cls.control_suffix = "_control"
        cls.sim_control_suffix = cls.simulation_suffix, cls.control_suffix

        random_number_generator = random.Random(test.utility.RANDOM_VALUE_SEED)
        control_random_number_generator = random.Random(853).randint(57, 107)

        cls.control_variance = Variance.create_random_variance(
            start_date=START_DATE,
            end_date=END_DATE,
            random_number_generator=control_random_number_generator
        )

        cls.simulation_variance = Variance.create_random_variance(
            start_date=START_DATE,
            end_date=END_DATE,
            random_number_generator=random_number_generator
        )

        cls.threshold_names = [f"threshold_{index}" for index in range(1, THRESHOLD_COUNT + 1)]
        cls.control_threshold_finder = functools.partial(
            determine_threshold_index,
            value_column_name=cls.value_column,
            threshold_names=cls.threshold_names
        )
        cls.simulation_threshold_finder = functools.partial(
            determine_threshold_index,
            value_column_name=cls.value_column + cls.simulation_suffix,
            threshold_names=cls.threshold_names
        )
        cls.value_variance_verifier = functools.partial(
            verify_value_variance,
            variance_column=cls.simulation_variance_column,
            control_threshold_column=cls.control_threshold_column,
            simulation_threshold_column=cls.simulation_threshold_column
        )

        cls.thresholds = make_context_free_thresholds(
            location_column=cls.location_column,
            day_column=cls.day_column,
            threshold_names=cls.threshold_names,
            random_number_generator=random_number_generator
        )

        print(f"[{cls.__qualname__}.setUpClass] Constructing Test Data")
        print(f"[{cls.__qualname__}.setUpClass] Constructing Single location input")

        single_location_model_rows = model.calculate_from_scratch(
            thresholds=cls.thresholds[cls.thresholds[cls.location_column] == LOCATIONS[0]],
            threshold_names=cls.threshold_names,
            variance=cls.control_variance,
            location=LOCATIONS[0],
            start=START_DATE,
            end=END_DATE,
            period=RESOLUTION,
            value_column=cls.value_column,
            location_column=cls.location_column,
            day_column=cls.day_column,
            time_column=cls.time_column,
            random_number_generator=control_random_number_generator
        )

        single_location_model = pandas.DataFrame(single_location_model_rows)
        single_location_model.reset_index(inplace=True)
        single_location_model[cls.day_column] = single_location_model[cls.time_column].apply(Day)

        cls.single_location_calculation_input = pandas.merge(
            single_location_model,
            cls.thresholds,
            how='inner',
            on=[cls.location_column, cls.day_column]
        ).set_index(cls.location_column)

        cls.single_location_calculation_input[cls.control_threshold_column] = cls.single_location_calculation_input.apply(
            cls.control_threshold_finder,
            axis=1
        )

        print(f"[{cls.__qualname__}.setUpClass] Constructing multiple location input")
        multiple_location_data = model.run_model_from_scratch(
            thresholds=cls.thresholds,
            threshold_names=cls.threshold_names,
            variance=cls.control_variance,
            start=START_DATE,
            end=END_DATE,
            period=RESOLUTION,
            time_column=cls.time_column,
            value_column=cls.value_column,
            location_column=cls.location_column,
            day_column=cls.day_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        multiple_location_data[cls.day_column] = multiple_location_data[cls.time_column].apply(Day)

        cls.multiple_location_calculation_input = pandas.merge(
            multiple_location_data,
            cls.thresholds,
            how="inner",
            on=[cls.location_column, cls.day_column]
        ).set_index(cls.location_column)

        print(f"[{cls.__qualname__}] Test data has been prepared")

    def test_calculate_values_from_scratch(self) -> None:
        self.assertEqual((END_DATE - START_DATE) / RESOLUTION, len(self.single_location_calculation_input))
        self.assertIn(self.time_column, self.single_location_calculation_input)
        self.assertIn(self.value_column, self.single_location_calculation_input)
        self.assertIn(
            self.location_column,
            [*self.single_location_calculation_input.columns, *self.single_location_calculation_input.index.names]
        )

    def test_multi_location_calculation_from_scratch(self) -> None:
        self.assertEqual(
            ((END_DATE - START_DATE) / RESOLUTION) * len(LOCATIONS), len(self.multiple_location_calculation_input)
        )
        self.assertIn(self.time_column, self.multiple_location_calculation_input)
        self.assertIn(self.value_column, self.multiple_location_calculation_input)
        self.assertIn(
            self.location_column,
            [
                *self.multiple_location_calculation_input.columns,
                *self.multiple_location_calculation_input.index.names
            ]
        )

    def test_calculate_values(self):
        test.utility.apply_seed()
        simulation = model.calculate_model_values(
            group=self.single_location_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.simulation_variance,
            location=LOCATIONS[0],
            time_column=self.time_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        self.assertEqual((END_DATE - START_DATE) / RESOLUTION, len(simulation))
        self.assertEqual(self.single_location_calculation_input.shape[0], simulation.shape[0])

        combined_frames: pandas.DataFrame = pandas.merge(
            simulation,
            self.single_location_calculation_input,
            how="inner",
            suffixes=self.sim_control_suffix,
            on=[self.location_column, self.time_column]
        ).set_index([self.time_column])

        combined_frames[self.simulation_variance_column] = self.simulation_variance.to_frame(RESOLUTION).set_index(self.time_column)['adjustment']
        combined_frames[self.simulation_threshold_column] = combined_frames.apply(self.simulation_threshold_finder, axis=1)
        variance_is_correct = combined_frames.apply(self.value_variance_verifier, axis=1)

        with pandas.option_context('display.max_columns', None, 'display.width', 500):
            incorrect_variance = combined_frames[variance_is_correct == False]
            message = (
                f"Variance was not correctly used "
                f"{len(variance_is_correct[variance_is_correct == False]) / len(variance_is_correct) * 100.0:.2f}% "
                f"of the time{os.linesep}{repr(incorrect_variance)}"
            )
            self.assertTrue(variance_is_correct.all(), message)

    def test_run_model(self):
        single_location_simulation = model.run_model(
            control_and_thresholds=self.single_location_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.simulation_variance,
            time_column=self.time_column,
            location_column=self.location_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )

        self.assertEqual((END_DATE - START_DATE) / RESOLUTION, len(single_location_simulation))
        self.assertEqual(self.single_location_calculation_input.shape[0], single_location_simulation.shape[0])
        del single_location_simulation

        multiple_location_simulation = model.run_model(
            self.multiple_location_calculation_input,
            threshold_names=self.threshold_names,
            variance=self.simulation_variance,
            time_column=self.time_column,
            location_column=self.location_column,
            value_column=self.value_column,
            random_number_generator=test.utility.RANDOM_VALUE_SEED
        )
        self.assertEqual(
            ((END_DATE - START_DATE) / RESOLUTION) * len(LOCATIONS), len(multiple_location_simulation)
        )

        self.assertEqual(
            self.multiple_location_calculation_input.shape[0],
            multiple_location_simulation.shape[0]
        )


if __name__ == '__main__':
    unittest.main()
