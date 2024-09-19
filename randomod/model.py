"""
@TODO: Put a module wide description here
"""
from __future__ import annotations

import math
import typing
import multiprocessing
import functools

from datetime import datetime
from datetime import timedelta
from random import Random

import numpy
import pandas

from randomod.schema.variance import ThresholdAdjustment
from randomod.schema.variance import Variance
from randomod.schema.variance import VarianceEntry
from onlyday import DayDict

BOUND_SECTION_COUNT = 8
BOUND_DISTANCE = BOUND_SECTION_COUNT // 3
OVERFLOW_THRESHOLD_NAME = "overflow"


def isnan(value) -> bool:
    if value is None:
        return True
    return math.isnan(value) or numpy.isnan(value) or pandas.isna(value)


def not_nan(value) -> bool:
    return not isnan(value)


def generate_value_from_source(
    source: pandas.Series,
    threshold_names: typing.List[str],
    variance: Variance,
    time_column: str,
    value_column: str,
    random_number_generator: Random
) -> float:
    thresholds: typing.Dict[str, float] = dict(
        sorted(
            [
                (threshold_name, source[threshold_name])
                for threshold_name in threshold_names
                if threshold_name in source
                   and not_nan(source[threshold_name])
            ], key=lambda x: x[1]
        )
    )

    current_time: pandas.Timestamp = source[time_column]
    variance_entry: VarianceEntry = variance.get_entry_for_date(current_time)

    if variance_entry is None:
        return numpy.nan

    observed_value: float = source[value_column]

    if isnan(observed_value):
        return numpy.nan

    lower_bound, upper_bound = get_threshold_bounds(
        date=current_time,
        previous_value=observed_value,
        variance=variance,
        thresholds=thresholds,
    )

    if variance_entry.threshold_adjustment == ThresholdAdjustment.STAY:
        # Widen the selection space when staying within the same bounds - this will help prevent the
        # value from being TOO close and from being TOO far
        lower_bound, upper_bound = create_microbounds(
            previous_value=observed_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            block_selection_count=int(BOUND_SECTION_COUNT * 0.75)
        )
    else:
        lower_bound, upper_bound = create_microbounds(
            previous_value=observed_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    new_value = random_number_generator.uniform(lower_bound, upper_bound)
    return new_value


def calculate_model_values(
    group: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    location: str,
    time_column: str,
    value_column: str,
    random_number_generator: typing.Union[Random, int, typing.Type[Random]] = None
) -> pandas.DataFrame:
    """
    Calculate the values of a model based on the values of an observation in relation to thresholds and instructions
    on whether to stay at a given threshold, an increase to a new threshold, or a decrese to a previous threshold

    `group` can be expected to look like:

    +----------+------------+---------------------+-------------+------+--------------+
    | location | value      | valid_date          | threshold 1 |  ... | threshold N  |
    +==========+============+=====================+=============+======+==============+
    | name     | 133.892079 | 2023-10-08 12:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 59.768121  | 2023-10-08 13:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 41.513576  | 2023-10-08 14:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 91.517383  | 2023-10-08 15:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 106.339565 | 2023-10-08 16:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 49.834195  | 2023-10-08 17:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+
    | name     | 146.227073 | 2023-10-08 18:00:00 | 39.881315   | ...  | 185.325346   |
    +----------+------------+---------------------+-------------+------+--------------+

    :param group:
    :param threshold_names:
    :param variance:
    :param location:
    :param time_column:
    :param location_column:
    :param value_column:
    :param random_number_generator:
    :return:
    """
    print(f"Beginning to model data for {location}...")
    random_number_generator = initialize_random_number_generator(random_number_generator)

    model_start_time: datetime = datetime.now()

    modeled_values: pandas.DataFrame = group[[time_column, value_column, *threshold_names]].copy()

    calculator = functools.partial(
        generate_value_from_source,
        threshold_names=threshold_names,
        variance=variance,
        time_column=time_column,
        value_column=value_column,
        random_number_generator=random_number_generator
    )

    modeled_values[value_column] = modeled_values.apply(calculator, axis=1)
    modeled_values.drop([*threshold_names], axis=1, inplace=True)

    duration = datetime.now() - model_start_time
    print(f"{len(modeled_values)} generated for {location} in {duration}")
    return modeled_values


def initialize_random_number_generator(
    random_number_generator: typing.Union[Random, int, typing.Type] = None
) -> Random:
    """
    Generate a random number generator

    :param random_number_generator: A generator, a type of generator, or a seed for a generator
    :return: A random number generator responsible for creating model values
    """
    if random_number_generator is None:
        return Random()
    if isinstance(random_number_generator, Random):
        return random_number_generator
    if isinstance(random_number_generator, int):
        return Random(random_number_generator)
    if isinstance(random_number_generator, type) and issubclass(random_number_generator, Random):
        return random_number_generator()
    return Random()


def get_overflow_threshold(threshold_index: int, value: float, last_maximum: float) -> typing.Tuple[int, str, float]:
    """
    Get threshold data for a value that surpasses the maximum threshold

    :param threshold_index: The index of the final threshold
    :param value: The value that surpassed the thresholds
    :param last_maximum: The largest value for the group of thresholds
    :return: The index, threshold name, and threshold maximum for a synthetic threshold that surpasses the maximum threshold
    """
    synthetic_maximum = last_maximum + (last_maximum + value) / 2

    if synthetic_maximum <= value:
        synthetic_maximum = value + value * 0.2

    return threshold_index + 1, OVERFLOW_THRESHOLD_NAME, synthetic_maximum


def get_threshold_for_value(
    value: float,
    thresholds: typing.Dict[str, float]
) -> typing.Optional[typing.Tuple[int, str, float]]:
    """
    Determine the threshold for the given value and what its upper limit should be. An artificial one will be created
    for a value that surpasses the final threshold

    :param value: The value whose threshold to find
    :param thresholds: The collection of all thresholds
    :return: The index of the appropriate threshold, name of the appropriate threshold and its maximum possible value
    """
    threshold_name: typing.Optional[str] = None
    threshold_maximum: typing.Optional[float] = None
    threshold_index: typing.Optional[int] = None

    for threshold_index, (threshold_name, threshold_maximum) in enumerate(thresholds.items()):
        # If we're on the last threshold and the value goes over, we want to generate an artifical threshold
        if threshold_index == len(thresholds) - 1 and value > threshold_maximum:
            return get_overflow_threshold(threshold_index=threshold_index, value=value, last_maximum=threshold_maximum)

        if value <= threshold_maximum:
            break

    return threshold_index, threshold_name, threshold_maximum


def get_threshold_bounds(
    date: datetime,
    previous_value: float,
    variance: Variance,
    thresholds: typing.Dict[str, float],
    previous_value_date: datetime = None,
) -> typing.Optional[typing.Tuple[float, float]]:
    threshold_names: typing.List[str] = list([name for name, maximum in thresholds.items() if not_nan(maximum)])
    previous_threshold_index, previous_threshold_name, previous_threshold_maximum = get_threshold_for_value(
        previous_value,
        thresholds
    )

    previous_variance = variance.get_entry_for_date(previous_value_date) if previous_value_date else None
    current_variance = variance.get_entry_for_date(date)

    if current_variance == previous_variance or current_variance.threshold_adjustment == ThresholdAdjustment.STAY:
        expected_threshold_index = previous_threshold_index

        if expected_threshold_index == 0:
            return (
                previous_threshold_maximum * 0.5,
                previous_threshold_maximum
            )
    elif current_variance.threshold_adjustment == ThresholdAdjustment.NEXT:
        expected_threshold_index = previous_threshold_index + 1

        if expected_threshold_index >= len(threshold_names):
            last_maximum = thresholds.get(threshold_names[len(threshold_names) - 1], previous_value)
            return last_maximum, last_maximum + last_maximum * 0.1
    else:
        expected_threshold_index = previous_threshold_index - 1

        if expected_threshold_index < 0:
            return previous_value - previous_value * 0.25, previous_value - previous_value * 0.1
        elif expected_threshold_index == 0:
            return (
                thresholds.get(threshold_names[expected_threshold_index], previous_value) / 2.0,
                thresholds.get(threshold_names[expected_threshold_index], previous_value)
            )

    if expected_threshold_index >= len(threshold_names):
        last_maximum = thresholds.get(threshold_names[len(threshold_names) - 1], previous_value)
        return last_maximum, last_maximum + last_maximum * 0.1

    expected_threshold_name = threshold_names[expected_threshold_index]

    return (
        thresholds.get(threshold_names[expected_threshold_index - 1], previous_value),
        thresholds.get(expected_threshold_name, previous_value + previous_value * 0.25)
    )


def create_microbounds(
    previous_value: float,
    lower_bound: float,
    upper_bound: float,
    block_count: int = BOUND_SECTION_COUNT,
    block_selection_count: int = BOUND_DISTANCE
) -> typing.Tuple[float, float]:
    """
    Cut a lower and upper bound into blocks and calculate new bounds that cover a limited area. This will help
    control the variablity of values when selecting new values.

    :param previous_value: The previous value that will inform what blocks in the microbounds to use
    :param lower_bound: The smallest possible value for the bounds
    :param upper_bound: The largets possible value for the bounds
    :param block_count: The number of blocks to partition the bounds into
    :param block_selection_count: The number of blocks that the microbounds should encompass
    :return: New lower and upper bounds that will constrain what random values will be generated
    """
    step_size = (upper_bound - lower_bound) / block_count
    blocks: typing.List[typing.Tuple[float, float]] = []
    block_max: float = numpy.nan
    block_min: float = numpy.nan

    for block_index in range(1, block_count + 1):
        block_boundaries = (
            (lower_bound + 0.00000000000000001) + step_size * (block_index - 1),
            lower_bound + step_size * block_index,
        )
        blocks.append(block_boundaries)

        if isnan(block_min) or block_boundaries[0] < block_min:
            block_min = block_boundaries[0]

        if isnan(block_max) or block_boundaries[1] > block_max:
            block_max = block_boundaries[1]

    if previous_value <= block_min:
        lower_bound = blocks[0][0]
        upper_bound = blocks[BOUND_DISTANCE - 1][1]
    elif previous_value > block_max:
        lower_bound = blocks[-BOUND_DISTANCE][0]
        upper_bound = blocks[-1][1]
    else:
        previous_block_index = 0
        for block_index, block in enumerate(blocks):
            if block[0] < previous_value <= block[1]:
                previous_block_index = block_index
                break

        if block_selection_count >= block_count:
            lower_bound = blocks[0][0]
            upper_bound = blocks[block_selection_count - 1][1]
        elif previous_block_index < block_selection_count:
            lower_bound = blocks[0][0]
            upper_bound = blocks[block_selection_count - 1][1]
        elif previous_block_index > block_count - block_selection_count:
            lower_bound = blocks[-block_selection_count][0]
            upper_bound = blocks[-1][1]
        else:
            split_distance = math.floor(previous_block_index / block_count)
            initial_index = max(previous_block_index - split_distance, 0)
            stride = min(previous_block_index + split_distance, block_count - 1)
            lower_bound = blocks[initial_index][0]
            upper_bound = blocks[stride][1]
    return lower_bound, upper_bound


def calculate_from_scratch(
    thresholds: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    location: str,
    start: datetime,
    end: datetime,
    period: timedelta,
    time_column: str,
    location_column: str,
    value_column: str,
    day_column: str,
    random_number_generator: typing.Union[Random, int, typing.Type[Random]] = None
) -> typing.Sequence[typing.Dict[str, typing.Union[float, str, datetime]]]:
    """
    Generates values in relation to a random point value rather than in relation to observations

    Expect `thresholds` to look like:

    +-----------+-------------+-----+-------------+
    | day       | threshold_1 | ... | threshold_n |
    +===========+=============+=====+=============+
    | January 1 | 39.5483     | ... | 72.5486     |
    +-----------+-------------+-----+-------------+
    | January 2 | 40.5783     | ... | 71.5856     |
    +-----------+-------------+-----+-------------+
    | January 3 | 37.5753     | ... | 74.8789     |
    +-----------+-------------+-----+-------------+

    :param thresholds: A frame containing threshold values per day
    :param threshold_names: The names of the thresholds within the thresholds frame
    :param variance: Definitions for how values should vary into and out of thresholds
    :param location: The name of the location being modeled
    :param start: The start of the time range
    :param end: The end of the time range
    :param period: The time between data points
    :param time_column: The name of the column containing the time values
    :param location_column: The name of the column containing the location values
    :param value_column: The name of the column containing generated values
    :param day_column: The name of the column in the thresholds frame defining the days that thresholds correspond to
    :param random_number_generator: The random number generator or seed that informs how values are generated
    :return: A randomly generated time series for the given location
    """
    print(f"Modeling data for {location}...")
    random_number_generator = initialize_random_number_generator(random_number_generator)
    new_values: typing.List[typing.Dict[str, typing.Union[float, str, datetime, pandas.Timestamp]]] = []

    model_start_time = datetime.now()
    current_time = start

    mean_per_threshold = [
        thresholds[threshold_name].mean()
        for threshold_name in threshold_names
        if threshold_name in thresholds
    ]

    previous_modeled_value = numpy.mean(mean_per_threshold)

    thresholds_per_day = DayDict(*[
        (
            day,
            dict(
                sorted([
                    (name, value)
                    for name, value in list(threshold_data_for_day.to_dict(orient="index").values())[0].items()
                    if name in threshold_names
                       and not_nan(value)
                ], key=lambda pair: pair[1]
                )
            )
        )
        for day, threshold_data_for_day in thresholds.groupby(day_column)
    ])

    while current_time < end:
        previous_date = current_time
        current_time += period

        threshold_values_for_day: typing.Dict[str, float] = thresholds_per_day[current_time]

        lower_bound, upper_bound = get_threshold_bounds(
            date=current_time,
            previous_value=previous_modeled_value,
            previous_value_date=previous_date,
            variance=variance,
            thresholds=threshold_values_for_day,
        )

        lower_bound, upper_bound = create_microbounds(
            previous_value=previous_modeled_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        modeled_value = random_number_generator.uniform(lower_bound, upper_bound)

        new_values.append({
            location_column: location,
            time_column: current_time,
            value_column: modeled_value,
        })
        previous_modeled_value = modeled_value

    duration = datetime.now() - model_start_time
    print(f"{len(new_values)} time steps generated for {location} in {duration}")
    return new_values


def run_model_from_scratch(
    thresholds: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    start: datetime,
    end: datetime,
    period: timedelta,
    time_column: str,
    location_column: str,
    value_column: str,
    day_column: str,
    random_number_generator: typing.Union[Random, int, typing.Type[Random]] = None
) -> pandas.DataFrame:
    with multiprocessing.Pool() as pool:
        arguments = [
            [
                location_thresholds,
                threshold_names,
                variance,
                location,
                start,
                end,
                period,
                time_column,
                location_column,
                value_column,
                day_column,
                random_number_generator
            ]
            for location, location_thresholds in thresholds.groupby(location_column)
        ]

        simulation_per_location = pool.starmap(
            calculate_from_scratch,
            arguments
        )

        return pandas.DataFrame(
            functools.reduce(
                lambda combined_list, new_list: [*combined_list, *new_list],
                simulation_per_location
            )
        )


def run_model(
    control_and_thresholds: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    time_column: str,
    location_column: str,
    value_column: str,
    random_number_generator: typing.Union[Random, int] = None
) -> pandas.DataFrame:
    """
    Run the 'model' and produce values based off of control values and daily thresholds

    :param control_and_thresholds: A dataframe of control values and thresholds
    :param threshold_names: The names of all the thresholds that exist within the dataframe
    :param variance: Instructions on how and when the modeled data should differ from the input
    :param time_column: The column that contains datetime information
    :param location_column: The column that contains location identifiers
    :param value_column: The column that contains model/input values
    :param random_number_generator: Information used to create or use a random number generator
    :return: A dataframe containing modeled data for all locations within the input control and thresholds
    """
    if control_and_thresholds.index.names != [location_column]:
        control_and_thresholds = control_and_thresholds.reset_index().set_index(location_column)

    with multiprocessing.Pool() as process_pool:
        arguments = [
                [
                    group,
                    threshold_names,
                    variance,
                    location,
                    time_column,
                    value_column,
                    random_number_generator
                ]
                for location, group in control_and_thresholds.groupby(location_column)  # type: typing.Tuple[str], pandas.DataFrame
            ]

        simulation_per_location = process_pool.starmap(
            calculate_model_values,
            arguments
        )

    combined_simulated_data = pandas.concat(simulation_per_location)
    return combined_simulated_data