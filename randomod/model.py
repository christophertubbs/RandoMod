"""
@TODO: Put a module wide description here
"""
from __future__ import annotations

import math
import typing
import multiprocessing
import functools

from datetime import datetime
from random import Random

import numpy
import pandas

from randomod.schema.variance import ThresholdAdjustment
from randomod.schema.variance import Variance
from randomod.schema.variance import VarianceEntry


def isnan(value) -> bool:
    if value is None:
        return True
    return math.isnan(value) or numpy.isnan(value) or pandas.isna(value)


def not_nan(value) -> bool:
    return not isnan(value)


def calculate_values(
    group: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    location: str,
    time_column: str,
    location_column: str,
    value_column: str,
    random_number_generator: typing.Union[Random, int] = None
) -> typing.Sequence[typing.Dict[str, typing.Union[float, str, datetime]]]:
    print(f"Modelling data for {location}")
    if random_number_generator is None or isinstance(random_number_generator, int):
        random_number_generator = Random(random_number_generator)

    if not isinstance(random_number_generator, Random):
        random_number_generator = Random()

    new_values = []

    final_threshold_index = len(threshold_names) - 1
    previous_modelled_value = numpy.nan

    model_start_time = datetime.now()
    for index, row in group.iterrows():
        current_time = row[time_column]

        new_value = {
            location_column: location,
            time_column: current_time,
            value_column: previous_modelled_value
        }

        variance_entry: VarianceEntry = variance.get_entry_for_date(current_time)

        if variance_entry is None:
            continue

        observed_value: float = row[value_column]

        if not_nan(observed_value):
            for threshold_index, threshold_name in enumerate(threshold_names):
                on_final_threshold = threshold_index == final_threshold_index
                threshold_upper_bound = row[threshold_name]

                if isnan(threshold_upper_bound):
                    continue

                name_of_previous_threshold = threshold_names[threshold_index - 1] if threshold_index > 0 else None

                threshold_lower_bound = numpy.nan

                if name_of_previous_threshold is not None:
                    threshold_lower_bound = row[name_of_previous_threshold]

                if isnan(threshold_lower_bound):
                    threshold_lower_bound = threshold_upper_bound / 2.0

                observed_lower_midpoint = (observed_value + threshold_lower_bound) / 2.0

                threshold_midpoint = (threshold_upper_bound + threshold_lower_bound) / 2.0

                if threshold_upper_bound > observed_value:
                    if variance_entry.threshold_adjustment is ThresholdAdjustment.NEXT and on_final_threshold:
                        model_threshold_upper_bound = threshold_upper_bound
                        model_threshold_lower_bound = threshold_midpoint
                    elif variance_entry.threshold_adjustment is ThresholdAdjustment.NEXT:
                        possible_next_threshold_names = [
                            possible_next_threshold_name
                            for possible_next_threshold_name in threshold_names[threshold_index + 1:]
                            if possible_next_threshold_name in row
                               and not_nan(row[possible_next_threshold_name])
                        ]

                        if possible_next_threshold_names:
                            next_threshold_value = row[possible_next_threshold_names[0]]
                        else:
                            next_threshold_value = threshold_upper_bound

                        if next_threshold_value == threshold_upper_bound:
                            model_threshold_upper_bound = next_threshold_value
                            model_threshold_lower_bound = threshold_midpoint
                        else:
                            model_threshold_lower_bound = threshold_upper_bound
                            model_threshold_upper_bound = (next_threshold_value + model_threshold_lower_bound) / 2.0
                    elif variance_entry.threshold_adjustment is ThresholdAdjustment.STAY:
                        if threshold_index == 0:
                            model_threshold_upper_bound = threshold_upper_bound
                            model_threshold_lower_bound = observed_lower_midpoint
                        else:
                            model_threshold_upper_bound = threshold_upper_bound
                            model_threshold_lower_bound = threshold_lower_bound
                    else:
                        # variance_entry.threshold_adjustment is ThresholdAdjustment.PREVIOUS
                        if threshold_index == 0:
                            model_threshold_upper_bound = observed_value
                            model_threshold_lower_bound = threshold_midpoint
                        elif threshold_index == 1:
                            previous_threshold_name = threshold_names[0]
                            previous_threshold_value = row[previous_threshold_name]
                            previous_threshold_value_is_nan = isnan(previous_threshold_value)

                            if previous_threshold_name not in row or previous_threshold_value_is_nan:
                                model_threshold_upper_bound = observed_value
                                model_threshold_lower_bound = threshold_midpoint
                            else:
                                model_threshold_upper_bound = row[previous_threshold_name]
                                model_threshold_lower_bound = row[previous_threshold_name] / 2.0
                        else:
                            possible_thresholds = sorted([
                                threshold
                                for threshold in threshold_names[:threshold_index]
                                if threshold in row
                                   and not_nan(row[threshold])
                            ], reverse=True)

                            if len(possible_thresholds) > 1:
                                previous_threshold_name = possible_thresholds[0]
                                yet_more_previous_threshold_name = possible_thresholds[1]
                                previous_previous_value = row[yet_more_previous_threshold_name]
                                model_threshold_upper_bound = row[previous_threshold_name]
                                model_threshold_lower_bound = (model_threshold_upper_bound + previous_previous_value) / 2.0
                            elif possible_thresholds == 1:
                                previous_threshold_name = possible_thresholds[0]
                                model_threshold_upper_bound = row[previous_threshold_name]
                                model_threshold_lower_bound = model_threshold_upper_bound / 2.0
                            else:
                                model_threshold_upper_bound = observed_value
                                model_threshold_lower_bound = threshold_midpoint

                    new_value[value_column] = random_number_generator.uniform(
                        model_threshold_lower_bound,
                        model_threshold_upper_bound
                    )
                    previous_modelled_value = new_value[value_column]
                    break

        new_values.append(new_value)

    duration = datetime.now() - model_start_time
    print(f"{len(new_values)} generated for {location} in {duration}")
    return new_values


def run_model(
    control_and_thresholds: pandas.DataFrame,
    threshold_names: typing.Sequence[str],
    variance: Variance,
    time_column: str,
    location_column: str,
    value_column: str,
    random_number_generator: typing.Union[Random, int] = None
) -> pandas.DataFrame:
    if control_and_thresholds.index.names != [location_column]:
        control_and_thresholds = control_and_thresholds.reset_index().set_index(location_column)

    with multiprocessing.Pool() as process_pool:
        arguments = [
                [
                    group,
                    threshold_names,
                    variance,
                    group_details,
                    time_column,
                    location_column,
                    value_column,
                    random_number_generator
                ]
                for group_details, group in control_and_thresholds.groupby(location_column)  # type: typing.Tuple[str], pandas.DataFrame
            ]

        simulation_per_location = process_pool.starmap(
            calculate_values,
            arguments
        )

    combined_simulated_data = functools.reduce(
        lambda accumulated_data, new_data: accumulated_data + new_data,
        simulation_per_location
    )

    combined_frames = pandas.DataFrame(combined_simulated_data).set_index(location_column)
    return combined_frames